"""Stage 3: the attack -- two modes, two targets.

  --mode pixel     (faithful MLLM-MSR port): PGD in PIXEL space (L_inf <= eps) on
                   the SAME public CLIP ViT-L/14. Needs raw covers.
  --mode embedding (fallback; no images/CLIP): PGD on the stored CLIP embedding
                   under an L2 budget (||delta|| <= rho * ||x||).

  --target_id N    push toward a SINGLE item N's embedding (e.g. a popular item)
                   instead of the popular CENTROID (mean of top-N popular). This
                   bridges 'centroid attack (fails)' and 'code hijack (succeeds)':
                   getting close enough to one popular item's embedding may quantize
                   to that item's actual code.

Outputs (TAG = eps<N>/emb<N>, optionally +t<target_id>):
  artifacts/adv_emb_<TAG>.npz, artifacts/clean_emb.npz, artifacts/attack_diag_<TAG>.json
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import _common as C


# --------------------------- pixel mode (CLIP PGD) ----------------------------
def load_clip(device, model_cache_dir=None):
    from clip import clip  # vendored data_process/clip; identical to clip_feature.py
    from torchvision.transforms import Compose
    model, preprocess = clip.load("ViT-L/14", device=device, download_root=model_cache_dir)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    geo = Compose(preprocess.transforms[:-1])           # everything but Normalize -> [0,1]
    norm = preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device=device).view(3, 1, 1)
    std = torch.tensor(norm.std, device=device).view(3, 1, 1)
    return model, geo, mean, std


def encode(model, x_pix, mean, std):
    return model.encode_image(((x_pix - mean) / std).unsqueeze(0)).float().squeeze(0)


def pgd_pixel(model, x0, target, mean, std, eps, alpha, steps):
    tdir = target / target.norm()
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(steps):
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        emb = encode(model, x, mean, std)
        loss = 1.0 - F.cosine_similarity(emb.unsqueeze(0), tdir.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
            delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
        delta.requires_grad_(True)
    with torch.no_grad():
        x_adv = torch.clamp(x0 + delta, 0.0, 1.0)
        return x_adv.detach(), encode(model, x_adv, mean, std).detach(), encode(model, x0, mean, std).detach()


def save_image(x_pix, path):
    from PIL import Image
    arr = x_pix.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


# ----------------------- embedding mode (no images) ---------------------------
def pgd_embedding(x0, target, rho, steps, alpha_frac=0.1):
    """PGD on the stored embedding; ||delta||_2 <= rho * ||x0||."""
    tdir = target / target.norm()
    budget = rho * x0.norm()
    alpha = alpha_frac * budget
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(steps):
        x = x0 + delta
        loss = 1.0 - F.cosine_similarity(x.unsqueeze(0), tdir.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * grad / (grad.norm() + 1e-12)
            n = delta.norm()
            if n > budget:
                delta = delta * (budget / n)
        delta.requires_grad_(True)
    return (x0 + delta).detach()


def _diag(clean_emb, adv_emb, target, linf=None, l2=None):
    cc = F.cosine_similarity(clean_emb.unsqueeze(0), target.unsqueeze(0)).item()
    ca = F.cosine_similarity(adv_emb.unsqueeze(0), target.unsqueeze(0)).item()
    d = {"cos_clean": cc, "cos_adv": ca, "cos_gain": ca - cc}
    if linf is not None:
        d["linf"] = linf
    if l2 is not None:
        d["l2_rel"] = l2
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--mode", choices=["pixel", "embedding"], default="pixel")
    ap.add_argument("--victim_images", default=os.path.join(C.ARTIFACT_DIR, "victim_images.json"))
    ap.add_argument("--victims", default=os.path.join(C.ARTIFACT_DIR, "victims.json"))
    ap.add_argument("--eps", type=float, default=16/255, help="[pixel] L_inf budget")
    ap.add_argument("--alpha", type=float, default=None, help="[pixel] PGD step (default eps/4)")
    ap.add_argument("--emb_rho", type=float, default=0.2, help="[embedding] L2 budget as frac of ||x||")
    ap.add_argument("--target_id", type=int, default=None,
                    help="push toward THIS single item's embedding instead of the popular centroid")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n_popular", type=int, default=20)
    ap.add_argument("--model_cache_dir", default=None)
    ap.add_argument("--save_adv_images", action="store_true")
    ap.add_argument("--out_dir", default=C.ARTIFACT_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    ddir = C.ddir(args.data_path, args.dataset)
    emb_npy = os.path.join(ddir, f"{args.dataset}.emb-ViT-L-14.npy")
    inter_json = os.path.join(ddir, f"{args.dataset}.inter.json")

    popular = C.top_popular(inter_json, k=args.n_popular)
    if args.target_id is not None:
        target_np = np.load(emb_npy)[args.target_id].astype(np.float32)
        print(f"Target = single item {args.target_id}'s embedding")
    else:
        target_np = C.centroid_from_embeddings(emb_npy, popular)
        print(f"Target = centroid of top-{args.n_popular} popular: {popular}")
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    suffix = f"t{args.target_id}" if args.target_id is not None else ""

    adv_embs, clean_embs, diag = {}, {}, {}

    if args.mode == "pixel":
        tag = f"eps{round(args.eps * 255)}{suffix}"
        alpha = args.alpha if args.alpha is not None else args.eps / 4.0
        model, geo, mean, std = load_clip(device, args.model_cache_dir)
        from PIL import Image
        victim_images = C.load_json(args.victim_images)
        img_out = os.path.join(args.out_dir, f"adv_images_{tag}", args.dataset)
        if args.save_adv_images:
            os.makedirs(img_out, exist_ok=True)
        for vid, path in victim_images.items():
            try:
                x0 = geo(Image.open(path)).to(device)
            except Exception as e:
                print(f"  victim {vid}: cannot load image ({e}); skipping")
                continue
            x_adv, adv_emb, clean_emb = pgd_pixel(model, x0, target, mean, std,
                                                  args.eps, alpha, args.steps)
            adv_embs[str(vid)] = adv_emb.cpu().numpy()
            clean_embs[str(vid)] = clean_emb.cpu().numpy()
            diag[str(vid)] = _diag(clean_emb, adv_emb, target,
                                   linf=(x_adv - x0).abs().max().item())
            if args.save_adv_images:
                save_image(x_adv, os.path.join(img_out, f"{vid}_adv.jpg"))
                save_image(torch.clamp(x0 + (x_adv - x0) * 10, 0, 1),
                           os.path.join(img_out, f"{vid}_pert_x10.jpg"))
    else:  # embedding mode
        tag = f"emb{round(args.emb_rho * 100)}{suffix}"
        emb = np.load(emb_npy)
        victims = C.load_json(args.victims)
        for vid in victims:
            x0 = torch.tensor(emb[int(vid)].astype(np.float32), device=device)
            adv_emb = pgd_embedding(x0, target, args.emb_rho, args.steps)
            adv_embs[str(vid)] = adv_emb.cpu().numpy()
            clean_embs[str(vid)] = x0.cpu().numpy()
            diag[str(vid)] = _diag(x0, adv_emb, target,
                                   l2=(adv_emb - x0).norm().item() / x0.norm().item())

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f"adv_emb_{tag}.npz"), **adv_embs)
    np.savez(os.path.join(args.out_dir, "clean_emb.npz"), **clean_embs)
    C.dump_json(diag, os.path.join(args.out_dir, f"attack_diag_{tag}.json"))

    if diag:
        cc = np.mean([d["cos_clean"] for d in diag.values()])
        ca = np.mean([d["cos_adv"] for d in diag.values()])
        tgt = f"item {args.target_id}" if args.target_id is not None else "centroid"
        print(f"[{tag}] {args.mode} attack on {len(diag)} victims | "
              f"mean cos->{tgt} {cc:.3f} -> {ca:.3f} (gain {ca - cc:+.3f})")


if __name__ == "__main__":
    main()
