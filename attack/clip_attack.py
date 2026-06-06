"""Stage 3: the faithful black-box PGD attack (the MLLM-MSR port).

For each victim image, run PGD in PIXEL space (L_inf <= eps) on the SAME public
CLIP ViT-L/14 the offline pipeline uses, pushing the image embedding toward the
'popular centroid' (mean CLIP embedding of the top-20 most-interacted items,
read straight from the shipped .npy). Transfer is exact because the target uses
this unmodified CLIP -- there is no surrogate gap.

Outputs (per eps):
  artifacts/adv_emb_eps{E}.npz    : {str id -> perturbed 768-d embedding}
  artifacts/clean_emb.npz         : {str id -> re-encoded clean embedding}
  artifacts/attack_diag_eps{E}.json : per-victim cosine(clean/adv, centroid), L_inf
Optionally saves adv covers (+ a x10-amplified perturbation view) for figures.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose

import _common as C
from clip import clip  # vendored data_process/clip -> identical to clip_feature.py


def load_clip(device, model_cache_dir=None):
    """Load the vendored CLIP ViT-L/14 in fp32 (stable PGD grads) and freeze it."""
    model, preprocess = clip.load("ViT-L/14", device=device, download_root=model_cache_dir)
    model = model.float().eval()          # force fp32 even on CUDA (CLIP loads fp16 there)
    for p in model.parameters():
        p.requires_grad_(False)
    # Split preprocess into geometry+ToTensor ([0,1] tensor) and the final Normalize.
    geo = Compose(preprocess.transforms[:-1])
    norm = preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device=device).view(3, 1, 1)
    std = torch.tensor(norm.std, device=device).view(3, 1, 1)
    return model, geo, mean, std


def encode(model, x_pix, mean, std):
    """x_pix: (3,H,W) in [0,1] -> (768,) embedding, differentiable wrt x_pix."""
    xn = (x_pix - mean) / std
    emb = model.encode_image(xn.unsqueeze(0)).float().squeeze(0)
    return emb


def pgd(model, x0, centroid, mean, std, eps, alpha, steps):
    """L_inf PGD pushing cos(emb, centroid) up. Returns (x_adv, adv_emb, clean_emb)."""
    centroid = centroid / centroid.norm()
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(steps):
        x = torch.clamp(x0 + delta, 0.0, 1.0)
        emb = encode(model, x, mean, std)
        loss = 1.0 - F.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
            delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0   # keep pixels valid
        delta.requires_grad_(True)
    with torch.no_grad():
        x_adv = torch.clamp(x0 + delta, 0.0, 1.0)
        adv_emb = encode(model, x_adv, mean, std)
        clean_emb = encode(model, x0, mean, std)
    return x_adv.detach(), adv_emb.detach(), clean_emb.detach()


def save_image(x_pix, path):
    arr = (x_pix.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--victim_images", default=os.path.join(C.ARTIFACT_DIR, "victim_images.json"))
    ap.add_argument("--eps", type=float, default=16/255, help="L_inf budget (pixel space)")
    ap.add_argument("--alpha", type=float, default=None, help="PGD step (default eps/4)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n_popular", type=int, default=20, help="centroid = mean of top-N popular")
    ap.add_argument("--model_cache_dir", default=None)
    ap.add_argument("--save_adv_images", action="store_true")
    ap.add_argument("--out_dir", default=C.ARTIFACT_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else args.eps / 4.0
    device = torch.device(args.device)

    ddir = C.ddir(args.data_path, args.dataset)
    emb_npy = os.path.join(ddir, f"{args.dataset}.emb-ViT-L-14.npy")
    inter_json = os.path.join(ddir, f"{args.dataset}.inter.json")

    # Popular centroid straight from shipped CLIP embeddings (no popular images needed).
    popular = C.top_popular(inter_json, k=args.n_popular)
    centroid_np = C.centroid_from_embeddings(emb_npy, popular)
    centroid = torch.tensor(centroid_np, dtype=torch.float32, device=device)
    print(f"Top-{args.n_popular} popular ids: {popular}")

    model, geo, mean, std = load_clip(device, args.model_cache_dir)
    victim_images = C.load_json(args.victim_images)

    adv_embs, clean_embs, diag = {}, {}, {}
    etag = f"{round(args.eps * 255)}"  # e.g. 16 for 16/255
    img_out = os.path.join(args.out_dir, f"adv_images_eps{etag}", args.dataset)
    if args.save_adv_images:
        os.makedirs(img_out, exist_ok=True)

    for vid, path in victim_images.items():
        try:
            x0 = geo(Image.open(path)).to(device)
        except Exception as e:
            print(f"  victim {vid}: cannot load image ({e}); skipping")
            continue
        x_adv, adv_emb, clean_emb = pgd(model, x0, centroid, mean, std,
                                        args.eps, alpha, args.steps)
        cos_clean = F.cosine_similarity(clean_emb.unsqueeze(0), centroid.unsqueeze(0)).item()
        cos_adv = F.cosine_similarity(adv_emb.unsqueeze(0), centroid.unsqueeze(0)).item()
        linf = (x_adv - x0).abs().max().item()
        adv_embs[str(vid)] = adv_emb.cpu().numpy()
        clean_embs[str(vid)] = clean_emb.cpu().numpy()
        diag[str(vid)] = {"cos_clean": cos_clean, "cos_adv": cos_adv,
                          "cos_gain": cos_adv - cos_clean, "linf": linf}
        if args.save_adv_images:
            save_image(x_adv, os.path.join(img_out, f"{vid}_adv.jpg"))
            amp = torch.clamp(x0 + (x_adv - x0) * 10.0, 0, 1)
            save_image(amp, os.path.join(img_out, f"{vid}_pert_x10.jpg"))

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f"adv_emb_eps{etag}.npz"), **adv_embs)
    np.savez(os.path.join(args.out_dir, "clean_emb.npz"), **clean_embs)
    C.dump_json(diag, os.path.join(args.out_dir, f"attack_diag_eps{etag}.json"))

    if diag:
        gains = [d["cos_gain"] for d in diag.values()]
        print(f"[eps={etag}/255] attacked {len(diag)} victims | "
              f"mean cos->centroid {np.mean([d['cos_clean'] for d in diag.values()]):.3f}"
              f" -> {np.mean([d['cos_adv'] for d in diag.values()]):.3f} "
              f"(mean gain {np.mean(gains):+.3f}) | max L_inf "
              f"{max(d['linf'] for d in diag.values()):.4f}")


if __name__ == "__main__":
    main()
