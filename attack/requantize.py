"""Stage 4: re-quantize perturbed embeddings into discrete codes ("swap-into-shipped").

Uses the authors' RQ-VAE (same one behind the shipped index) to map each victim's
*perturbed* embedding to its 4 image-code tokens (deterministic L2 argmin,
use_sk=False). Builds an ATTACKED index = a copy of the shipped index with ONLY
the victim rows replaced, so clean (=shipped) vs attacked differ by exactly the
victim's code -> a perfectly controlled comparison, no model re-finetune.

Emits the quantization diagnostics that ARE the scientific payload: did the code
change? how many tokens? did it move toward a popular item's prefix? -- reported
even when the attack fails to move the code (the quantization 'attenuator').
"""
import argparse
import os

import numpy as np
import torch

import _common as C
from models.rqvae import RQVAE  # index/models/rqvae.py


def load_rqvae(ckpt_path, in_dim, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=a.num_emb_list,
        e_dim=a.e_dim,
        layers=a.layers,
        dropout_prob=getattr(a, "dropout_prob", 0.0),
        bn=getattr(a, "bn", False),
        loss_type=getattr(a, "loss_type", "mse"),
        quant_loss_weight=getattr(a, "quant_loss_weight", 1.0),
        kmeans_init=getattr(a, "kmeans_init", True),
        kmeans_iters=getattr(a, "kmeans_iters", 100),
        sk_epsilons=a.sk_epsilons,
        sk_iters=getattr(a, "sk_iters", 50),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), len(a.num_emb_list)


@torch.no_grad()
def quantize(model, vecs, n_levels, device):
    """vecs: (N, dim) np array -> list of N token-lists like ['<A_x>','<B_y>',...]."""
    x = torch.tensor(np.asarray(vecs), dtype=torch.float32, device=device)
    indices, _ = model.get_indices(x, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    out = []
    for row in indices:
        out.append([C.IMAGE_PREFIX[i].format(int(v)) for i, v in enumerate(row[:n_levels])])
    return out


def prefix_match(a_tokens, b_tokens):
    """Number of leading code tokens that match (0..len)."""
    n = 0
    for x, y in zip(a_tokens, b_tokens):
        if x == y:
            n += 1
        else:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--ckpt_path", required=True,
                    help="RQ-VAE ckpt, e.g. index/log/Instruments/ViT-L-14_256/best_collision_model.pth")
    ap.add_argument("--adv_emb", required=True, help="adv_emb_eps{E}.npz from clip_attack")
    ap.add_argument("--clean_emb", default=os.path.join(C.ARTIFACT_DIR, "clean_emb.npz"))
    ap.add_argument("--n_popular", type=int, default=20)
    ap.add_argument("--out_index", default=None)
    ap.add_argument("--out_diag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    ddir = C.ddir(args.data_path, args.dataset)
    emb_npy = os.path.join(ddir, f"{args.dataset}.emb-ViT-L-14.npy")
    shipped_index_path = os.path.join(ddir, f"{args.dataset}.index_vitemb.json")
    inter_json = os.path.join(ddir, f"{args.dataset}.inter.json")
    in_dim = int(np.load(emb_npy).shape[1])

    etag = os.path.basename(args.adv_emb).replace("adv_emb_", "").replace(".npz", "")  # e.g. eps16
    out_index = args.out_index or os.path.join(C.ARTIFACT_DIR, f"index_vitemb_ATTACKED_{etag}.json")
    out_diag = args.out_diag or os.path.join(C.ARTIFACT_DIR, f"requant_diag_{etag}.json")

    model, n_levels = load_rqvae(args.ckpt_path, in_dim, device)
    shipped = C.load_index_json(shipped_index_path)
    popular = C.top_popular(inter_json, k=args.n_popular)
    popular_codes = [shipped[str(p)] for p in popular]

    adv_npz = np.load(args.adv_emb)
    clean_npz = np.load(args.clean_emb) if os.path.exists(args.clean_emb) else None
    vids = list(adv_npz.files)

    adv_codes = quantize(model, [adv_npz[v] for v in vids], n_levels, device)
    clean_regen = (quantize(model, [clean_npz[v] for v in vids], n_levels, device)
                   if clean_npz is not None else [None] * len(vids))

    attacked = dict(shipped)  # shallow copy; we only replace victim rows
    diag = {}
    n_changed = n_regen_match = 0
    for vid, adv_t, regen_t in zip(vids, adv_codes, clean_regen):
        shipped_t = shipped[str(vid)]
        attacked[str(vid)] = adv_t  # swap-into-shipped
        changed = adv_t != shipped_t
        n_changed += int(changed)
        regen_ok = (regen_t == shipped_t) if regen_t is not None else None
        if regen_ok:
            n_regen_match += 1
        pm_pop = max(prefix_match(adv_t, pc) for pc in popular_codes)
        pm_pop_clean = max(prefix_match(shipped_t, pc) for pc in popular_codes)
        diag[str(vid)] = {
            "shipped": shipped_t, "attacked": adv_t, "clean_regen": regen_t,
            "code_changed": changed,
            "n_tokens_changed": sum(int(x != y) for x, y in zip(adv_t, shipped_t)),
            "clean_regen_matches_shipped": regen_ok,
            "prefix_match_to_popular_attacked": pm_pop,
            "prefix_match_to_popular_clean": pm_pop_clean,
            "moved_toward_popular": pm_pop > pm_pop_clean,
        }

    C.dump_json(attacked, out_index)
    C.dump_json(diag, out_diag)

    n = len(vids)
    print(f"[{etag}] re-quantized {n} victims")
    print(f"  code-change-rate           : {n_changed}/{n} = {n_changed/max(n,1):.1%}")
    print(f"  moved toward popular prefix: "
          f"{sum(d['moved_toward_popular'] for d in diag.values())}/{n}")
    if clean_npz is not None:
        print(f"  clean-regen == shipped     : {n_regen_match}/{n} = {n_regen_match/max(n,1):.1%}"
              f"   (consistency check; want high)")
    print(f"  -> attacked index: {out_index}")
    print(f"  -> diagnostics   : {out_diag}")


if __name__ == "__main__":
    main()
