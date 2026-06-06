"""Stage 5: the MLLM-MSR-mirror evaluation (1 positive + 20 negatives).

For every task, score the 21 candidates twice -- with the victim's CLEAN code and
with its ATTACKED code (only the victim differs) -- under two systems:
  * image-channel only  (where the attack acts; strongest effect)
  * ensemble            (mean of image + UNATTACKED text channel; shows dilution)
Reports avg rank, hit@10, NDCG@10 and a P(Yes)-analog, before vs after, plus the
quantization diagnostics (code-change-rate, moved-toward-popular) that explain why.
"""
import argparse
import math
import os

import numpy as np

import _common as C
from score import Scorer


def metrics(scores, target_idx=0, k=10):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank = order.index(target_idx) + 1
    hit = 1.0 if rank <= k else 0.0
    ndcg = (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
    s = np.array(scores, dtype=np.float64)
    s = s - s.max()
    p = np.exp(s)
    pyes = float(p[target_idx] / p.sum())
    return rank, hit, ndcg, pyes


def agg(rows):
    """rows: list of (rank, hit, ndcg, pyes) -> dict of means."""
    if not rows:
        return {"rank": float("nan"), "hit@10": float("nan"),
                "ndcg@10": float("nan"), "P(Yes)": float("nan"), "n": 0}
    a = np.array(rows, dtype=np.float64)
    return {"rank": a[:, 0].mean(), "hit@10": a[:, 1].mean(),
            "ndcg@10": a[:, 2].mean(), "P(Yes)": a[:, 3].mean(), "n": len(rows)}


def print_block(title, clean, att):
    print(f"\n=== {title}  (n={clean['n']}) ===")
    print(f"{'metric':10}{'clean':>10}{'attacked':>12}{'change':>12}")
    for m, better_down in [("rank", True), ("hit@10", False),
                           ("ndcg@10", False), ("P(Yes)", False)]:
        c, a = clean[m], att[m]
        d = a - c
        arrow = ""
        if not math.isnan(d):
            good = (d < 0) if better_down else (d > 0)
            arrow = "  (attack helps)" if good and abs(d) > 1e-9 else ""
        print(f"{m:10}{c:>10.4f}{a:>12.4f}{d:>+12.4f}{arrow}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--ckpt_path", default=os.path.join(C.REPO_ROOT, "log", "Instruments"),
                    help="finetuned recommender dir (T5 + tokenizer)")
    ap.add_argument("--tasks", default=os.path.join(C.ARTIFACT_DIR, "tasks.json"))
    ap.add_argument("--attacked_index", required=True,
                    help="index_vitemb_ATTACKED_eps*.json from requantize.py")
    ap.add_argument("--requant_diag", default=None,
                    help="requant_diag_eps*.json (for changed-subset breakdown)")
    ap.add_argument("--max_his_len", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ddir = C.ddir(args.data_path, args.dataset)
    text_index = C.load_index_json(os.path.join(ddir, f"{args.dataset}.index_lemb.json"))
    img_clean = C.load_index_json(os.path.join(ddir, f"{args.dataset}.index_vitemb.json"))
    img_att = C.load_index_json(args.attacked_index)
    tasks_obj = C.load_json(args.tasks)
    tasks = tasks_obj["tasks"]
    mode = tasks_obj.get("mode", "standard")
    diag = C.load_json(args.requant_diag) if args.requant_diag and os.path.exists(args.requant_diag) else {}

    # Only tasks whose victim was actually attacked (present in the attacked set).
    attackable = set(diag.keys()) if diag else None
    tasks = [t for t in tasks if attackable is None or str(t["pos"]) in attackable]

    scorer = Scorer(args.ckpt_path, device=args.device, max_his_len=args.max_his_len)

    res = {k: {"all": [], "changed": []} for k in
           ("img_clean", "img_att", "ens_clean", "ens_att")}

    for t in tasks:
        pos, negs, hist = t["pos"], t["negs"], t["history"]
        cands = [pos] + negs                       # target at index 0
        changed = bool(diag.get(str(pos), {}).get("code_changed", True))

        img_hist = scorer.history_input(hist, img_clean)
        txt_hist = scorer.history_input(hist, text_index)

        img_clean_cands = [C.code_str(img_clean[str(c)]) for c in cands]
        txt_cands = [C.code_str(text_index[str(c)]) for c in cands]

        img_clean_scores = scorer.score(img_hist, img_clean_cands)
        txt_scores = scorer.score(txt_hist, txt_cands)
        # Only the victim's image code changes under attack -> rescore index 0 only.
        pos_att_code = C.code_str(img_att[str(pos)])
        img_att_scores = list(img_clean_scores)
        img_att_scores[0] = scorer.score(img_hist, [pos_att_code])[0]

        ens_clean = [(i + t_) / 2 for i, t_ in zip(img_clean_scores, txt_scores)]
        ens_att = [(i + t_) / 2 for i, t_ in zip(img_att_scores, txt_scores)]

        for key, sc in (("img_clean", img_clean_scores), ("img_att", img_att_scores),
                        ("ens_clean", ens_clean), ("ens_att", ens_att)):
            m = metrics(sc, target_idx=0)
            res[key]["all"].append(m)
            if changed:
                res[key]["changed"].append(m)

    n = len(tasks)
    n_changed = sum(1 for t in tasks if bool(diag.get(str(t["pos"]), {}).get("code_changed", True)))
    print(f"\nEvaluated {n} tasks ({args.dataset}) | mode = {mode}.")
    if mode == "promote":
        print("  PROMOTION test: the attacked item is a COLD item the user did NOT interact with.")
        print("  'rank' = that target's rank among 21 (LOWER = more successfully promoted);")
        print("  clean rank should be high/bottom (irrelevant). 'attack helps' = target rose.")
    else:
        print("  STANDARD test: the attacked item is the user's true next item (already ranks well);")
        print("  here the attack can only hurt it -> measures robustness, not promotion.")
    if diag:
        moved = sum(1 for t in tasks if diag.get(str(t["pos"]), {}).get("moved_toward_popular"))
        print(f"  code changed by attack : {n_changed}/{n} = {n_changed/max(n,1):.1%}")
        print(f"  moved toward popular   : {moved}/{n} = {moved/max(n,1):.1%}")

    print_block("IMAGE channel only — ALL tasks",
                agg(res["img_clean"]["all"]), agg(res["img_att"]["all"]))
    print_block("ENSEMBLE (image+text) — ALL tasks",
                agg(res["ens_clean"]["all"]), agg(res["ens_att"]["all"]))
    if diag and n_changed:
        print_block("IMAGE channel only — tasks where code CHANGED",
                    agg(res["img_clean"]["changed"]), agg(res["img_att"]["changed"]))
        print_block("ENSEMBLE — tasks where code CHANGED",
                    agg(res["ens_clean"]["changed"]), agg(res["ens_att"]["changed"]))

    out = args.out or os.path.join(C.ARTIFACT_DIR,
                                   "eval_" + os.path.basename(args.attacked_index).replace(".json", "") + ".json")
    summary = {grp: {k: agg(res[k][grp]) for k in res} for grp in ("all", "changed")}
    summary["n_tasks"] = n
    summary["n_changed"] = n_changed
    C.dump_json(summary, out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
