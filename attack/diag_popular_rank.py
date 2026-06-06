"""Diagnostic: where do the POPULAR items rank in an arbitrary user's list?

Hypothesis (user's): popular items do NOT rank high for a random user -> which
would explain why pushing an item toward the popular centroid can't promote it
(MQL4GRec ranks by history-fit, not global popularity).

For sampled users, build a candidate pool = {the user's true positive} +
{top-N popular} + {N random}, score it with the CLEAN model (image / text /
ensemble), and report the average rank of each GROUP (positive vs popular vs
random), per-popular-item mean rank, and a few concrete user examples.

No attack here -- pure measurement. Run from attack/:
  python diag_popular_rank.py --ckpt_path ../log/Instruments --n_users 100
"""
import argparse
import os
import random

import numpy as np

import _common as C
from score import Scorer


def ranks_of(scores):
    """rank[i] = position of candidate i when sorted by score desc (1 = best)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rank = [0] * len(scores)
    for pos, idx in enumerate(order):
        rank[idx] = pos + 1
    return rank


def summarize(rank_list, pool):
    a = np.array(rank_list, dtype=np.float64)
    return (f"mean {a.mean():5.1f} / {pool}  | median {np.median(a):4.0f} "
            f"| top10 {np.mean(a <= 10):.2f} | top1 {np.mean(a == 1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--ckpt_path", default=os.path.join(C.REPO_ROOT, "log", "Instruments"))
    ap.add_argument("--n_users", type=int, default=100)
    ap.add_argument("--n_popular", type=int, default=20)
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max_his_len", type=int, default=20)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ddir = C.ddir(args.data_path, args.dataset)
    inter_path = os.path.join(ddir, f"{args.dataset}.inter.json")
    inter = C.load_json(inter_path)
    text_index = C.load_index_json(os.path.join(ddir, f"{args.dataset}.index_lemb.json"))
    img_index = C.load_index_json(os.path.join(ddir, f"{args.dataset}.index_vitemb.json"))
    n_items = len(img_index)
    popular = C.top_popular(inter_path, k=args.n_popular)
    counts = C.item_popularity(inter_path)
    pop_set = set(popular)

    rng = random.Random(args.seed)
    users = [u for u, it in inter.items() if len(it) >= 2]
    rng.shuffle(users)
    users = users[:args.n_users]

    scorer = Scorer(args.ckpt_path, device=args.device, max_his_len=args.max_his_len)

    agg = {ch: {"pos": [], "pop": [], "rand": []} for ch in ("image", "ensemble")}
    pop_item_ranks = {p: [] for p in popular}
    examples = []
    pool_size = 0

    for ui, u in enumerate(users):
        items = [int(i) for i in inter[u]]
        pos, history, seen = items[-1], items[:-1], set(items)
        pop_for_user = [p for p in popular if p != pos]
        rnd = []
        guard = 0
        while len(rnd) < args.n_random and guard < args.n_random * 50:
            c = rng.randrange(n_items)
            guard += 1
            if c not in seen and c not in pop_set and c != pos and c not in rnd:
                rnd.append(c)

        cands = [pos] + pop_for_user + rnd
        groups = ["pos"] + ["pop"] * len(pop_for_user) + ["rand"] * len(rnd)
        pool_size = len(cands)

        img_hist = scorer.history_input(history, img_index)
        txt_hist = scorer.history_input(history, text_index)
        img_s = scorer.score(img_hist, [C.code_str(img_index[str(c)]) for c in cands])
        txt_s = scorer.score(txt_hist, [C.code_str(text_index[str(c)]) for c in cands])
        ens = [(a + b) / 2 for a, b in zip(img_s, txt_s)]

        for ch, sc in (("image", img_s), ("ensemble", ens)):
            r = ranks_of(sc)
            for idx, g in enumerate(groups):
                agg[ch][g].append(r[idx])
            if ch == "ensemble":
                for j, p in enumerate(pop_for_user):
                    pop_item_ranks[p].append(r[1 + j])
        if ui < 5:
            r = ranks_of(ens)
            examples.append((u, r[0], [(pop_for_user[j], r[1 + j]) for j in range(min(6, len(pop_for_user)))]))

    print(f"\nPool = 1 positive + {args.n_popular} popular + {args.n_random} random "
          f"= ~{pool_size} candidates; users = {len(users)}")
    for ch in ("image", "ensemble"):
        print(f"\n[{ch}] average rank by GROUP (1 = best):")
        print(f"  true positive : {summarize(agg[ch]['pos'], pool_size)}")
        print(f"  POPULAR items : {summarize(agg[ch]['pop'], pool_size)}")
        print(f"  random items  : {summarize(agg[ch]['rand'], pool_size)}")
    print(f"\n(For reference: a uniformly-random item would average ~{(pool_size+1)/2:.1f}, top10 ~{10/pool_size:.2f})")

    print("\nPer-popular-item mean ENSEMBLE rank (sorted by popularity):")
    for p in popular:
        a = np.array(pop_item_ranks[p]) if pop_item_ranks[p] else np.array([float('nan')])
        print(f"  item {p:5d}: mean rank {a.mean():5.1f} | top10 {np.mean(a <= 10):.2f} "
              f"| {counts.get(p, 0)} interactions  (n={len(pop_item_ranks[p])})")

    print("\nConcrete examples (ensemble ranks):")
    for u, pr, pops in examples:
        s = ", ".join(f"item{pid}:#{rk}" for pid, rk in pops)
        print(f"  user {u}: their TRUE next item #{pr};  popular items -> {s}")


if __name__ == "__main__":
    main()
