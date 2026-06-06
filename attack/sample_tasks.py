"""Stage 1: sample the evaluation tasks and freeze them to disk.

Two task types:
  * standard (default): victim = each user's true next item (held-out positive).
    Measures robustness of a relevant item to code perturbation.
  * --promote: victim = a COLD target item the user did NOT interact with, ranked
    among [T] + (true positive + random non-interacted) negatives. Measures whether
    attacking T's image can PROMOTE it into the user's top-k.

The set of distinct victims (positives) is what download/clip_attack/requantize act on.
Output: attack/artifacts/tasks.json , attack/artifacts/victims.json
"""
import argparse
import os

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--num_tasks", type=int, default=200)
    ap.add_argument("--num_neg", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024)
    # promotion mode
    ap.add_argument("--promote", action="store_true",
                    help="attack COLD non-interacted target items instead of the user's positive")
    ap.add_argument("--n_targets", type=int, default=5,
                    help="[promote] number of target items to promote")
    ap.add_argument("--targets", type=str, default=None,
                    help="[promote] explicit comma-separated item ids (overrides --n_targets)")
    ap.add_argument("--out_dir", default=C.ARTIFACT_DIR)
    args = ap.parse_args()

    ddir = C.ddir(args.data_path, args.dataset)
    inter_json = os.path.join(ddir, f"{args.dataset}.inter.json")
    image_index = os.path.join(ddir, f"{args.dataset}.index_vitemb.json")
    n_items = len(C.load_index_json(image_index))

    if args.promote:
        if args.targets:
            targets = [int(x) for x in args.targets.split(",")]
        else:
            targets = C.coldest_items(inter_json, args.n_targets, n_items)
        tasks = C.build_promote_tasks(inter_json, targets, args.num_tasks,
                                      num_neg=args.num_neg, seed=args.seed, n_items=n_items)
        mode = "promote"
        print(f"Promotion mode. Target (cold) items: {targets}")
    else:
        tasks = C.build_tasks(inter_json, args.num_tasks, num_neg=args.num_neg,
                              seed=args.seed, n_items=n_items)
        mode = "standard"

    victims = sorted({t["pos"] for t in tasks})
    C.dump_json({"dataset": args.dataset, "seed": args.seed, "mode": mode,
                 "num_neg": args.num_neg, "tasks": tasks},
                os.path.join(args.out_dir, "tasks.json"))
    C.dump_json(victims, os.path.join(args.out_dir, "victims.json"))

    print(f"[{mode}] sampled {len(tasks)} tasks, {len(victims)} distinct victim items.")
    print(f"  -> {os.path.join(args.out_dir, 'tasks.json')}")
    print(f"  -> {os.path.join(args.out_dir, 'victims.json')}")


if __name__ == "__main__":
    main()
