"""Stage 1: sample the evaluation tasks and freeze them to disk.

Each task = (user, positive=victim item, 20 random negatives), leave-one-out.
The set of distinct positives is the list of 'victim' items whose covers the
later stages will download / perturb. Freezing tasks makes the whole pipeline
deterministic and lets download/attack/eval agree on the same victims.

Output: attack/artifacts/tasks.json
        attack/artifacts/victims.json   (sorted unique victim item ids)
"""
import argparse
import os

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--num_tasks", type=int, default=200,
                    help="number of (user, victim, 20-neg) tasks")
    ap.add_argument("--num_neg", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out_dir", default=C.ARTIFACT_DIR)
    args = ap.parse_args()

    ddir = C.ddir(args.data_path, args.dataset)
    inter_json = os.path.join(ddir, f"{args.dataset}.inter.json")
    image_index = os.path.join(ddir, f"{args.dataset}.index_vitemb.json")
    n_items = len(C.load_index_json(image_index))

    tasks = C.build_tasks(inter_json, args.num_tasks, num_neg=args.num_neg,
                          seed=args.seed, n_items=n_items)
    victims = sorted({t["pos"] for t in tasks})

    C.dump_json({"dataset": args.dataset, "seed": args.seed,
                 "num_neg": args.num_neg, "tasks": tasks},
                os.path.join(args.out_dir, "tasks.json"))
    C.dump_json(victims, os.path.join(args.out_dir, "victims.json"))

    print(f"Sampled {len(tasks)} tasks, {len(victims)} distinct victim items.")
    print(f"  -> {os.path.join(args.out_dir, 'tasks.json')}")
    print(f"  -> {os.path.join(args.out_dir, 'victims.json')}")


if __name__ == "__main__":
    main()
