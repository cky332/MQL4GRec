"""Stage 4-alt: WHITE-BOX CODE HIJACK (ceiling of code-forging attacks).

Directly set each cold target's image code = a chosen top-popular item's code
(default item 29, which the diagnostic showed ranks ~#3.4 / top10 97%). This
tests whether a target can inherit the popular item's high rank if the attacker
can forge the discrete code -- the attack surface the centroid attack could not
reach. No CLIP / PGD / RQ-VAE needed: pure index construction.

Output (TAG = hijack<P>):
  artifacts/index_vitemb_ATTACKED_<TAG>.json  (shipped index, target rows -> P's code)
  artifacts/hijack_diag_<TAG>.json            (so run_eval prints the breakdown)
"""
import argparse
import os

import _common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--hijack_id", type=int, default=29,
                    help="popular item whose code is copied onto every target")
    ap.add_argument("--victims", default=os.path.join(C.ARTIFACT_DIR, "victims.json"))
    ap.add_argument("--out_index", default=None)
    ap.add_argument("--out_diag", default=None)
    args = ap.parse_args()

    ddir = C.ddir(args.data_path, args.dataset)
    shipped_index = os.path.join(ddir, f"{args.dataset}.index_vitemb.json")
    shipped = C.load_index_json(shipped_index)

    hijack_code = shipped[str(args.hijack_id)]
    victims = C.load_json(args.victims)

    tag = f"hijack{args.hijack_id}"
    out_index = args.out_index or os.path.join(C.ARTIFACT_DIR, f"index_vitemb_ATTACKED_{tag}.json")
    out_diag = args.out_diag or os.path.join(C.ARTIFACT_DIR, f"hijack_diag_{tag}.json")

    attacked = dict(shipped)
    diag = {}
    for vid in victims:
        orig = shipped[str(vid)]
        attacked[str(vid)] = list(hijack_code)          # forge target's code = popular code
        diag[str(vid)] = {
            "shipped": orig, "attacked": list(hijack_code), "hijack_id": args.hijack_id,
            "code_changed": hijack_code != orig,
            "n_tokens_changed": sum(int(x != y) for x, y in zip(hijack_code, orig)),
            "moved_toward_popular": True,
            "prefix_match_to_popular_attacked": len(hijack_code),  # it IS a popular code
        }

    C.dump_json(attacked, out_index)
    C.dump_json(diag, out_diag)
    print(f"Hijacked {len(victims)} target items -> item {args.hijack_id}'s code "
          f"{''.join(hijack_code)}")
    print(f"  -> attacked index: {out_index}")
    print(f"  -> diagnostics   : {out_diag}")


if __name__ == "__main__":
    main()
