"""Stage 2: download cover images for the victim items only.

The repo ships only CLIP embeddings, not raw images. PGD needs pixels, but ONLY
for the items we attack (the victims). The popular 'centroid' is computed later
straight from the shipped .npy embeddings, so popular-item images are NOT needed.

Inputs : attack/artifacts/victims.json (from sample_tasks.py), the dataset's
         item2id, and the Amazon-2018 metadata gz (meta_<FullName>.json.gz).
Output : <image_out>/<dataset>/<item_id>.jpg  and  artifacts/victim_images.json
         ({str item id -> local jpg path}).
"""
import argparse
import gzip
import json
import os

import requests

import _common as C

try:
    from utils import amazon18_dataset2fullname  # data_process/utils.py
except Exception:
    # Minimal fallback so we don't hard-depend on transformers just to download.
    amazon18_dataset2fullname = {
        "Instruments": "Musical_Instruments", "Arts": "Arts_Crafts_and_Sewing",
        "Games": "Video_Games", "Pet": "Pet_Supplies",
        "Cell": "Cell_Phones_and_Accessories", "Automotive": "Automotive",
        "Tools": "Tools_and_Home_Improvement", "Toys": "Toys_and_Games",
        "Sports": "Sports_and_Outdoors",
    }


def is_valid_jpg(path):
    if not os.path.exists(path) or os.path.getsize(path) < 2:
        return False
    with open(path, "rb") as f:
        f.seek(os.path.getsize(path) - 2)
        return f.read() == b"\xff\xd9"


def download_image(url, save_path):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"  download failed: {e}")
        return False


def collect_urls(meta_file, needed_asins):
    """Stream the metadata gz and grab imageURLHighRes for the needed ASINs only."""
    asin2urls = {}
    needed = set(needed_asins)
    with gzip.open(meta_file, "r") as fp:
        for line in fp:
            data = json.loads(line)
            asin = data.get("asin")
            if asin in needed:
                asin2urls[asin] = data.get("imageURLHighRes", []) or []
                if len(asin2urls) == len(needed):
                    break
    return asin2urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(C.REPO_ROOT, "data"))
    ap.add_argument("--dataset", default="Instruments")
    ap.add_argument("--meta_data_path", required=True,
                    help="dir containing meta_<FullName>.json.gz")
    ap.add_argument("--image_out", default=os.path.join(C.ARTIFACT_DIR, "images"))
    ap.add_argument("--victims_file", default=os.path.join(C.ARTIFACT_DIR, "victims.json"))
    args = ap.parse_args()

    ddir = C.ddir(args.data_path, args.dataset)
    id2asin, _ = C.load_id_maps(os.path.join(ddir, f"{args.dataset}.item2id"))
    victims = C.load_json(args.victims_file)
    victim_asins = {vid: id2asin[int(vid)] for vid in victims}

    full = amazon18_dataset2fullname[args.dataset]
    meta_file = os.path.join(args.meta_data_path, f"meta_{full}.json.gz")
    print(f"Reading {len(victims)} victim ASINs from {meta_file} ...")
    asin2urls = collect_urls(meta_file, victim_asins.values())

    out_dir = os.path.join(args.image_out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    victim_images, missing = {}, []
    for vid in victims:
        asin = victim_asins[vid]
        save_path = os.path.join(out_dir, f"{vid}.jpg")
        if is_valid_jpg(save_path):
            victim_images[str(vid)] = save_path
            continue
        ok = False
        for url in asin2urls.get(asin, []):
            if download_image(url, save_path) and is_valid_jpg(save_path):
                ok = True
                break
        if ok:
            victim_images[str(vid)] = save_path
        else:
            missing.append(vid)

    C.dump_json(victim_images, os.path.join(C.ARTIFACT_DIR, "victim_images.json"))
    print(f"Downloaded {len(victim_images)} covers; {len(missing)} missing.")
    if missing:
        print(f"  missing victim ids (no usable image): {missing[:20]}"
              f"{' ...' if len(missing) > 20 else ''}")
        print("  NOTE: tasks whose victim has no image are skipped in run_eval.")


if __name__ == "__main__":
    main()
