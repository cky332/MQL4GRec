"""Shared helpers + sys.path bootstrap for the MQL4GRec attack toolkit.

This package ports the MLLM-MSR pixel-PGD item-promotion attack onto MQL4GRec.
Because MQL4GRec freezes the CLIP embedding into *discrete codes* before the
recommender sees it, the attack only matters if a small pixel perturbation
flips the quantized code toward a popular item's code region. Every stage here
is built to *measure* that, not assume it.

Run order:  sample_tasks -> download_subset -> clip_attack -> requantize -> run_eval
(orchestrated by attack/run_attack.sh).
"""
import os
import sys
import json
from collections import Counter

# --- make the repo's vendored modules importable from attack/ -----------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Order matters: data_process first so `from clip import clip` resolves to the
# vendored CLIP (data_process/clip), not a pip-installed `clip`.
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "index"),
           os.path.join(REPO_ROOT, "data_process")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default dataset layout (override via CLI where exposed).
DATA_DIR = os.path.join(REPO_ROOT, "data", "Instruments")
DATASET = "Instruments"
ARTIFACT_DIR = os.path.join(REPO_ROOT, "attack", "artifacts")

# Image-code token prefixes (uppercase), matching generate_indices_distance.py.
IMAGE_PREFIX = ["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>", "<E_{}>"]


def ddir(data_path=None, dataset=None):
    """Resolve the per-dataset directory (e.g. ./data/Instruments)."""
    data_path = data_path or os.path.join(REPO_ROOT, "data")
    dataset = dataset or DATASET
    return os.path.join(data_path, dataset)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_id_maps(item2id_path):
    """Return (id2asin, asin2id). item2id lines are '<ASIN>\\t<int id>'."""
    id2asin, asin2id = {}, {}
    with open(item2id_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            asin, idx = line.split("\t")
            id2asin[int(idx)] = asin
            asin2id[asin] = int(idx)
    return id2asin, asin2id


def load_index_json(path):
    """item-id (str) -> list of 4 code tokens, e.g. ['<A_154>','<B_106>',...]."""
    return load_json(path)


def code_str(tokens):
    """Join a list of code tokens into the single string the model consumes."""
    return "".join(tokens)


def item_popularity(inter_json_path):
    """Counter: item-id (int) -> number of interactions across all users."""
    inters = load_json(inter_json_path)
    counts = Counter()
    for _user, items in inters.items():
        counts.update(int(i) for i in items)
    return counts


def top_popular(inter_json_path, k=20, exclude=None):
    """Return the top-k most-interacted item ids (ints), optionally excluding some."""
    exclude = set(exclude or [])
    counts = item_popularity(inter_json_path)
    ranked = [i for i, _c in counts.most_common() if i not in exclude]
    return ranked[:k]


def build_tasks(inter_json_path, num_tasks, num_neg=20, seed=2024,
                n_items=None, min_hist=1):
    """Sample MLLM-MSR-style (user, positive, negatives) tasks via leave-one-out.

    positive = the user's last item (the held-out target). history = items[:-1].
    The positive item is the per-task 'victim' whose cover we attack. Negatives
    are random items not in the user's interaction set.
    """
    import random
    rng = random.Random(seed)
    inters = load_json(inter_json_path)

    if n_items is None:
        n_items = 1 + max(int(i) for items in inters.values() for i in items)

    users = [u for u, items in inters.items() if len(items) >= min_hist + 1]
    rng.shuffle(users)

    tasks = []
    for u in users:
        if len(tasks) >= num_tasks:
            break
        items = [int(i) for i in inters[u]]
        pos = items[-1]
        history = items[:-1]
        seen = set(items)
        negs = []
        guard = 0
        while len(negs) < num_neg and guard < num_neg * 50:
            cand = rng.randrange(n_items)
            guard += 1
            if cand not in seen and cand not in negs:
                negs.append(cand)
        if len(negs) < num_neg:
            continue
        tasks.append({"user": u, "pos": pos, "history": history, "negs": negs})
    return tasks


def centroid_from_embeddings(emb_npy_path, item_ids):
    """Mean CLIP embedding (the 'popular centroid') over the given item ids."""
    import numpy as np
    emb = np.load(emb_npy_path)
    rows = emb[np.array(item_ids, dtype=np.int64)].astype(np.float32)
    return rows.mean(axis=0)


def coldest_items(inter_json_path, n, n_items):
    """Return the n least-interacted item ids (incl. items with 0 interactions).

    These are the realistic 'promotion' victims: obscure products an attacker
    would want to push up. Ties broken by item id for determinism.
    """
    counts = item_popularity(inter_json_path)
    ranked = sorted(range(n_items), key=lambda i: (counts.get(i, 0), i))
    return ranked[:n]


def build_promote_tasks(inter_json_path, targets, num_tasks, num_neg=20,
                        seed=2024, n_items=None, include_true_positive=True):
    """Sample PROMOTION tasks: for each target item T (an item the user did NOT
    interact with), build (user, pos=T, history, negs) so that downstream code
    attacks T and ranks T among [T] + negs.

    negs = [the user's genuine held-out next item] + random non-interacted items,
    so we can see whether the promoted T can outrank truly relevant items.
    The schema matches build_tasks(), so clip_attack/requantize/run_eval are reused
    unchanged (they treat 'pos' as the item to attack & rank).
    """
    import random
    rng = random.Random(seed)
    inters = load_json(inter_json_path)
    if n_items is None:
        n_items = 1 + max(int(i) for items in inters.values() for i in items)

    users = [u for u, items in inters.items() if len(items) >= 2]
    per = max(1, num_tasks // max(len(targets), 1))

    tasks = []
    for T in targets:
        cand_users = [u for u in users if T not in {int(i) for i in inters[u]}]
        rng.shuffle(cand_users)
        taken = 0
        for u in cand_users:
            if taken >= per:
                break
            items = [int(i) for i in inters[u]]
            seen = set(items)
            history = items[:-1]
            negs = [items[-1]] if include_true_positive else []  # genuine competitor
            guard = 0
            while len(negs) < num_neg and guard < num_neg * 50:
                c = rng.randrange(n_items)
                guard += 1
                if c not in seen and c != T and c not in negs:
                    negs.append(c)
            if len(negs) < num_neg:
                continue
            tasks.append({"user": u, "pos": T, "history": history, "negs": negs})
            taken += 1
    return tasks
