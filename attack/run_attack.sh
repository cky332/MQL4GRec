#!/bin/bash
# ---------------------------------------------------------------------------
# End-to-end runner for the MLLM-MSR-style pixel-PGD attack on MQL4GRec.
#   sample_tasks -> download_subset -> clip_attack(eps sweep) -> requantize -> run_eval
#
# Prerequisites (see attack/README.md):
#   * a finetuned recommender at ./log/Instruments  (from scripts/run_local.sh)
#   * the authors' RQ-VAE ckpt at $RQVAE_CKPT
#   * Amazon-2018 metadata dir $META_DATA_PATH containing meta_<FullName>.json.gz
#
# Usage:
#   META_DATA_PATH=/path/to/amazon18/Metadata \
#   RQVAE_CKPT=index/log/Instruments/ViT-L-14_256/best_collision_model.pth \
#   bash attack/run_attack.sh
#
# Overrides: DATASET, NUM_TASKS, EPS_LIST (in /255 units), MODEL_CKPT, GPUS, STEPS
# ---------------------------------------------------------------------------
set -e
set -o pipefail
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1   # harmless; consistent with run_local.sh

cd "$(dirname "$0")"                          # run from attack/ (imports use _common)

DATASET=${DATASET:-Instruments}
DATA_PATH=${DATA_PATH:-../data}
NUM_TASKS=${NUM_TASKS:-200}
EPS_LIST=${EPS_LIST:-"16 32 64"}              # L_inf budgets in /255
STEPS=${STEPS:-30}
MODEL_CKPT=${MODEL_CKPT:-../log/$DATASET}
RQVAE_CKPT=${RQVAE_CKPT:-../index/log/$DATASET/ViT-L-14_256/best_collision_model.pth}
GPUS=${GPUS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPUS}

: "${META_DATA_PATH:?set META_DATA_PATH to the dir with meta_<FullName>.json.gz}"
[ -f "$RQVAE_CKPT" ] || { echo "ERROR: RQ-VAE ckpt not found: $RQVAE_CKPT"; exit 1; }
[ -d "$MODEL_CKPT" ] || { echo "ERROR: recommender dir not found: $MODEL_CKPT"; exit 1; }

echo "=== [1/4] sample tasks ==="
python sample_tasks.py --data_path "$DATA_PATH" --dataset "$DATASET" --num_tasks "$NUM_TASKS"

echo "=== [2/4] download victim covers ==="
python download_subset.py --data_path "$DATA_PATH" --dataset "$DATASET" \
    --meta_data_path "$META_DATA_PATH"

for E in $EPS_LIST; do
    EPS=$(python3 -c "print($E/255)")
    echo "=== [3/4] PGD attack (eps=$E/255) ==="
    python clip_attack.py --data_path "$DATA_PATH" --dataset "$DATASET" \
        --eps "$EPS" --steps "$STEPS" --save_adv_images

    echo "=== [3b] re-quantize (eps=$E/255) ==="
    python requantize.py --data_path "$DATA_PATH" --dataset "$DATASET" \
        --ckpt_path "$RQVAE_CKPT" --adv_emb "artifacts/adv_emb_eps${E}.npz"

    echo "=== [4/4] evaluate (eps=$E/255) ==="
    python run_eval.py --data_path "$DATA_PATH" --dataset "$DATASET" \
        --ckpt_path "$MODEL_CKPT" \
        --attacked_index "artifacts/index_vitemb_ATTACKED_eps${E}.json" \
        --requant_diag "artifacts/requant_diag_eps${E}.json"
done

echo "=== done. results in attack/artifacts/eval_*.json ==="
