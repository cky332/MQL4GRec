#!/bin/bash
# ---------------------------------------------------------------------------
# End-to-end runner for the attack on MQL4GRec.
#
#   MODE=pixel     (default): sample_tasks -> download_subset -> clip_attack(PGD
#                  on covers, L_inf eps sweep) -> requantize -> run_eval.
#                  Needs raw images: set META_DATA_PATH (Amazon metadata dir).
#
#   MODE=embedding (no images / no metadata): sample_tasks -> clip_attack(PGD on
#                  the stored embedding, L2 rho sweep) -> requantize -> run_eval.
#
# Prerequisites (both modes):
#   * finetuned recommender at $MODEL_CKPT (./log/Instruments)
#   * RQ-VAE ckpt at $RQVAE_CKPT
#
# Examples:
#   # image-free, runnable right now:
#   RQVAE_CKPT=index/log/Instruments/ViT-L-14_256/best_collision_model.pth \
#   MODE=embedding bash attack/run_attack.sh
#
#   # faithful pixel attack (needs Amazon metadata):
#   RQVAE_CKPT=...best_collision_model.pth META_DATA_PATH=/path/amazon18/Metadata \
#   bash attack/run_attack.sh
#
# Overrides: MODE, DATASET, NUM_TASKS, EPS_LIST (/255), RHO_LIST (%), STEPS,
#            MODEL_CKPT, GPUS
# ---------------------------------------------------------------------------
set -e
set -o pipefail
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

cd "$(dirname "$0")"

MODE=${MODE:-pixel}
DATASET=${DATASET:-Instruments}
DATA_PATH=${DATA_PATH:-../data}
NUM_TASKS=${NUM_TASKS:-200}
EPS_LIST=${EPS_LIST:-"16 32 64"}        # pixel: L_inf in /255
RHO_LIST=${RHO_LIST:-"10 20 30 50"}     # embedding: L2 budget in % of ||x||
STEPS=${STEPS:-30}
MODEL_CKPT=${MODEL_CKPT:-../log/$DATASET}
RQVAE_CKPT=${RQVAE_CKPT:-../index/log/$DATASET/ViT-L-14_256/best_collision_model.pth}
GPUS=${GPUS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPUS}

[ -f "$RQVAE_CKPT" ] || { echo "ERROR: RQ-VAE ckpt not found: $RQVAE_CKPT"; exit 1; }
[ -d "$MODEL_CKPT" ] || { echo "ERROR: recommender dir not found: $MODEL_CKPT"; exit 1; }

COMMON="--data_path $DATA_PATH --dataset $DATASET"

echo "=== [1] sample tasks ==="
python sample_tasks.py $COMMON --num_tasks "$NUM_TASKS"

if [ "$MODE" = "pixel" ]; then
    : "${META_DATA_PATH:?MODE=pixel needs META_DATA_PATH (dir with meta_<FullName>.json.gz)}"
    echo "=== [2] download victim covers ==="
    python download_subset.py $COMMON --meta_data_path "$META_DATA_PATH"
    BUDGETS="$EPS_LIST"
else
    echo "=== [2] (embedding mode: no image download) ==="
    BUDGETS="$RHO_LIST"
fi

run_one() {  # $1 = TAG (eps16 / emb20), $2... = clip_attack args
    local TAG="$1"; shift
    echo "=== [3] attack ($TAG) ==="
    python clip_attack.py $COMMON --steps "$STEPS" "$@"
    echo "=== [3b] re-quantize ($TAG) ==="
    python requantize.py $COMMON --ckpt_path "$RQVAE_CKPT" \
        --adv_emb "artifacts/adv_emb_${TAG}.npz"
    echo "=== [4] evaluate ($TAG) ==="
    python run_eval.py $COMMON --ckpt_path "$MODEL_CKPT" \
        --attacked_index "artifacts/index_vitemb_ATTACKED_${TAG}.json" \
        --requant_diag "artifacts/requant_diag_${TAG}.json"
}

for B in $BUDGETS; do
    if [ "$MODE" = "pixel" ]; then
        EPS=$(python3 -c "print($B/255)")
        run_one "eps${B}" --mode pixel --eps "$EPS" --save_adv_images
    else
        RHO=$(python3 -c "print($B/100)")
        run_one "emb${B}" --mode embedding --emb_rho "$RHO"
    fi
done

echo "=== done. results in attack/artifacts/eval_*.json ==="
