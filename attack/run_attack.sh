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

# Resolve user-facing paths against the INVOCATION dir (usually the repo root)
# BEFORE we cd into attack/, so repo-root-relative args like
# 'index/log/.../best_collision_model.pth' keep working.
abspath() { case "$1" in /*) printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }

MODE=${MODE:-pixel}
DATASET=${DATASET:-Instruments}
DATA_PATH=$(abspath "${DATA_PATH:-data}")
NUM_TASKS=${NUM_TASKS:-200}
PROMOTE=${PROMOTE:-0}                    # 1 = promote COLD non-interacted target items
N_TARGETS=${N_TARGETS:-5}               # [promote] how many cold targets
HIJACK_ID=${HIJACK_ID:-29}              # [hijack] popular item whose code is forged onto targets
TARGET_ID=${TARGET_ID:-}                # [pixel/embedding] aim at THIS item's embedding (not the centroid)
[ "$MODE" = "hijack" ] && PROMOTE=1     # hijack only makes sense as a promotion attack
EPS_LIST=${EPS_LIST:-"16 32 64"}        # pixel: L_inf in /255
RHO_LIST=${RHO_LIST:-"10 20 30 50"}     # embedding: L2 budget in % of ||x||
STEPS=${STEPS:-30}
MODEL_CKPT=$(abspath "${MODEL_CKPT:-log/$DATASET}")
RQVAE_CKPT=$(abspath "${RQVAE_CKPT:-index/log/$DATASET/ViT-L-14_256/best_collision_model.pth}")
[ -n "${META_DATA_PATH:-}" ] && META_DATA_PATH=$(abspath "$META_DATA_PATH")
GPUS=${GPUS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPUS}

cd "$(dirname "$0")"                     # into attack/ (absolute paths above stay valid)

[ "$MODE" = "hijack" ] || [ -f "$RQVAE_CKPT" ] || { echo "ERROR: RQ-VAE ckpt not found: $RQVAE_CKPT"; exit 1; }
[ -d "$MODEL_CKPT" ] || { echo "ERROR: recommender dir not found: $MODEL_CKPT"; exit 1; }

COMMON="--data_path $DATA_PATH --dataset $DATASET"

echo "=== [1] sample tasks ==="
SAMPLE_ARGS="--num_tasks $NUM_TASKS"
[ "$PROMOTE" = "1" ] && SAMPLE_ARGS="$SAMPLE_ARGS --promote --n_targets $N_TARGETS"
python sample_tasks.py $COMMON $SAMPLE_ARGS

if [ "$MODE" = "hijack" ]; then
    TAG="hijack${HIJACK_ID}"
    echo "=== [3] code hijack -> item $HIJACK_ID ==="
    python hijack_codes.py $COMMON --hijack_id "$HIJACK_ID"
    echo "=== [4] evaluate ($TAG) ==="
    python run_eval.py $COMMON --ckpt_path "$MODEL_CKPT" \
        --attacked_index "artifacts/index_vitemb_ATTACKED_${TAG}.json" \
        --requant_diag "artifacts/hijack_diag_${TAG}.json"
    echo "=== done. results in attack/artifacts/eval_*.json ==="
    exit 0
fi

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

TARG_ARG=""; TSUF=""
if [ -n "$TARGET_ID" ]; then TARG_ARG="--target_id $TARGET_ID"; TSUF="t${TARGET_ID}"; fi
for B in $BUDGETS; do
    if [ "$MODE" = "pixel" ]; then
        EPS=$(python3 -c "print($B/255)")
        run_one "eps${B}${TSUF}" --mode pixel --eps "$EPS" --save_adv_images $TARG_ARG
    else
        RHO=$(python3 -c "print($B/100)")
        run_one "emb${B}${TSUF}" --mode embedding --emb_rho "$RHO" $TARG_ARG
    fi
done

echo "=== done. results in attack/artifacts/eval_*.json ==="
