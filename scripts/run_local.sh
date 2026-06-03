#!/bin/bash
# ---------------------------------------------------------------------------
# Single-GPU, end-to-end local run-through of MQL4GRec on the Instruments
# dataset, using the pre-generated semantic-ID indices shipped in
# data/Instruments/ (so NO image download / CLIP / RQ-VAE step is needed).
#
# Pipeline:  fine-tune  ->  inference (text channel + image channel)  ->  ensemble
#
# Usage:
#   bash scripts/run_local.sh                 # faithful settings (epochs=200, early-stops)
#   EPOCHS=2 BATCH=128 bash scripts/run_local.sh   # quick smoke test (just prove it runs)
#   LOAD_MODEL_NAME=/path/to/pretrain_ckpt bash scripts/run_local.sh  # start from a pretrained ckpt
#
# Env overrides: EPOCHS, BATCH, NUM_BEAMS, DATASET, GPUS (nproc_per_node), LOAD_MODEL_NAME
# ---------------------------------------------------------------------------
set -e

export WANDB_MODE=disabled

DATASET=${DATASET:-Instruments}
DATA_PATH=./data/
INDEX_FILE=.index_lemb.json
IMAGE_INDEX_FILE=.index_vitemb.json
TASKS='seqrec,seqimage,item2image,image2item,fusionseqrec'
VALID_TASK=seqrec

EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
NUM_BEAMS=${NUM_BEAMS:-20}
GPUS=${GPUS:-1}
PORT=${PORT:-2309}

OUTPUT_DIR=./log/$DATASET
mkdir -p "$OUTPUT_DIR"

# Optional: resume from a pretrained checkpoint (e.g. downloaded from the
# project's Google Drive). If unset, fine-tune from the small base T5 in config/ckpt.
LOAD_ARG=""
if [ -n "$LOAD_MODEL_NAME" ]; then
    LOAD_ARG="--load_model_name $LOAD_MODEL_NAME"
fi

echo "============================================================"
echo " MQL4GRec local run | dataset=$DATASET gpus=$GPUS epochs=$EPOCHS batch=$BATCH beams=$NUM_BEAMS"
echo "============================================================"

# Quick sanity check that the pre-built data is present.
for f in "$DATASET.inter.json" "$DATASET$INDEX_FILE" "$DATASET$IMAGE_INDEX_FILE"; do
    if [ ! -f "$DATA_PATH/$DATASET/$f" ]; then
        echo "ERROR: missing data file $DATA_PATH/$DATASET/$f"; exit 1
    fi
done

# ---- 1. Fine-tune (multi-task: text + image + cross-modal alignment) -------
echo ">>> [1/3] Fine-tuning ..."
torchrun --nproc_per_node="$GPUS" --master_port="$PORT" finetune.py \
    --base_model ./config/ckpt \
    $LOAD_ARG \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_batch_size "$BATCH" \
    --learning_rate 5e-4 \
    --epochs "$EPOCHS" \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file "$INDEX_FILE" \
    --image_index_file "$IMAGE_INDEX_FILE" \
    --tasks "$TASKS" \
    --valid_task "$VALID_TASK" 2>&1 | tee "$OUTPUT_DIR/train.log"

# ---- 2. Inference for BOTH channels (ensemble.py needs both save files) ----
for TASK in seqrec seqimage; do
    echo ">>> [2/3] Inference: $TASK channel ..."
    torchrun --nproc_per_node="$GPUS" --master_port="$PORT" test_ddp_save.py \
        --ckpt_path "$OUTPUT_DIR" \
        --data_path "$DATA_PATH" \
        --dataset "$DATASET" \
        --test_batch_size 64 \
        --num_beams "$NUM_BEAMS" \
        --index_file "$INDEX_FILE" \
        --image_index_file "$IMAGE_INDEX_FILE" \
        --test_task "$TASK" \
        --results_file "$OUTPUT_DIR/results_${TASK}_${NUM_BEAMS}.json" \
        --save_file "$OUTPUT_DIR/save_${TASK}_${NUM_BEAMS}.json" \
        --filter_items 2>&1 | tee "$OUTPUT_DIR/test_${TASK}.log"
done

# ---- 3. Ensemble (fuse text + image predictions) --------------------------
echo ">>> [3/3] Ensemble ..."
python ensemble.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --index_file "$INDEX_FILE" \
    --image_index_file "$IMAGE_INDEX_FILE" \
    --num_beams "$NUM_BEAMS"

echo "============================================================"
echo " Done. Results in:"
echo "   text  channel : $OUTPUT_DIR/results_seqrec_${NUM_BEAMS}.json"
echo "   image channel : $OUTPUT_DIR/results_seqimage_${NUM_BEAMS}.json"
echo "   ensemble      : $OUTPUT_DIR/results_ensemble_${NUM_BEAMS}.json"
echo "============================================================"
