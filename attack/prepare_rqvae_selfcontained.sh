#!/bin/bash
# ---------------------------------------------------------------------------
# Self-contained RQ-VAE setup, for when the authors' RQ-VAE checkpoint is NOT
# available (Option B). Produces a quantizer that is CONSISTENT with the codes
# the recommender is trained on, by doing all three steps together:
#   1) train an RQ-VAE on the shipped image embeddings
#   2) regenerate the image codes  (overwrites <dataset>.index_vitemb.json;
#      the shipped file is backed up to *.index_vitemb.SHIPPED.bak.json)
#   3) RE-FINETUNE the recommender on the new codes (replaces ./log/<dataset>)
#
# WARNING: step 3 retrains the recommender. The attack measures *relative* rank
# change, so you can use fewer epochs here (e.g. EPOCHS=50) to save time.
#
# After this, run the attack with:
#   RQVAE_CKPT=index/log/<dataset>/ViT-L-14_256/best_collision_model.pth \
#   META_DATA_PATH=/path/to/amazon18/Metadata bash attack/run_attack.sh
#
# Usage:  EPOCHS=50 bash attack/prepare_rqvae_selfcontained.sh
# Overrides: DATASET, RQVAE_EPOCHS, EPOCHS (recommender finetune), GPUS
# ---------------------------------------------------------------------------
set -e
set -o pipefail
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPUS:-0}}

cd "$(dirname "$0")/.."                         # repo root

DATASET=${DATASET:-Instruments}
RQVAE_EPOCHS=${RQVAE_EPOCHS:-500}
FT_EPOCHS=${EPOCHS:-200}
CKPT_DIR=index/log/$DATASET/ViT-L-14_256
EMB=data/$DATASET/$DATASET.emb-ViT-L-14.npy
INDEX=data/$DATASET/$DATASET.index_vitemb.json

[ -f "$EMB" ] || { echo "ERROR: image embeddings not found: $EMB"; exit 1; }

echo "=== [1/3] train RQ-VAE on $EMB (epochs=$RQVAE_EPOCHS) ==="
cd index
python main.py \
    --data_path "../$EMB" \
    --num_emb_list 256 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0 0.003 \
    --e_dim 32 \
    --layers 2048 1024 512 256 128 64 \
    --batch_size 1024 --epochs "$RQVAE_EPOCHS" --device cuda:0 \
    --ckpt_dir "log/$DATASET/ViT-L-14_256"
cd ..

[ -f "$CKPT_DIR/best_collision_model.pth" ] || {
    echo "ERROR: RQ-VAE training did not produce $CKPT_DIR/best_collision_model.pth"; exit 1; }

echo "=== [2/3] regenerate image codes (backing up shipped index) ==="
cp -n "$INDEX" "data/$DATASET/$DATASET.index_vitemb.SHIPPED.bak.json" 2>/dev/null || true
cd index
python generate_indices_distance.py \
    --device cuda:0 \
    --ckpt_path "log/$DATASET/ViT-L-14_256/best_collision_model.pth" \
    --output_dir "../data/$DATASET" \
    --output_file "$DATASET.index_vitemb.json" \
    --content image
cd ..

echo "=== [3/3] re-finetune the recommender on the new codes (epochs=$FT_EPOCHS) ==="
EPOCHS=$FT_EPOCHS bash scripts/run_local.sh

echo "============================================================"
echo " Done. Consistent RQ-VAE + codes + recommender are ready."
echo "   RQ-VAE  : $CKPT_DIR/best_collision_model.pth"
echo "   codes   : $INDEX  (shipped backed up to *.SHIPPED.bak.json)"
echo "   model   : ./log/$DATASET/"
echo " Next:"
echo "   RQVAE_CKPT=$CKPT_DIR/best_collision_model.pth \\"
echo "   META_DATA_PATH=/path/to/amazon18/Metadata bash attack/run_attack.sh"
echo "============================================================"
