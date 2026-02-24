#!/bin/bash
# ============================================================
# Thyroid TERT Mutation Prediction Training Script
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Optional runtime overrides
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CV_SPLIT_FILE="${CV_SPLIT_FILE:-$PROJECT_ROOT/config/cv_splits_tert_example.json}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$PROJECT_ROOT/outputs/thyroid_tert_model_v0.1.0}"

EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
BAG_SIZE="${BAG_SIZE:-5000}"
SEED="${SEED:-42}"
MODEL_TYPE="${MODEL_TYPE:-abmil}"   # abmil or transmil

IN_DIM="${IN_DIM:-1536}"
DROPOUT="${DROPOUT:-0.25}"

EMBED_DIM="${EMBED_DIM:-768}"
ATTN_DIM="${ATTN_DIM:-512}"
NUM_FC_LAYERS="${NUM_FC_LAYERS:-2}"

TRANSMIL_EMBED_DIM="${TRANSMIL_EMBED_DIM:-512}"
TRANSMIL_NUM_HEADS="${TRANSMIL_NUM_HEADS:-8}"
TRANSMIL_NUM_LAYERS="${TRANSMIL_NUM_LAYERS:-2}"
TRANSMIL_NUM_LANDMARKS="${TRANSMIL_NUM_LANDMARKS:-256}"
TRANSMIL_PINV_ITERATIONS="${TRANSMIL_PINV_ITERATIONS:-6}"

python main.py \
    --cv_split_file "$CV_SPLIT_FILE" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --bag_size "$BAG_SIZE" \
    --model_type "$MODEL_TYPE" \
    --in_dim "$IN_DIM" \
    --dropout "$DROPOUT" \
    --embed_dim "$EMBED_DIM" \
    --attn_dim "$ATTN_DIM" \
    --num_fc_layers "$NUM_FC_LAYERS" \
    --transmil_embed_dim "$TRANSMIL_EMBED_DIM" \
    --transmil_num_heads "$TRANSMIL_NUM_HEADS" \
    --transmil_num_layers "$TRANSMIL_NUM_LAYERS" \
    --transmil_num_landmarks "$TRANSMIL_NUM_LANDMARKS" \
    --transmil_pinv_iterations "$TRANSMIL_PINV_ITERATIONS" \
    --seed "$SEED" \
    --save_model \
    --save_best_only \
    --generate_plots
