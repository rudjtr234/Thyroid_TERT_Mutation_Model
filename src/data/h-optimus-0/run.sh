#!/bin/bash
# H-Optimus-0 Feature Extraction - Thyroid TERT Dataset
# 143 서버에서 실행
#
# Usage:
#   cd /path/to/Thyroid_TERT_prediction_model/src/data/h-optimus-0
#   bash run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RTX 3080(1,5,6) 제외하고 V100(2,3,4)만 사용
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPUS[@]}
MASTER_PORT=29502
BATCH_SIZE=512

TILE_BASE="/path/to/dataset/40x_patch"
OUT_BASE="/path/to/dataset/h_optimus_embeddings"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "========== [1/3] C228T =========="
torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/extract_features.py" \
    --tile_dir "$TILE_BASE/C228T" \
    --out_dir  "$OUT_BASE/C228T" \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "$LOG_DIR/c228t_hoptimus.log"

echo "========== [2/3] C250T =========="
torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/extract_features.py" \
    --tile_dir "$TILE_BASE/C250T" \
    --out_dir  "$OUT_BASE/C250T" \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "$LOG_DIR/c250t_hoptimus.log"

echo "========== [3/3] Wild =========="
torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/extract_features.py" \
    --tile_dir "$TILE_BASE/Wild" \
    --out_dir  "$OUT_BASE/Wild" \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "$LOG_DIR/wild_hoptimus.log"

echo "========== 전체 완료 =========="
