#!/bin/bash
# ============================================================
# UNI2 embedding helper for Thyroid TERT dataset
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILE_BASE="${TILE_BASE:-/path/to/40x_patch}"
OUT_BASE="${OUT_BASE:-/path/to/embedding}"
BATCH_SIZE="${BATCH_SIZE:-512}"

GPU_C228T="${GPU_C228T:-0}"
GPU_C250T="${GPU_C250T:-1}"
GPU_WILD="${GPU_WILD:-2}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU_C228T" python "$SCRIPT_DIR/preprocess_data.py" \
    --tile_dir "$TILE_BASE/C228T" \
    --out_dir "$OUT_BASE/C228T" \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_DIR/c228t.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES="$GPU_C250T" python "$SCRIPT_DIR/preprocess_data.py" \
    --tile_dir "$TILE_BASE/C250T" \
    --out_dir "$OUT_BASE/C250T" \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_DIR/c250t.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES="$GPU_WILD" python "$SCRIPT_DIR/preprocess_data.py" \
    --tile_dir "$TILE_BASE/Wild" \
    --out_dir "$OUT_BASE/Wild" \
    --batch_size "$BATCH_SIZE" \
    > "$LOG_DIR/wild.log" 2>&1 &
PID3=$!

wait $PID1 $PID2 $PID3
