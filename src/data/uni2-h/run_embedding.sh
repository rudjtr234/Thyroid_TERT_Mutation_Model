#!/bin/bash
# Thyroid TERT 데이터 임베딩 (UNI2-h) - 병렬 버전
# 사용법: bash run_embedding.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILE_BASE="/path/to/dataset/40x_patch"
OUT_BASE="/path/to/dataset/embedding"

BATCH_SIZE=512

# CPU 스레드 제한 (각 프로세스당 17개, 총 51코어 = 64코어의 80%)
export OMP_NUM_THREADS=17
export MKL_NUM_THREADS=17

echo "========================================"
echo "Thyroid TERT Embedding with UNI2-h"
echo "병렬 실행: 각 카테고리별 다른 GPU 사용"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

# 로그 디렉토리 생성
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# C228T - GPU 1,2
echo "[1/3] C228T 임베딩 시작... (GPU 1,2)"
CUDA_VISIBLE_DEVICES=1,2 python "$SCRIPT_DIR/preprocess_data.py" \
    --tile_dir "$TILE_BASE/C228T" \
    --out_dir "$OUT_BASE/C228T" \
    --batch_size $BATCH_SIZE \
    > "$LOG_DIR/c228t.log" 2>&1 &
PID1=$!

# C250T - GPU 3,4
echo "[2/3] C250T 임베딩 시작... (GPU 3,4)"
CUDA_VISIBLE_DEVICES=3,4 python "$SCRIPT_DIR/preprocess_data.py" \
    --tile_dir "$TILE_BASE/C250T" \
    --out_dir "$OUT_BASE/C250T" \
    --batch_size $BATCH_SIZE \
    > "$LOG_DIR/c250t.log" 2>&1 &
PID2=$!

# Wild - GPU 5,6,7
echo "[3/3] Wild 임베딩 시작... (GPU 5,6,7)"
CUDA_VISIBLE_DEVICES=5,6,7 python "$SCRIPT_DIR/preprocess_data.py" \
    --tile_dir "$TILE_BASE/Wild" \
    --out_dir "$OUT_BASE/Wild" \
    --batch_size $BATCH_SIZE \
    > "$LOG_DIR/wild.log" 2>&1 &
PID3=$!

echo ""
echo "========================================"
echo "백그라운드 실행 중..."
echo "  C228T PID: $PID1 (로그: $LOG_DIR/c228t.log)"
echo "  C250T PID: $PID2 (로그: $LOG_DIR/c250t.log)"
echo "  Wild  PID: $PID3 (로그: $LOG_DIR/wild.log)"
echo ""
echo "진행 상황 확인: tail -f $LOG_DIR/*.log"
echo "========================================"

# 모든 프로세스 완료 대기
wait $PID1 $PID2 $PID3

echo ""
echo "========================================"
echo "모든 임베딩 완료!"
echo "결과 저장 위치: $OUT_BASE"
echo "========================================"