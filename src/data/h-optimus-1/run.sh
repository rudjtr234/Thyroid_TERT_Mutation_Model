#!/bin/bash
# H-Optimus-1 Feature Extraction - Thyroid TERT Dataset
#
# Usage:
#   cd /path/to/Thyroid_TERT_prediction_model/src/data/h-optimus-1
#   bash run.sh
#
# H-optimus-1은 gated repo다. 서버마다 토큰/캐시가 따로 필요하므로
# 처음 쓰는 서버에서는 huggingface-cli login 후 실행할 것.
#
# ===== 2026-09-09 GPU 장애 이후 구성 =====
# 101호 deepgadget에서 Xid 79("GPU has fallen off the bus")로 4장이 동시에 이탈해
# 7장 중 3장(V100S ×2, RTX 3080 ×1)만 남았다. 공통 전원 문제로 추정되므로
# 남은 3장을 각각 독립 작업 1개씩에 배정하고, 전력 캡을 걸어 동시 피크를 낮춘다.
#   V100S ×2 → BRAF TCGA 40x 임베딩 (샤드 2개)
#   RTX 3080 ×1 → 이 스크립트 (TERT 임베딩)
# DDP로 여러 장을 묶지 않는다 — 한 장이 또 이탈하면 NCCL이 600초 멈추고
# 그 슬라이드가 통째로 유실되기 때문이다.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DataLoader worker가 fd를 많이 쓴다 — "Too many open files"(Errno 24) 방지.
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# ------------------------------------------------------- GPU 지정 (UUID 필수)
# 장애 후 GPU 번호가 재배치되었다. 번호로 지정하면 엉뚱한 장치에 잡히므로 UUID로 건다.
# !! 실행 전 nvidia-smi --query-gpu=index,name,uuid --format=csv 로 확인할 것 !!
RTX3080_UUID=${RTX3080_UUID:-}

# 2026-09-14: ainode144(RTX 6000 Ada ×8)에서도 재개할 수 있게 후보 GPU를 넓혔다.
# 101호 deepgadget 에서는 3080 이 먼저 잡히므로 기존 동작 그대로다.
GPU_MATCH=${GPU_MATCH:-"3080|6000 Ada"}

if [ -z "$RTX3080_UUID" ]; then
    # UUID를 안 넘겼으면 쓸 수 있는 GPU를 자동으로 찾는다.
    RTX3080_UUID=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null \
                   | grep -iE "$GPU_MATCH" | head -1 | cut -d, -f2 | tr -d ' ')
fi

if [ -z "$RTX3080_UUID" ]; then
    echo "[FATAL] 쓸 수 있는 GPU를 찾을 수 없다 (패턴: $GPU_MATCH)."
    echo "        현재 인식되는 GPU를 확인할 것:"
    echo "        nvidia-smi --query-gpu=index,name,uuid --format=csv"
    exit 1
fi

# --------------------------------------------------- 전력 캡 + 존재 확인 (필수)
GUARD="scripts/gpu_guard.sh"
if [ -f "$GUARD" ]; then
    source "$GUARD"
    gpu_guard_require "$RTX3080_UUID" || exit 1
    gpu_guard_watch_xid "$RTX3080_UUID" 60 &
    WATCHDOG_PID=$!
    trap 'kill $WATCHDOG_PID 2>/dev/null' EXIT
else
    echo "[WARN] $GUARD 가 없다 — 전력 캡 없이 진행한다."
fi

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$RTX3080_UUID"
MASTER_PORT=29504          # BRAF TCGA 샤드(29505/29506) 및 학습과 겹치지 않게 분리

# RTX 3080은 10GB라 V100S(32GB) 기준 512로는 OOM이 난다.
# 40x 512 타일은 224로 resize되므로 타일당 메모리는 배율과 무관하다.
BATCH_SIZE=${BATCH_SIZE:-32}

# 주의: 이 디렉토리의 extract_features.py 는 DataLoader 를 쓰지 않고
# embed_batch() 에서 이미지를 직접 연다 — --num_workers 인자가 없다.
# 넘기면 argparse 가 "unrecognized arguments" 로 죽으므로 전달하지 않는다.
# (BRAF 쪽 h-optimus-1/extract_features.py 는 DataLoader 기반이라 그 인자를 받는다.)

# 40x 512 패치 사용 — BRAF v0.20.x(h-optimus-1 40x512)와 배율을 맞춰 비교 가능하게 한다.
TILE_BASE="/path/to/dataset/40x_patch"
OUT_BASE="/path/to/dataset/h_optimus_1_embeddings_40x"

# 로그 디렉토리를 못 만들면(권한 등) 로그 없이 진행한다 — 임베딩이 권한 문제로 멈추지 않도록.
LOG_DIR="$SCRIPT_DIR/logs"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG_DIR="${TMPDIR:-/tmp}/tert_hoptimus1_logs"
    mkdir -p "$LOG_DIR"
    echo "[WARN] $SCRIPT_DIR/logs 생성 실패 → 로그를 $LOG_DIR 에 남긴다"
fi

# 라벨은 2-class(Wild=0 / Mutant=C228T+C250T=1)로 묶지만,
# 임베딩은 원본 클래스 디렉토리 구조 그대로 뽑는다 — 묶는 것은 CV split 단계의 일이다.
for CLS in C228T C250T Wild; do
    echo "========== $CLS =========="
    torchrun \
        --nproc_per_node=1 \
        --master_port=$MASTER_PORT \
        "$SCRIPT_DIR/extract_features.py" \
        --tile_dir "$TILE_BASE/$CLS" \
        --out_dir  "$OUT_BASE/$CLS" \
        --batch_size $BATCH_SIZE \
        2>&1 | tee "$LOG_DIR/$(echo $CLS | tr A-Z a-z)_hoptimus1.log"
done

echo "========== 전체 완료 =========="
