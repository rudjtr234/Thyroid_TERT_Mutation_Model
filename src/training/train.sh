#!/bin/bash
# ============================================================
# Thyroid TERT Mutation Prediction Training Script
# Task    : TERT Promoter Mutation 2-class (Wild=0 vs Mutant=1, C228T+C250T)
# Encoder : CV_SPLIT_FILE 선택에 따름 (UNI2-H / H-Optimus-0 / H-Optimus-1 = 1536-dim,
#                                      Hibou-L = 1024-dim, Virchow2 = 2560-dim)
# Model   : ABMIL / TransMIL / ACMIL / DTFD / MHIM / CLAM
# ============================================================

# 스크립트 디렉토리로 이동 (상대 임포트 문제 해결)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================
# GPU 선택
#   번호는 nvidia-smi 와 동일하다 (common_env.sh 가 CUDA_DEVICE_ORDER=PCI_BUS_ID
#   를 설정하므로 PyTorch 기본값인 FASTEST_FIRST 로 뒤섞이지 않는다).
#
#   예) 혼종 GPU 서버 : 0,1=RTX3080  2,3,4=V100S  5,6=RTX3080
#       -> 추출이 0~4 를 쓰는 동안 학습은 5, 6 번.  GPU_ID=5 bash train.sh
#
# 환경변수로 덮어쓸 수 있다:  GPU_ID=5 bash train.sh
# ============================================================
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

# Python 인터프리터 / CUDA 점검 / GPU 번호 고정은 공통 스크립트에 모아 뒀다.
# 서버마다 conda 경로가 달라 자동 탐지한다 (PYTHON=... 로 덮어쓰기 가능).
source "$SCRIPT_DIR/../../scripts/common_env.sh" || exit 1

# ============================================================
# CV split / 출력 경로 / 클래스 수 — 필요시 아래만 수정
# 주의 1: CV_SPLIT_FILE 과 NUM_CLASSES 는 반드시 같은 계열로 맞출 것
#         (2class split ↔ NUM_CLASSES=2 / 3class split ↔ NUM_CLASSES=3)
# 주의 2: IN_DIM 은 인코더마다 다르다. Hibou-L=1024, Virchow2=2560, 그 외(UNI2-H/H-Optimus)=1536.
#         아래 IN_DIM 은 split JSON의 embed_dim 에서 자동으로 읽으므로 보통 건드릴 필요 없다.
# ============================================================
# [1] UNI2-H (40x 512×512)
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42.json"

# [2] H-Optimus-0 (40x 구버전, v0.6.x~v0.8.x에서 사용)
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hoptimus.json"

# [3] H-Optimus-0 (20x 224×224, 최신)
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hoptimus_20x.json"

# [4] H-Optimus-0 (40x, 3-class Wild/C228T/C250T)
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hoptimus_3class.json"

# [5] H-Optimus-1 (40x, 2-class Wild vs Mutant)
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hoptimus1_40x_2class.json"

# [6] H-Optimus-1 (40x, 3-class Wild/C228T/C250T)
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hoptimus1_40x_3class.json"

# [7] Hibou-L (40x, 2-class Wild vs Mutant)  — 1024-dim 주의  ← 현재 사용
CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hibou_L_40x_2class.json"

# [8] Hibou-L (40x, 3-class Wild/C228T/C250T)  — 1024-dim 주의
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_hibou_L_40x_3class.json"

# [9] Virchow2 (40x 512, 2-class Wild vs Mutant)  — 2560-dim 주의
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_virchow2_40x_2class.json"

# [10] Virchow2 (40x 512, 3-class Wild/C228T/C250T)  — 2560-dim 주의
# CV_SPLIT_FILE="$PROJECT_ROOT/config/cv_splits_tert_5fold_seed42_virchow2_40x_3class.json"
#
# [9][10] 을 쓰려면 먼저 추출 + split 생성이 끝나야 한다:
#   cd ../data/virchow2 && bash run.sh
#   bash scripts/make_virchow2_splits.sh                # 2class (기본)
# 임베딩 경로 구조: {embedding_root}/virchow2_40x512_concat/{class}/npy/

# 인코더 계열별 버전 라인을 분리해 관리한다:
#   UNI2-H / H-Optimus-0 → thyroid_tert_model_vX.X.X
#   H-Optimus-1          → thyroid_tert_hoptimus1_vX.X.X
#   Hibou-L              → thyroid_tert_hibou_vX.X.X
#   Virchow2             → thyroid_tert_virchow2_vX.X.X
MODEL_SAVE_DIR="$PROJECT_ROOT/outputs/thyroid_tert_hibou_v0.5.9"

NUM_CLASSES=2   # 2 (Wild vs Mutant) | 3 (Wild/C228T/C250T) — CV split 파일과 반드시 일치시킬 것

# 학습 파라미터
EPOCHS=100
LR=1e-4
BAG_SIZE=400
SEED=42
MODEL_TYPE=mhim  # abmil | transmil | acmil | dtfd | mhim | clam

# ============================================================
# 외부 오버라이드 (scripts/sweep.sh 가 사용한다)
#   위에서 고른 값들을 TE_* 환경변수로 덮어쓸 수 있게 한다.
#   스윕에서 run 마다 bag_size / 출력경로를 바꿔야 하는데, 위 라인들은
#   그냥 대입문이라 환경변수가 그대로는 먹지 않기 때문이다.
#   직접 실행할 때는 TE_* 를 안 주면 되므로 동작이 바뀌지 않는다.
# ============================================================
CV_SPLIT_FILE="${TE_CV_SPLIT_FILE:-$CV_SPLIT_FILE}"
MODEL_SAVE_DIR="${TE_MODEL_SAVE_DIR:-$MODEL_SAVE_DIR}"
NUM_CLASSES="${TE_NUM_CLASSES:-$NUM_CLASSES}"
EPOCHS="${TE_EPOCHS:-$EPOCHS}"
LR="${TE_LR:-$LR}"
BAG_SIZE="${TE_BAG_SIZE:-$BAG_SIZE}"
SEED="${TE_SEED:-$SEED}"
MODEL_TYPE="${TE_MODEL_TYPE:-$MODEL_TYPE}"

# Model 공통 파라미터 (abmil / acmil / dtfd / mhim / clam 공용)
# IN_DIM: split JSON의 embed_dim을 읽어 자동 설정 (Hibou-L=1024 / Virchow2=2560 / 그 외=1536).
# 수동 고정이 필요하면 아래 줄을 IN_DIM=1536 처럼 직접 덮어쓰면 된다.
# split 파일 존재 확인을 IN_DIM 추론보다 먼저 한다.
# (없는 파일을 python 에 넘기면 FileNotFoundError 트레이스백이 떠서
#  원인이 "split 경로 오타"라는 걸 알아보기 어렵다)
if [ ! -f "$CV_SPLIT_FILE" ]; then
    echo "[ERROR] CV split 파일이 없습니다: $CV_SPLIT_FILE" >&2
    exit 1
fi
IN_DIM=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('embed_dim',1536))" "$CV_SPLIT_FILE")
if [ -z "$IN_DIM" ]; then
    echo "[!] IN_DIM 추론 실패 — CV_SPLIT_FILE 확인 필요: $CV_SPLIT_FILE"
    exit 1
fi
echo "[INFO] CV split   : $CV_SPLIT_FILE"
echo "[INFO] IN_DIM     : $IN_DIM (split JSON의 embed_dim 기준)"
echo "[INFO] NUM_CLASSES: $NUM_CLASSES"
DROPOUT=0.25
EMBED_DIM=512
ATTN_DIM=384
NUM_FC_LAYERS=2

# TransMIL params
TRANSMIL_EMBED_DIM=512
TRANSMIL_NUM_HEADS=8
TRANSMIL_NUM_LAYERS=2
TRANSMIL_NUM_LANDMARKS=256
TRANSMIL_PINV_ITERATIONS=6

# ACMIL params
ACMIL_N_TOKEN=5
ACMIL_N_MASKED_PATCH=10
ACMIL_MASK_DROP=0.6

# DTFD-MIL params
DTFD_N_PSEUDO_BAGS=4

# MHIM-MIL params
MHIM_MASK_RATIO=0.5
MHIM_EMA_DECAY=0.999

# CLAM params
CLAM_K_SAMPLE=8

# 사전 점검: CUDA / GPU 점유 / 모델별 의존성 (split 파일은 위에서 확인)
te_guard_gpu_choice || exit 1   # 가드 대상 서버: 추출용 GPU 선택 차단
te_check_cuda || exit 1
te_warn_busy_gpu
te_power_guard      # 가드 대상 서버에서만 전력 캡 적용 (Xid 79 재발 방지)
if [ "$MODEL_TYPE" = "transmil" ]; then
    te_check_transmil || exit 1
fi

echo "============================================================"
echo "Thyroid TERT Mutation Prediction Training"
echo "============================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $PYTHON"
echo "Num Classes: $NUM_CLASSES"
echo "Model Type: $MODEL_TYPE"
echo "CV Split File: $CV_SPLIT_FILE"
echo "Output Dir: $MODEL_SAVE_DIR"
echo "============================================================"

# 학습 실행
"$PYTHON" -u main.py \
    --cv_split_file "$CV_SPLIT_FILE" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --lr $LR \
    --bag_size $BAG_SIZE \
    --model_type $MODEL_TYPE \
    --in_dim $IN_DIM \
    --dropout $DROPOUT \
    --embed_dim $EMBED_DIM \
    --attn_dim $ATTN_DIM \
    --num_fc_layers $NUM_FC_LAYERS \
    --transmil_embed_dim $TRANSMIL_EMBED_DIM \
    --transmil_num_heads $TRANSMIL_NUM_HEADS \
    --transmil_num_layers $TRANSMIL_NUM_LAYERS \
    --transmil_num_landmarks $TRANSMIL_NUM_LANDMARKS \
    --transmil_pinv_iterations $TRANSMIL_PINV_ITERATIONS \
    --acmil_n_token $ACMIL_N_TOKEN \
    --acmil_n_masked_patch $ACMIL_N_MASKED_PATCH \
    --acmil_mask_drop $ACMIL_MASK_DROP \
    --dtfd_n_pseudo_bags $DTFD_N_PSEUDO_BAGS \
    --mhim_mask_ratio $MHIM_MASK_RATIO \
    --mhim_ema_decay $MHIM_EMA_DECAY \
    --clam_k_sample $CLAM_K_SAMPLE \
    --seed $SEED \
    --save_model \
    --save_best_only \
    --generate_plots

echo "============================================================"
echo "Training completed!"
echo "Results saved to: $MODEL_SAVE_DIR"
echo "============================================================"
