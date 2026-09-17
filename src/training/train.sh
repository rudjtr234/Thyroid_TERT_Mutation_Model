#!/bin/bash
# ============================================================
# Thyroid TERT Mutation Prediction Training Script
# Task: TERT Promoter Mutation (Wild vs Mutant)
# Model: ABMIL / TransMIL / ACMIL / DTFD / MHIM / CLAM + UNI2 (1536-dim)
# ============================================================

# 스크립트 디렉토리로 이동 (상대 임포트 문제 해결)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 설정
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=GPU-d74d0409-85c9-ba24-a0c1-d9297389ebc5

# ============================================================
# 아래 5개 변수만 필요시 직접 수정
# ============================================================
# [1] UNI2-H (40x 512×512)
# CV_SPLIT_FILE="config/cv_splits_tert_5fold_seed42.json"

# [2] H-Optimus-0 (40x 구버전, v0.6.x~v0.8.x에서 사용)
# CV_SPLIT_FILE="config/cv_splits_tert_5fold_seed42_hoptimus.json"

# [3] H-Optimus-0 (20x 224×224, 최신)
# CV_SPLIT_FILE="config/cv_splits_tert_5fold_seed42_hoptimus_20x.json"

# [4] H-Optimus-0 (40x, 3-class Wild/C228T/C250T)
CV_SPLIT_FILE="config/cv_splits_tert_5fold_seed42_hoptimus_3class.json"

MODEL_SAVE_DIR="outputs/thyroid_tert_model_v0.15.9"

NUM_CLASSES=3   # 2 (Wild vs Mutant) | 3 (Wild/C228T/C250T) — 3일 때는 3class CV split 사용

# 학습 파라미터
EPOCHS=100
LR=1e-4
BAG_SIZE=400
SEED=42
MODEL_TYPE=clam   # abmil | transmil | acmil | dtfd | mhim | clam

# Model 공통 파라미터 (abmil / acmil / dtfd / mhim / clam 공용)
IN_DIM=1536
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

echo "============================================================"
echo "Thyroid TERT Mutation Prediction Training"
echo "============================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model Type: $MODEL_TYPE"
echo "CV Split File: $CV_SPLIT_FILE"
echo "Output Dir: $MODEL_SAVE_DIR"
echo "============================================================"

# 학습 실행
python main.py \
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
