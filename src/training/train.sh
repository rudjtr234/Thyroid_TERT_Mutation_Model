
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
export CUDA_VISIBLE_DEVICES=0

# ============================================================
# 아래 2개만 필요시 직접 수정
# ============================================================
# UNI2-H
# CV_SPLIT_FILE="/path/to/project/config/cv_splits_tert_5fold_seed42.json"
# H-Optimus-0
CV_SPLIT_FILE="/path/to/project/config/cv_splits_tert_5fold_seed42_hoptimus.json"
MODEL_SAVE_DIR="/path/to/project/outputs/thyroid_tert_model_v0.7.2"

# 학습 파라미터
EPOCHS=100
LR=1e-4
BAG_SIZE=2000
SEED=42
MODEL_TYPE=transmil   # abmil | transmil | acmil | dtfd | mhim | clam

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
