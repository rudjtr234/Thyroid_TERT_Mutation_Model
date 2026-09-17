# Thyroid TERT Promoter Mutation Prediction

## Overview

<p align="center">
  <img src="./images/tert_pipeline.png" alt="Pipeline Overview" width="90%"/>
</p>

WSI(Whole Slide Image)에서 추출된 patch-level embedding 기반으로
**TERT promoter mutation**(Wild vs Mutant)을 예측하는 MIL 파이프라인입니다.

- Task: Binary classification (`Wild=0`, `Mutant(C228T/C250T)=1`) 또는 3-class (`Wild=0`, `C228T=1`, `C250T=2`)
- Input: UNI2-H 또는 H-Optimus-0 기반 1536-dim patch embedding (`.npy`)
- Eval: Stratified 5-fold CV, Test는 Full WSI(전체 patch) 평가

## Dataset

- 총 WSI: **201개** (`Wild 113`, `C228T 69`, `C250T 19`)
- 원본 SVS 경로: `/path/to/dataset/root/thyroid`
- 슬라이드 라벨은 SVS 파일명에 없고, 별도 Excel 기준으로 관리
- 패치셋은 `40x 512×512`와 `20x 224×224 @ 0.5MPP`를 모두 보유

### 실제 데이터 스냅샷 (2026-05-08 확인)

| Asset | Verified Count | Size | Notes |
|---|---:|---:|---|
| `thyroid/` | 201 `.svs` | 328.95 GiB | 원본 WSI |
| `40x_patch/` | 201 slide dirs / 4,273,287 `.png` | 1,770.08 GiB | UNI2-H 및 H-Optimus-0 40x 실험 원본 |
| `20x_patch/` | 201 slide dirs / 5,259,005 `.png` + 201 `.done` | 514.99 GiB | H-Optimus-0 20x 실험 원본 |
| `embedding/` | 201 `.npy` + 201 slide `.json` + 3 `all_slides.json` | 25.22 GiB | UNI2-H 40x 임베딩 |
| `h_optimus_embeddings/` | 201 `.npy` + 201 slide `.json` + 3 `all_slides.json` | 25.24 GiB | H-Optimus-0 40x 임베딩 |
| `h_optimus_embeddings_20x/` | 201 `.npy` + 201 slide `.json` + 3 `all_slides.json` | 31.06 GiB | H-Optimus-0 20x 임베딩 |

- `20x_patch/`의 regular file 총수는 `5,259,206`이며, 이 중 `201`개는 슬라이드별 완료 표시용 `*.done` 파일
- 각 embedding 계열 디렉터리의 regular file 총수는 `405`개이며, slide-level metadata `201`개 외에 클래스별 `all_slides.json`이 1개씩 추가 존재
- 전체 저장 용량은 약 `2,696 GiB`

## Supported MIL Models

| `model_type` | 설명 |
|---|---|
| `abmil` | Gated attention 기반 기본 MIL |
| `transmil` | Transformer MIL (Nyström attention + PPEG) |
| `acmil` | Attention-challenging MIL |
| `dtfd` | Double-tier feature distillation MIL |
| `mhim` | Masked hard instance mining MIL |
| `clam` | CLAM-SB style MIL |

## Training / Evaluation Protocol

| 항목 | 설정 |
|---|---|
| Train | Bag sampling (`bag_size`) |
| Validation | Full WSI (all patches) |
| Test | Full WSI (all patches) |
| Threshold | 0.5 고정 |
| Optimizer | Adam (`lr=1e-4`, `weight_decay=1e-4`) |
| Scheduler | ReduceLROnPlateau (`patience=15`, `factor=0.5`) |
| Early stopping | Validation **loss** 기준 (`patience=8`, `min_delta=0.001`) |
| Loss | CrossEntropyLoss |

## Quick Start

### 1) 임베딩 추출

```bash
# H-Optimus-0 (V100 3장, DDP, batch_size=512)
cd src/data/h-optimus-0
bash run.sh

# UNI2-H
cd src/data/uni2-h
bash run_embedding.sh
```

### 2) CV split 생성 (H-Optimus용)

```bash
# UNI2-H split 경로를 h_optimus_embeddings_20x로 변환
python src/data/create_cv_splits_hoptimus.py \
  --old_root /path/to/dataset/root/embedding \
  --new_root /path/to/dataset/root/h_optimus_embeddings_20x \
  --output_json config/cv_splits_tert_5fold_seed42_hoptimus_20x.json

# 임베딩 파일 존재 여부 검증
python src/data/create_cv_splits_hoptimus.py \
  --old_root /path/to/dataset/root/embedding \
  --new_root /path/to/dataset/root/h_optimus_embeddings_20x \
  --output_json config/cv_splits_tert_5fold_seed42_hoptimus_20x.json \
  --verify
```

### 3) 학습

```bash
cd src/training
bash train.sh
```

`train.sh`에서 수정할 주요 항목:
- `CV_SPLIT_FILE`: UNI2-H / H-Optimus 주석 전환
- `MODEL_SAVE_DIR`
- `MODEL_TYPE` (`abmil` | `transmil` | `acmil` | `dtfd` | `mhim` | `clam`)
- `BAG_SIZE`, `EMBED_DIM`, `ATTN_DIM`

### 4) CLI 직접 실행

```bash
python src/training/main.py \
  --model_save_dir outputs/thyroid_tert_model_vX.X.X \
  --cv_split_file config/cv_splits_tert_5fold_seed42_hoptimus_20x.json \
  --model_type abmil \
  --epochs 100 \
  --lr 1e-4 \
  --bag_size 1000 \
  --save_model \
  --save_best_only \
  --generate_plots
```

## Project Structure

```text
src/
├── data/
│   ├── uni2-h/
│   │   ├── preprocess_data.py        # UNI2-H 임베딩 추출
│   │   └── run_embedding.sh
│   ├── h-optimus-0/
│   │   ├── extract_features.py       # H-Optimus-0 임베딩 추출 (DDP)
│   │   └── run.sh
│   ├── create_cv_splits.py           # UNI2-H CV split 생성
│   ├── create_cv_splits_hoptimus.py  # H-Optimus-0 CV split 생성 (경로 변환)
│   ├── datasets.py
│   └── tert_common.py
├── models/
│   ├── abmil.py
│   ├── abmil/
│   ├── acmil/
│   ├── dtfd/
│   ├── mhim/
│   ├── clam/
│   ├── transmil/
│   ├── layers.py
│   └── mil_template.py
├── training/
│   ├── main.py
│   ├── train_tert.py
│   ├── train.sh
│   ├── mlflow_utils.py
│   └── register_model.py  (→ src/inference/register_model.py)
├── inference/
│   ├── register_model.py
│   ├── export_torchscript.py
│   └── kpi_eval.py
└── evaluation/
    ├── metric.py
    └── visualization.py

config/
├── cv_splits_tert_5fold_seed42.json              # UNI2-H CV split (40x)
├── cv_splits_tert_5fold_seed42_hoptimus.json     # H-Optimus-0 CV split (40x)
├── cv_splits_tert_5fold_seed42_hoptimus_20x.json # H-Optimus-0 CV split (20x)
├── cv_splits_tert_5fold_seed42_hoptimus_3class.json # H-Optimus-0 CV split (3-class, 40x)
└── tert_classification.json

outputs/
└── thyroid_tert_model_vX.X.X/
    ├── checkpoints/
    ├── results_cv_summary_optimal.json
    ├── attention_scores/
    ├── visualizations/
    └── heatmaps/
```

## Embedding Variants

| Encoder | 패치 기반 | 경로 | 학습 버전 | 상태 |
|---|---|---|---|---|
| UNI2-H | 40x `512×512` | `/path/to/dataset/root/embedding/{class}/npy/` | `v0.1.x ~ v0.5.x` | 완료 |
| H-Optimus-0 | 40x `512×512` → resize `224×224` | `/path/to/dataset/root/h_optimus_embeddings/{class}/npy/` | `v0.6.x ~ v0.8.x` | 완료 |
| H-Optimus-0 | 20x `224×224 @ 0.5MPP` | `/path/to/dataset/root/h_optimus_embeddings_20x/{class}/npy/` | `v0.9.x ~` | 임베딩 완료 / 학습 예정 |
| H-Optimus-0 (3-class) | 40x `512×512` | `/path/to/dataset/h_optimus_embeddings/{Wild,C228T,C250T}/npy/` | `v0.10.x ~` | 학습 진행 중 |

## Experiment Results

### UNI2-H 기준

| run_dir | model_type | AUC | F1 | Acc | Sens | Spec |
|---|---|---:|---:|---:|---:|---:|
| thyroid_tert_v0.1.3 | abmil | **0.9711** | 0.8691 | 0.8756 | 0.9098 | 0.8470 |
| thyroid_tert_model_v0.2.2 | abmil | 0.9699 | **0.9312** | 0.9404 | 0.9209 | 0.9557 |
| thyroid_tert_model_v0.2.5 | abmil | 0.9652 | 0.9297 | **0.9406** | 0.8987 | 0.9739 |
| thyroid_tert_model_v0.3.2 | transmil | 0.9513 | 0.8745 | 0.8906 | 0.8536 | 0.9194 |

### H-Optimus-0 기준 (40x→resize 224×224, v0.6.x~v0.8.x)

| run_dir | model_type | AUC | F1 | Acc | Sens | Spec |
|---|---|---:|---:|---:|---:|---:|
| thyroid_tert_model_v0.6.0 | abmil | **0.9716** | 0.8997 | 0.9106 | 0.9098 | 0.9107 |
| thyroid_tert_model_v0.7.2 | transmil | 0.9575 | 0.8926 | 0.9106 | 0.8536 | 0.9549 |
| thyroid_tert_model_v0.8.9 | acmil | 0.9626 | 0.9031 | 0.9104 | 0.9209 | 0.9008 |

### H-Optimus-0 기준 (20x 224×224, v0.9.x~)

- `h_optimus_embeddings_20x/` 기준 임베딩은 `201/201` 완료
- 완료 실험 결과는 아직 없음

### H-Optimus-0 3-class 기준 (Wild/C228T/C250T, v0.10.x~)

- Task: 3-class (`Wild=0`, `C228T=1`, `C250T=2`), argmax 예측 (threshold 미적용)
- CV split: `config/cv_splits_tert_5fold_seed42_hoptimus_3class.json` (201 WSI: Wild 113 / C228T 69 / C250T 19)
- `thyroid_tert_model_v0.10.0` 학습 진행 중 (`model_type=abmil`, `bag_size=500`)
- 완료 실험 결과는 아직 없음

## Registration (MLflow Model Registry)

```bash
cd src/inference
python register_model.py --model_save_dir ../../outputs/thyroid_tert_model_vX.X.X
```

- Tracking URI: `http://localhost:5000`
- Registered Model: `thyr-tert`
