# Thyroid TERT Promoter Mutation Prediction

WSI 패치 임베딩 기반으로 TERT promoter mutation을 예측하는 MIL 학습 파이프라인입니다.

## 1. 문제 정의

- 태스크: 이진 분류 (`Wild=0`, `Mutant(C228T/C250T)=1`)
- 입력: 슬라이드 단위 패치 임베딩 배열(`.npy`)
- 출력: 5-fold CV 성능 지표, 체크포인트, 시각화, attention heatmap
- 지원 모델: `ABMIL`, `TransMIL`, `ACMIL`, `DTFD`, `MHIM`, `CLAM`

---

## 2. 저장소 구조

```text
thyroid_tert_public/
├── src/
│   ├── data/
│   │   ├── datasets.py                    # TERTWSIDataset
│   │   ├── tert_common.py                 # 공통 유틸 (라벨 로드, seed)
│   │   ├── create_cv_splits.py            # UNI2-H CV split 생성
│   │   ├── create_cv_splits_hoptimus.py   # H-Optimus-0 CV split 생성
│   │   ├── uni2-h/
│   │   │   ├── preprocess_data.py         # UNI2-H 임베딩 추출 (DDP)
│   │   │   └── run_embedding.sh
│   │   └── h-optimus-0/
│   │       ├── extract_features.py        # H-Optimus-0 임베딩 추출 (DDP)
│   │       └── run.sh
│   ├── models/
│   │   ├── abmil/abmil.py                 # Gated Attention MIL
│   │   ├── transmil/transmil.py           # Nyström Attention Transformer MIL
│   │   ├── acmil/acmil.py                 # Multi-branch Attention MIL
│   │   ├── dtfd/dtfd.py                   # DTFD-MIL
│   │   ├── mhim/mhim.py                   # MHIM
│   │   ├── clam/clam.py                   # CLAM
│   │   ├── mil_template.py                # MIL 베이스 클래스
│   │   └── layers.py                      # Attention 레이어, MLP 빌더
│   ├── training/
│   │   ├── main.py                        # 학습 엔트리포인트
│   │   ├── train_tert.py                  # 5-fold CV 학습/평가 루프
│   │   ├── mlflow_utils.py                # MLflow 로깅 (선택)
│   │   └── train.sh                       # 실행 스크립트
│   ├── inference/
│   │   ├── register_model.py              # MLflow Registry 등록 (선택)
│   │   ├── kpi_eval.py                    # KPI 평가
│   │   └── export_torchscript.py          # TorchScript 내보내기
│   └── evaluation/
│       ├── metric.py                      # 지표 계산
│       └── visualization.py              # Attention heatmap 생성
├── config/
│   ├── cv_splits_tert_example.json        # CV split 포맷 예시
│   └── tert_classification_template.json  # 라벨 카탈로그 템플릿
└── requirements.txt
```

---

## 3. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

권장 환경:
- Python 3.10 이상
- CUDA 사용 가능한 PyTorch 환경 (임베딩/학습용)

---

## 4. 데이터 구조

### 패치 추출

**UNI2-H 기준** (40x, 512×512 PNG):
```text
/path/to/40x_patch/
  C228T/TC_XX_0001/tile_x123_y456.png
  C250T/
  Wild/
```

**H-Optimus-0 기준** (20x, 224×224 PNG):
```text
/path/to/20x_patch/
  C228T/TC_XX_0001/TC_XX_0001_x123_y456.png
  C250T/
  Wild/
```

### 임베딩 출력

```text
/path/to/embedding/
  C228T/
    npy/TC_XX_0001.npy   # shape: (N, 1536), float32
    json/TC_XX_0001.json
  C250T/
  Wild/
```

---

## 5. 임베딩 생성

### UNI2-H

```bash
cd src/data/uni2-h
bash run_embedding.sh
```

또는 직접 실행:
```bash
python src/data/uni2-h/preprocess_data.py \
  --tile_dir /path/to/40x_patch/C228T \
  --out_dir /path/to/embedding/C228T \
  --batch_size 512
```

### H-Optimus-0

```bash
cd src/data/h-optimus-0
bash run.sh
```

---

## 6. CV Split 생성

### UNI2-H용

```bash
python src/data/create_cv_splits.py \
  --data_root /path/to/embedding \
  --label_file /path/to/Thyroid_TERT_labels.xlsx \
  --output_dir /path/to/splits \
  --n_splits 5 --seed 42
```

### H-Optimus-0용 (UNI2-H split 기반 경로 변환)

```bash
# UNI2-H split 기반으로 경로만 h_optimus_embeddings로 변환
python src/data/create_cv_splits_hoptimus.py

# 임베딩 파일 존재 여부까지 검증
python src/data/create_cv_splits_hoptimus.py --verify
```

CV split JSON 구조:
```json
{
  "seed": 42,
  "k_folds": 5,
  "folds": [
    {
      "fold": 1,
      "train_wsis": ["/path/to/embedding/C228T/npy/TC_XX_0001.npy", "..."],
      "val_wsis": ["..."],
      "test_wsis": ["..."]
    }
  ]
}
```

---

## 7. 모델 학습

```bash
cd src/training
bash train.sh
```

파라미터 오버라이드 예시:
```bash
cd src/training
CV_SPLIT_FILE=/path/to/cv_splits_tert.json \
MODEL_SAVE_DIR=/path/to/outputs/thyroid_tert_v0.1.0 \
MODEL_TYPE=abmil \
CUDA_VISIBLE_DEVICES=0 \
EPOCHS=100 \
LR=1e-4 \
BAG_SIZE=3000 \
bash train.sh
```

지원 모델: `abmil`, `transmil`, `acmil`, `dtfd`, `mhim`, `clam`

참고:
- 학습(train): `bag_size` 개수 랜덤 샘플링
- 검증/테스트: 전체 WSI (full bag)
- 데이터 누수 검사(train/val/test 중복)가 기본 적용, 중복 검출 시 학습 중단

---

## 8. 모델 아키텍처

### 공통 입력

- `X = {x_i}_{i=1..N}`, `x_i ∈ R^1536` (Foundation Model 패치 임베딩)
- UNI2-H / H-Optimus-0: 임베딩 차원 1536
- 출력: slide-level binary logits (`Wild / Mutant`)

### 지원 모델 목록

| Model | 핵심 방법 | 구현 파일 |
|-------|----------|-----------|
| `abmil` | Gated Attention weighted pooling | `src/models/abmil/abmil.py` |
| `transmil` | Nyström Attention Transformer | `src/models/transmil/transmil.py` |
| `acmil` | Multi-branch attention + top-k masking | `src/models/acmil/acmil.py` |
| `dtfd` | Double-Tier Feature Distillation | `src/models/dtfd/dtfd.py` |
| `mhim` | Masked Hard Instance Mining | `src/models/mhim/mhim.py` |
| `clam` | Clustering-constrained Attention MIL | `src/models/clam/clam.py` |

---

## 9. 주요 산출물

학습 완료 후 `MODEL_SAVE_DIR`에 생성:

- `results_cv_summary_optimal.json`
- `attention_scores/attention_scores_fold*.json`
- `checkpoints/*.pt` (`--save_model` 사용 시)
- `visualizations/` (`--generate_plots` 사용 시)
- `heatmaps/` (best fold attention heatmap)

---

## 10. MLflow 연동 (선택)

학습 전 환경변수 설정:
```bash
export MLFLOW_TRACKING_URI="https://your-mlflow-server"
export MLFLOW_EXPERIMENT_NAME="thyroid_tert"
# TLS 미검증 서버 사용 시
# export MLFLOW_TRACKING_INSECURE_TLS="true"
```

모델 Registry 등록:
```bash
python src/inference/register_model.py \
  --model_save_dir /path/to/outputs/thyroid_tert_v0.1.0

# 특정 체크포인트 지정
python src/inference/register_model.py \
  --model_save_dir /path/to/outputs/thyroid_tert_v0.1.0 \
  --checkpoint_path /path/to/outputs/thyroid_tert_v0.1.0/checkpoints/best_model_fold2.pt
```

---

## 11. 실험 결과

### UNI2-H 기준

| run | model | AUC | F1 | Acc | Sens | Spec |
|-----|-------|----:|---:|----:|-----:|-----:|
| v0.1.3 | abmil | **0.9711** | 0.8691 | 0.8756 | 0.9098 | 0.8470 |
| v0.2.2 | abmil | 0.9699 | **0.9312** | 0.9404 | 0.9209 | 0.9557 |
| v0.2.5 | abmil | 0.9652 | 0.9297 | **0.9406** | 0.8987 | 0.9739 |
| v0.3.2 | transmil | 0.9513 | 0.8745 | 0.8906 | 0.8536 | 0.9194 |

### H-Optimus-0 기준

진행 중.

---

## 12. 보안 주의사항

이 공개용 패키지는 내부 호스트/IP/절대경로를 제거하고 플레이스홀더 경로만 유지합니다.
아래 항목은 커밋하지 마세요:

- 비공개 원본 데이터셋
- 내부 인증정보/토큰
- 내부 추적 서버 주소
