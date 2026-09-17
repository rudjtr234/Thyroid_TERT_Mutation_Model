# Thyroid TERT Promoter Mutation Prediction Model

갑상선 WSI 기반 **TERT promoter mutation 분류**.
- 이진분류: Wild=0 vs Mutant(C228T+C250T)=1
- 3-class: Wild=0, C228T=1, C250T=2

지원 모델: **ABMIL, TransMIL, ACMIL, DTFD, MHIM, CLAM**

## 데이터

- **201 WSIs**: Wild 113, C228T 69, C250T 19
- CV split: 5-fold stratified, seed=42, train 140 / val ~21 / test ~40

| Encoder | 패치 | 임베딩 경로 | CV split | 학습 버전 |
|---|---|---|---|---|
| UNI2-H | 40x 512×512 | `/path/to/dataset/embedding/{class}/npy/` | `config/cv_splits_tert_5fold_seed42.json` | v0.1.x~v0.5.x |
| H-Optimus-0 | 40x→resize 224×224 | `/path/to/dataset/h_optimus_embeddings/{class}/npy/` | `config/cv_splits_tert_5fold_seed42_hoptimus.json` | v0.6.x~v0.8.x |
| H-Optimus-0 | 20x 224×224 | `/path/to/dataset/h_optimus_embeddings_20x/{class}/npy/` | `config/cv_splits_tert_5fold_seed42_hoptimus_20x.json` | v0.9.x~ |
| H-Optimus-0 (3-class) | 40x 512×512 | `/path/to/dataset/h_optimus_embeddings/{Wild,C228T,C250T}/npy/` | `config/cv_splits_tert_5fold_seed42_hoptimus_3class.json` | v0.10.x~ |
| H-Optimus-1 | 40x 512×512 → 224 | `/path/to/dataset/h_optimus_1_embeddings_40x/{class}/npy/` | UNI2-H split 경로 변환 | 추출 중단 (10/201) |

- 임베딩: 1536-dim float32
- 혼용 금지: CV_SPLIT_FILE과 임베딩 경로 반드시 같은 계열 사용

> 경로는 모두 `/path/to/...` 플레이스홀더다. 실행 전 `train.sh` 및 각 스크립트의
> 경로 변수를 실제 환경에 맞게 수정해야 한다. 실제 슬라이드 ID를 포함하는
> CV split JSON은 이 저장소에 포함되지 않으며, `config/`의 예시 템플릿을 참고해
> `src/data/create_cv_splits*.py`로 생성한다.

## 모델

| model_type | 파일 |
|---|---|
| `abmil` | `src/models/abmil/abmil.py` |
| `transmil` | `src/models/transmil/transmil.py` (`nystrom-attention` 필수) |
| `acmil` | `src/models/acmil/acmil.py` |
| `dtfd` | `src/models/dtfd/dtfd.py` |
| `mhim` | `src/models/mhim/mhim.py` |
| `clam` | `src/models/clam/clam.py` |

## 학습 파이프라인

- 엔트리포인트: `src/training/main.py` → `train_tert.py`의 `run_k_fold_cv()`
- train: `bag_size` 랜덤 샘플링 / val·test: full WSI
- Optimizer: Adam `lr=1e-4, weight_decay=1e-4`
- Scheduler: `ReduceLROnPlateau(patience=15, factor=0.5)`
- Early stopping: `patience=8, min_delta=0.001, monitor=val_loss` (주석에 val_auc로 표기된 곳 있으나 실제 구현은 val_loss 기준)

## 실행

```bash
# 임베딩 추출 (완료된 .npy는 건너뛰는 resume 구조)
cd src/data/h-optimus-0 && bash run.sh      # H-Optimus-0 (V100 3장, bs=512)
cd src/data/h-optimus-1 && bash run.sh      # H-Optimus-1 (RTX 3080 3장, bs=32, HF 토큰 필요)
cd src/data/uni2-h && bash run_embedding.sh # UNI2-H

# CV split 생성 (H-Optimus용)
python src/data/create_cv_splits_hoptimus.py [--verify]

# 학습: train.sh에서 아래 4개 변수만 수정
# CV_SPLIT_FILE / MODEL_SAVE_DIR / MODEL_TYPE / BAG_SIZE
# UNI2-H (40x)        → cv_splits_tert_5fold_seed42.json
# H-Optimus (40x)     → cv_splits_tert_5fold_seed42_hoptimus.json        (v0.6~v0.8)
# H-Optimus (20x)     → cv_splits_tert_5fold_seed42_hoptimus_20x.json    (v0.9~)
# H-Optimus (3-class) → cv_splits_tert_5fold_seed42_hoptimus_3class.json (v0.10~)
cd src/training && bash train.sh

# MLflow Registry 등록
python src/inference/register_model.py --model_save_dir outputs/thyroid_tert_model_vX.X.X
```

## 출력

`outputs/<run_dir>/`: `results_cv_summary_optimal.json`, `checkpoints/`, `attention_scores/`, `visualizations/`, `heatmaps/`

## 실험 결과 (5-fold CV test 평균)

| Encoder | 패치 | Model | AUC | F1 | Acc | Sens | Spec |
|---|---|---|---:|---:|---:|---:|---:|
| UNI2-H | 40x 512×512 | ABMIL (v0.2.2) | 0.9699 | 0.9312 | 0.9404 | 0.9209 | 0.9557 |
| H-Optimus-0 | 40x→resize 224×224 | ABMIL (v0.6.0) | 0.9716 | 0.8997 | 0.9106 | 0.9098 | 0.9107 |
| H-Optimus-0 | 40x→resize 224×224 | TransMIL (v0.7.2) | 0.9575 | 0.8926 | 0.9106 | 0.8536 | 0.9549 |

## MLflow

- Tracking URI: `MLFLOW_TRACKING_URI` 환경변수로 지정 (기본 `http://localhost:5000`)
  - 자체 서명 인증서 사용 시 `MLFLOW_TRACKING_INSECURE_TLS=true`
- Experiment: `thyroid_tert` / Registry: `thyr-tert`

## 주의사항
- Attention heatmap 색상: Mutant = 짙은파랑→빨강 / Wild = 옅은파랑→짙은파랑 (`visualization.py`)
- `register_model.py`는 `--with_torchscript` 사용 시 체크포인트에서 모델 타입을 읽어
  TorchScript를 생성한다 (ABMIL 외 모델도 지원).
- 임베딩 추출 중 `OSError: [Errno 24] Too many open files` 방지:
  `torch.multiprocessing.set_sharing_strategy("file_system")` + `ulimit -n 65536`
  (타일 수가 많은 슬라이드에서 fd 한도 초과로 중단됨)
- H-Optimus-1은 gated repo — HuggingFace 토큰 인증 필요, 토큰·모델 캐시는 서버별로 별도 설정
- H-Optimus-1 정규화 상수·출력 차원은 H-Optimus-0과 동일 (코드 차이는 모델 ID 한 줄)
- **H-Optimus-1 임베딩은 201장 중 10장(5%)에서 중단된 상태.** 사유는 GPU 탈락에 따른
  NCCL watchdog timeout + 출력 디렉토리 쓰기 권한 문제. 환경 복구 후 재실행하면 이어서 진행됨
