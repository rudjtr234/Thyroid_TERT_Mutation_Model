# Thyroid TERT Promoter Mutation Prediction Model

갑상선 WSI 기반 **TERT promoter mutation 예측**.
- **2-class**: Wild=0 vs Mutant(C228T+C250T)=1
- **3-class**: Wild=0 / C228T=1 / C250T=2 (argmax, threshold 미적용)

지원 모델: **ABMIL, TransMIL, ACMIL, DTFD, MHIM, CLAM**

## 데이터

- **201 WSIs**: Wild 113, C228T 69, C250T 19
- CV split: 5-fold stratified, seed=42, train 140 / val ~21 / test ~40

| Encoder | 패치 | 임베딩 경로 | CV split | 학습 버전 |
|---|---|---|---|---|
| UNI2-H | 40x 512×512 | `embedding/{class}/npy/` | `cv_splits_tert_5fold_seed42.json` | v0.1.x~v0.5.x |
| H-Optimus-0 | 40x→resize 224×224 | `h_optimus_embeddings/{class}/npy/` | `..._hoptimus.json` | v0.6.x~v0.9.x |
| H-Optimus-0 | 20x 224×224 | `h_optimus_embeddings_20x/{class}/npy/` | `..._hoptimus_20x.json` | (아래 주의 참고) |
| H-Optimus-0 (3class) | 40x→resize 224×224 | `h_optimus_embeddings/{class}/npy/` | `..._hoptimus_3class.json` | v0.10.x~v0.15.x |
| H-Optimus-1 | 40x 512×512 | `h_optimus_1_embeddings_40x/{class}/npy/` | `..._hoptimus1_40x_2class.json` | hoptimus1_v0.1.x~v0.6.x |
| H-Optimus-1 (3class) | 40x 512×512 | `h_optimus_1_embeddings_40x/{class}/npy/` | `..._hoptimus1_40x_3class.json` | (미사용) |
| Hibou-L | 40x 512×512→224 | `hibou_L_embeddings_40x/{class}/npy/` | `..._hibou_L_40x_2class.json` | hibou_v0.1.x~v0.5.x |
| Hibou-L (3class) | 40x 512×512→224 | `hibou_L_embeddings_40x/{class}/npy/` | `..._hibou_L_40x_3class.json` | (미사용) |
| Virchow2 | 40x 512×512→224 | `virchow2_40x512_concat/{class}/npy/` | `..._virchow2_40x_{2,3}class.json` | virchow2_vX.X.X (학습 대기) |

- 경로 prefix: `/path/to/dataset/` · CV split prefix: `config/`
- 임베딩: Hibou-L **1024-dim**, Virchow2 **2560-dim**(CLS+mean patch concat), 나머지는 1536-dim (float32).
  Virchow2를 뺀 5종은 201/201 추출 완료
- **혼용 금지**: CV_SPLIT_FILE과 임베딩 경로는 반드시 같은 계열. `NUM_CLASSES`도 split의 class 수와,
  `IN_DIM`도 split의 `embed_dim`과 일치시킬 것

> 경로는 모두 `/path/to/...` 플레이스홀더다. 실행 전 각 스크립트의 경로 변수를 실제 환경에
> 맞게 수정해야 한다. 실제 슬라이드 ID를 포함하는 CV split JSON은 이 저장소에 포함되지 않으며,
> `config/`의 예시 템플릿을 참고해 `src/data/create_cv_splits*.py`로 생성한다.

> **주의 — 20x 임베딩은 추출만 완료, 학습 귀속 불명**
> `h_optimus_embeddings_20x/`(201 WSI, 32G, 2026-03-28 완료)와 20x CV split(2026-04-02)은 존재한다.
> 다만 v0.9.x 런의 결과 JSON·체크포인트에 `cv_split_file`이 기록돼 있지 않아 v0.9.x가 20x를
> 썼는지 40x를 썼는지 확인할 수 없다. 버전 배치상 v0.9.x는 DTFD 슬롯(v0.6=ABMIL,
> v0.7=TransMIL, v0.8=ACMIL, v0.9=DTFD)이라 **40x 계열일 가능성이 높다.**
> 20x 전용 실험이 필요하면 새 버전으로 다시 돌리고, 그때는 split 경로가 기록되는지 확인할 것.

## 인코더 메타데이터 흐름

- `create_cv_splits_hoptimus.py --encoder <이름> --embed_dim <차원>` → split JSON에 `encoder`/`embed_dim` 기록
- `train.sh`는 split JSON의 `embed_dim`을 읽어 `IN_DIM`을 자동 설정 (없으면 1536)
- `mlflow_utils.infer_encoder()`: split JSON의 `encoder`/`embed_dim` → 없으면 파일명(`hibou` / `hoptimus1` / `hoptimus`)
  → 그래도 없으면 `in_dim`(1024=Hibou-L, 768=Hibou-B)으로 인코더명·패치 정보를 정해 MLflow param/tag/description에 기록.
  `hoptimus1`은 `hoptimus`를 부분 문자열로 포함하므로 반드시 먼저 검사한다
- `register_model.py`: 체크포인트 `model_hparams.in_dim`으로 `embedding` 태그 결정 (하드코딩 시 Hibou가 UNI2-H로 등록됨)

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
# Hibou-L 추출 스크립트는 이 저장소에 없음 (BRAF 저장소 src/data/hibou/ 참고)
# Virchow2 (2560-dim, HF 토큰 필요): 환경 점검 → 추출 → split
cd src/data/virchow2 && python check_env.py && bash run.sh   # GPUS=2,3,4,6 NPROC=4 로 다중 GPU
bash scripts/make_virchow2_splits.sh        # 201장 완료·차원 2560 확인 후 split 생성

# CV split 생성 (경로 변환 + 인코더 메타 기록)
python src/data/create_cv_splits_hoptimus.py [--new_root ... --encoder Hibou-L --embed_dim 1024] [--verify]

# 학습: train.sh 상단 [1]~[10] 중 CV_SPLIT_FILE 하나만 주석 해제 + 아래 변수 수정
# CV_SPLIT_FILE / MODEL_SAVE_DIR / MODEL_TYPE / BAG_SIZE / NUM_CLASSES  (IN_DIM은 자동)
cd src/training && bash train.sh            # GPU_ID=5 / PYTHON=/path/to/python / TE_BAG_SIZE=800 등으로 덮어쓰기

# bag_size 스윕을 여러 GPU에 분배 (GPU당 run 1개씩 직렬, 기존 출력 있으면 거부)
bash scripts/sweep.sh --gpus 5,6 --bags 200,400,800,1600 --model abmil --prefix thyroid_tert_v9.9 [--dry-run]
bash scripts/sweep_status.sh thyroid_tert_v9   # run별 AUC 요약

# MLflow Registry 등록
python src/inference/register_model.py --model_save_dir outputs/thyroid_tert_model_vX.X.X
```

## 출력

`outputs/<run_dir>/`: `results_cv_summary_optimal.json`, `checkpoints/`, `attention_scores/`, `visualizations/`, `heatmaps/`

- 네이밍: `thyroid_tert_model_vX.X.X` (UNI2-H / H-Optimus-0), `thyroid_tert_hoptimus1_vX.X.X` (H-Optimus-1),
  `thyroid_tert_hibou_vX.X.X` (Hibou-L), `thyroid_tert_virchow2_vX.X.X` (Virchow2)

## 실험 결과 (5-fold CV test 평균, 각 마이너 버전의 best run)

출처: MLflow experiment `thyroid_tert` (2026-10-28 기준 186 runs) + 이후 Hibou-L 50 runs.
각 마이너 버전은 bag_size 스윕이고 아래는 그중 test AUC 최고치다.

### 2-class (Wild vs Mutant)

| Encoder | Model | best run | AUC | F1 | Acc | Sens | Spec |
|---|---|---|---:|---:|---:|---:|---:|
| UNI2-H 40x | abmil | v0.2.2 | 0.9699 | **0.9312** | **0.9404** | 0.9209 | 0.9557 |
| UNI2-H 40x | transmil | v0.3.5 | 0.9585 | 0.8724 | 0.8910 | 0.8418 | 0.9285 |
| UNI2-H 40x | acmil | v0.4.4 | 0.9667 | 0.9233 | 0.9356 | 0.8876 | 0.9739 |
| UNI2-H 40x | dtfd | v0.5.3 | 0.9676 | 0.8885 | 0.9059 | 0.8536 | 0.9470 |
| H-Optimus-0 | abmil | v0.6.0 | **0.9716** | 0.8997 | 0.9106 | 0.9098 | 0.9107 |
| H-Optimus-0 | transmil | v0.7.7 | 0.9633 | 0.9034 | 0.9206 | 0.8542 | 0.9731 |
| H-Optimus-0 | acmil | v0.8.0 | 0.9645 | 0.8982 | 0.9156 | 0.8536 | 0.9644 |
| H-Optimus-0 | dtfd | v0.9.2 | 0.9709 | 0.9149 | 0.9306 | 0.8765 | 0.9735 |
| H-Optimus-1 40x | abmil | hoptimus1_v0.1.9 | 0.9617 | 0.9026 | 0.9209 | 0.8542 | 0.9739 |
| H-Optimus-1 40x | transmil | hoptimus1_v0.2.5 | 0.9688 | 0.8935 | 0.9055 | 0.8987 | 0.9107 |
| H-Optimus-1 40x | acmil | hoptimus1_v0.3.3 | 0.9543 | 0.9058 | 0.9209 | 0.8647 | 0.9652 |
| H-Optimus-1 40x | dtfd | hoptimus1_v0.4.5 | 0.9612 | 0.9026 | 0.9209 | 0.8542 | 0.9739 |
| H-Optimus-1 40x | mhim | hoptimus1_v0.5.6 | 0.9619 | 0.8941 | 0.9109 | 0.8647 | 0.9462 |
| H-Optimus-1 40x | clam | hoptimus1_v0.6.6 | 0.9596 | 0.9039 | 0.9207 | 0.8654 | 0.9644 |
| Hibou-L 40x | abmil | hibou_v0.1.2 | 0.9594 | 0.8579 | 0.8759 | 0.8542 | 0.8937 |
| Hibou-L 40x | transmil | hibou_v0.2.7 | 0.9614 | 0.8931 | 0.9105 | 0.8654 | 0.9470 |
| Hibou-L 40x | acmil | hibou_v0.3.7 | 0.9609 | 0.8952 | 0.9107 | 0.8647 | 0.9470 |
| Hibou-L 40x | dtfd | hibou_v0.4.7 | 0.9590 | 0.9078 | 0.9207 | 0.8876 | 0.9462 |
| Hibou-L 40x | mhim | hibou_v0.5.9 | 0.9613 | 0.8884 | 0.9057 | 0.8654 | 0.9375 |

### 3-class (Wild / C228T / C250T, H-Optimus-0 40x)

| Model | best run | AUC | F1 | Acc | Sens | Spec |
|---|---|---:|---:|---:|---:|---:|
| abmil | v0.10.0 | 0.8731 | 0.5584 | 0.7215 | 0.5583 | 0.8487 |
| transmil | v0.11.0 | **0.8894** | **0.6637** | **0.8409** | **0.6789** | **0.9144** |
| acmil | v0.12.6 | 0.8787 | 0.5556 | 0.7909 | 0.5785 | 0.8843 |
| dtfd | v0.13.2 | 0.8809 | 0.6025 | 0.7960 | 0.6047 | 0.8806 |
| mhim | v0.14.4 | 0.8871 | 0.5245 | 0.7761 | 0.5524 | 0.8567 |
| clam | v0.15.1 | 0.8879 | 0.5917 | 0.8307 | 0.6191 | 0.9087 |

### 요약

- 2-class 최고: **H-Optimus-0 + ABMIL (v0.6.0, AUC 0.9716)**. UNI2-H ABMIL(v0.2.2)은 AUC는
  근소하게 낮지만 F1·Acc는 최고다.
- 인코더 교체 효과는 작다. UNI2-H / H-Optimus-0 / H-Optimus-1이 모두 AUC 0.95~0.97 구간에
  몰려 있고, fold 간 분산을 고려하면 유의한 차이로 보기 어렵다.
- 3-class는 2-class보다 크게 낮다 (AUC ~0.88, F1 ~0.6). C250T가 19장뿐이라 소수 클래스
  성능이 전체를 끌어내린다. 3-class에서는 TransMIL이 전 지표 1위다.
- Hibou-L은 5개 MIL 모델이 모두 AUC 0.959~0.961 (best transmil hibou_v0.2.7, 0.9614).
- 인코더 4종 best의 AUC 폭은 0.0102, fold std는 약 0.019 → 인코더 순위는 통계적으로 성립하지 않는다.
- **선택 편향 실측**: val AUC로 run을 고르면 test AUC가 test 기준 선택보다 평균 0.013 (최대 0.038)
  낮다 (2-class 15개 스윕). 편향만으로 인코더 간 폭을 넘는다. val 기준으로는 ABMIL이 가장 안정적.
- 다음 인코더(Virchow2)부터: val 기준 선택 · test 1회 · fold별 AUC + 95% CI · DeLong/bootstrap 유의성 검정.

> **해석 주의**: 각 마이너 버전은 bag_size 스윕이고 위 표는 test AUC 기준 best를 뽑은 값이다.
> 즉 test set으로 하이퍼파라미터를 고른 셈이라 낙관적으로 편향돼 있다. 모델 간 차이가
> fold 분산 안에 들어오는 상황에서는 이 편향이 순위를 좌우할 수 있으니, 논문용 수치는
> val 기준 선택으로 다시 뽑을 것.

## MLflow

- Tracking URI: `MLFLOW_TRACKING_URI` 환경변수로 지정 (기본 `http://localhost:5000`)
  - 자체 서명 인증서 사용 시 `MLFLOW_TRACKING_INSECURE_TLS=true`
- Experiment: `thyroid_tert` / Registry: `thyr-tert`
- 기록 param/tag: `encoder`, `in_dim` (위 "인코더 메타데이터 흐름" 참고)

## 주의사항
- Attention heatmap 색상: Mutant = 짙은파랑→빨강 / Wild = 옅은파랑→짙은파랑 (`visualization.py`)
- `register_model.py`는 `--with_torchscript` 사용 시 체크포인트에서 모델 타입을 읽어
  TorchScript를 생성한다 (ABMIL 외 모델도 지원).
- 임베딩 추출 중 `OSError: [Errno 24] Too many open files` 방지:
  `torch.multiprocessing.set_sharing_strategy("file_system")` + `ulimit -n 65536`
  (타일 수가 많은 슬라이드에서 fd 한도 초과로 중단됨)
- H-Optimus-1 / Hibou-L은 gated repo — HuggingFace 토큰 인증 필요, 토큰·모델 캐시는 서버별로 별도 설정
- H-Optimus-1 정규화 상수·출력 차원은 H-Optimus-0과 동일 (코드 차이는 모델 ID 한 줄)
- Hibou-L 정규화는 다르다: `mean=(0.7068, 0.5755, 0.7220)`, `std=(0.1950, 0.2316, 0.1816)`, 출력 1024-dim
- Virchow2: 출력 `[B, 261, 1280]` = CLS 1 + register 4 + patch 256. register 4개를 건너뛰고(`out[:, 5:]`)
  `concat(CLS, mean(patch))` = 2560-dim. timm 인자 `reg_tokens=4, mlp_ratio=5.3375, SwiGLUPacked, SiLU` 필수,
  정규화는 ImageNet 값. fp16 고정(Volta는 bf16 미지원), sm_80 미만에서는 FlashAttention SDPA를 끈다
- Virchow2 추출기는 슬라이드마다 DataLoader를 새로 만든다 → `persistent_workers=False` + fd 한도 상향 필수.
  NCCL 타임아웃은 2시간(느린 슬라이드에서 `all_gather` 대기)
- `scripts/common_env.sh`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`로 GPU 번호를 nvidia-smi와 일치시킨다. GPU 가드/전력 캡은
  `TE_GUARD_HOST_PATTERN`에 맞는 호스트에서만 동작한다 (기본 비활성)
- `train.sh`는 학습 전 인터프리터 / split 파일 / `torch.cuda.is_available()`을 점검하고 실패 시 즉시 종료한다
  (torch CUDA 빌드가 드라이버보다 새로우면 CUDA 초기화 실패 — 맞는 환경을 `PYTHON=`으로 지정)
