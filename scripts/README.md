# scripts/ — 서버 공용 실행 도구

GPU 구성이 서로 다른 서버 어디서든 **같은 명령으로** 학습을 돌리기 위한 스크립트 모음.

## 왜 필요했나

| 문제 | 증상 | 해결 |
|---|---|---|
| conda 경로가 서버마다 다르다 | 한 서버용 경로가 다른 서버에 없어 `Python 인터프리터를 찾을 수 없습니다` | `common_env.sh` 가 후보를 순회하며 `torch+sklearn+numpy+pandas` 가 import 되는 python 을 고른다 |
| nvidia-smi 와 PyTorch 의 GPU 번호가 다르다 | RTX 3080 / V100S 가 섞인 서버에서 PyTorch 기본값(FASTEST_FIRST)이 번호를 재정렬한다. 비었다고 본 GPU 에 안 붙는다 | `CUDA_DEVICE_ORDER=PCI_BUS_ID` 를 강제해 nvidia-smi 번호와 1:1 일치시킨다 |
| GPU 번호가 스크립트에 하드코딩 | 매번 `train.sh` 를 열어 `CUDA_VISIBLE_DEVICES` 를 고쳐야 했다 | `GPU_ID=5 bash train.sh` |
| 남는 GPU 를 못 쓴다 | bag_size 스윕을 1장에서 직렬로 돌렸다 | `sweep.sh` 로 여러 GPU 에 분배 |
| 추출과 학습이 같은 GPU 를 잡는다 | 임베딩 추출이 느려지거나 10GB GPU 에서 OOM | 가드 대상 서버에서는 학습용 GPU 만 허용 (`te_guard_gpu_choice`) |

번호는 모두 **nvidia-smi 기준**이다 (`common_env.sh` 가 보장).

## common_env.sh

학습 스크립트가 `source` 하는 공통 설정. 직접 실행하지 않는다.

제공하는 것:
- `PYTHON` — 자동 탐지한 인터프리터. `PYTHON=/path/to/python` 로 덮어쓰거나 `TE_PYTHON_CANDIDATES=a:b` 로 후보 추가
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` — GPU 번호 고정
- `te_check_cuda` — CUDA 가능 여부 + 보이는 GPU 의 이름/VRAM/여유메모리/sm 버전 출력
- `te_warn_busy_gpu` — 고른 GPU 에 1GB 넘게 쓰는 프로세스가 있으면 경고
- `te_guard_gpu_choice` / `te_power_guard` — 가드 대상 서버 전용 (아래)
- `te_check_transmil` — `nystrom_attention` 설치 확인 (TransMIL 전용)

## 단일 학습

```bash
cd src/training
bash train.sh              # 기본 GPU
GPU_ID=5 bash train.sh     # GPU 지정
```

`train.sh` 안의 `CV_SPLIT_FILE` / `MODEL_SAVE_DIR` / `MODEL_TYPE` / `BAG_SIZE` /
`NUM_CLASSES` 를 고치는 기존 방식은 그대로다.

환경변수로 일시 덮어쓰기도 된다 (스윕이 이걸 쓴다):

```bash
GPU_ID=5 TE_BAG_SIZE=800 TE_MODEL_TYPE=abmil \
  TE_MODEL_SAVE_DIR=$PWD/../../outputs/thyroid_tert_v9.9.0 \
  bash train.sh
```

사용 가능: `TE_CV_SPLIT_FILE` `TE_MODEL_SAVE_DIR` `TE_NUM_CLASSES`
`TE_EPOCHS` `TE_LR` `TE_BAG_SIZE` `TE_SEED` `TE_MODEL_TYPE`

## sweep.sh — 여러 GPU 에 스윕 분배

bag_size 스윕을 GPU 장수만큼 병렬로 돌린다.

```bash
cd scripts

# 먼저 계획만 확인
bash sweep.sh --gpus 5,6 --bags 200,400,800,1600 \
              --model abmil --prefix thyroid_tert_v9.9 --dry-run

# 실행
bash sweep.sh --gpus 5,6 --bags 200,400,800,1600 \
              --model abmil --prefix thyroid_tert_v9.9
```

- 출력: `outputs/thyroid_tert_v9.9.0` ~ `.3` (bags 순서대로)
- 로그: `outputs/sweep_logs/<prefix>.<i>_bag<N>_gpu<G>.log`
- 기존 출력 디렉토리가 있으면 **실행을 거부한다** (결과 덮어쓰기 방지)
- GPU 1장에는 run 을 하나씩 직렬로 올린다 (동시에 올려 VRAM 터지는 것 방지)

옵션: `--gpus` `--bags` `--model` `--prefix` `--split` `--classes` `--dry-run`

### DDP 를 쓰지 않는 이유

MIL 학습은 미리 뽑아 둔 임베딩 벡터만 쓴다. 모델이 작아 GPU 1장에서
VRAM 2GB 정도로 끝나므로, 한 번의 학습을 DDP 로 2장에 쪼개도 통신
오버헤드 때문에 거의 빨라지지 않는다. 반면 스윕은 독립적인 run 이
여러 개라 GPU 마다 다른 run 을 붙이면 장수만큼 선형으로 빨라진다.
(임베딩 **추출** 쪽은 사정이 달라 DDP 를 쓴다 — `src/data/virchow2/`)

## 가드 대상 서버 (선택 기능)

임베딩 추출과 MIL 학습을 한 서버에서 같이 돌릴 때 쓰는 보호 장치다.
`TE_GUARD_HOST_PATTERN` 에 호스트명 glob 을 주면 그 서버에서만 켜진다.
비워 두면(기본) 아무 서버에서도 동작하지 않는다.

```bash
export TE_GUARD_HOST_PATTERN="*gpu01*"   # 이 호스트에서만 가드 ON
export TE_TRAINING_GPUS="5 6"            # 학습에 허용할 GPU
```

### 1) 학습용 GPU 제한

추출이 쓰는 GPU 를 학습이 고르면 **실행을 거부한다.** 추출과 연산을
경쟁하면 추출이 느려지고, 10GB 급 GPU 는 VRAM 여유가 3GB 남짓이라
피크에서 OOM 으로 죽기 때문이다.

추출이 끝났거나 일부러 써야 하면 우회할 수 있다:

```bash
TE_ALLOW_BUSY_GPU=1 GPU_ID=2 bash train.sh
```

### 2) 전력 가드

혼종 GPU 서버 한 대에서 **Xid 79** (`GPU has fallen off the bus`) 로 GPU 여러
장이 동시에 버스에서 이탈한 적이 있다. 개별 GPU 고장이 아니라 공통
전원(PSU/power rail) 또는 PCIe fabric 문제로 추정되어, 그 서버에서는
`gpu_guard.sh` 로 **전력 캡을 자동 적용**해 동시 피크를 줄인다.

| GPU | 정격 | 캡 |
|---|---|---|
| Tesla V100S | 250W | 200W |
| RTX 3080 | 320W | 250W |
| RTX 6000 Ada | 300W | 250W |

- 캡 적용에는 root 권한이 필요하다. 없으면 경고만 내고 정격으로 진행한다
  (학습이 권한 때문에 아예 못 도는 것보다 낫다)
- `gpu_guard_watch_xid` 는 GPU 가 조회 불가 상태가 되면 NCCL watchdog 을
  기다리지 않고 즉시 작업을 중단시킨다
- 캡 확인: `nvidia-smi --query-gpu=index,name,power.draw,power.limit --format=csv`

## 사전 점검 / 모니터링

```bash
bash scripts/check_server.sh      # python·패키지·CUDA·split·임베딩 경로·쓰기 권한 점검
bash scripts/gpu_status.sh        # GPU + 돌고 있는 작업 1회 출력
bash scripts/gpu_status.sh -w     # 5초마다 갱신

bash scripts/sweep_status.sh                   # 모든 run 결과 (AUC 내림차순)
bash scripts/sweep_status.sh thyroid_tert_v9   # 이름 필터
```

`gpu_status.sh` 는 프로세스 커맨드라인을 파싱해 임베딩 추출과 학습을
구분하고, 학습은 모델명 / `bag_size` / 출력 디렉토리까지 보여준다
(`--model_type`, `--model_name` 둘 다 인식한다).

## Virchow2 split 생성

```bash
bash scripts/make_virchow2_splits.sh              # 2class (기본)
CLASSES="2 3" bash scripts/make_virchow2_splits.sh
```

추출이 201장 전부 끝났는지, 임베딩 차원이 2560 인지 확인한 뒤 기존 split 의
fold 구성을 그대로 두고 임베딩 경로만 바꿔 생성한다 (인코더 간 공정 비교).
