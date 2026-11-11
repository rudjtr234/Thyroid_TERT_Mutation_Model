#!/bin/bash
# Virchow2 TERT 임베딩 추출 (201 WSI, 40x 512 → 224 resize)
# 실행: cd src/data/virchow2 && bash run.sh
#
# === 이 스크립트는 TERT 전용이다 ===
# BRAF 추출은 Thyroid_Mutation_Prediction_Model/src/data/virchow2/run_mixed_gpu.sh 가 담당한다.
# 두 추출이 한 서버에서 동시에 돌 수 있으므로 아래 항목을 서로 다르게 고정했다:
#   GPU        TERT=6          / BRAF=0~4
#   DDP port   TERT=29507      / BRAF=29500번대 (run_mixed_gpu.sh)
#   출력       .../tert/...    / .../braf/...
# TERT는 클래스별 디렉터리(Wild/C228T/C250T) 구조라 클래스마다 한 번씩 돌린다.
#
# 사전 준비: https://huggingface.co/paige-ai/Virchow2 라이선스 동의 + HF 로그인
set -e

# torch 의 GPU 열거 순서를 nvidia-smi 와 동일(PCI 버스 순서)하게 고정한다.
# 기본값 FASTEST_FIRST 는 번호를 재정렬해 혼동을 일으킨다.
# 혼종 GPU 서버 nvidia-smi 기준: 0,1=3080  2,3,4=V100S  5,6=3080
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU 선택: 6번 한 장만 전용으로 쓴다.
#   혼종 GPU 서버의 GPU 7장(0~6) 배분:
#     0~4  BRAF virchow2 추출 (Thyroid_Mutation_Prediction_Model)
#     5    MIL 학습 전용 (train.sh / run_bag.sh)
#     6    TERT virchow2 추출 ← 이 스크립트
#   BRAF 추출이 끝난 뒤 더 쓰려면 개수를 맞춰 넘긴다:
#     GPUS=2,3,4,6 NPROC=4 bash run.sh
GPUS="${GPUS:-6}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NPROC="${NPROC:-1}"

# GPUS 항목 수와 NPROC 이 다르면 DDP 가 엉뚱하게 뜬다. 여기서 막는다.
_ngpu=$(echo "${GPUS}" | tr ',' '\n' | grep -c .)
if [ "${_ngpu}" != "${NPROC}" ]; then
    echo "[X] GPUS(${GPUS} → ${_ngpu}개) 와 NPROC(${NPROC}) 이 다릅니다." >&2
    echo "    GPUS=5,6 NPROC=2  처럼 개수를 맞추세요." >&2
    exit 1
fi
unset _ngpu

# DDP rendezvous 포트. BRAF 추출(29500번대)과 겹치면 두 작업이 서로를 붙잡는다.
# TERT 는 29507 을 쓴다. 이미 점유 중이면 여기서 막는다.
MASTER_PORT="${MASTER_PORT:-29507}"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${MASTER_PORT}\b"; then
    echo "[X] 포트 ${MASTER_PORT} 가 이미 사용 중입니다 (다른 추출이 도는 중?)." >&2
    echo "    MASTER_PORT=29517 bash run.sh  처럼 다른 포트를 주세요." >&2
    exit 1
fi
export HF_HOME=/path/to/embeddings/.hf_cache

# --- HF 토큰 승격 -------------------------------------------------------------
# 주의: 위에서 HF_HOME 을 공유 스토리지로 바꾸므로, `huggingface-cli login` 이
# 홈($HOME/.cache/huggingface/token)에 저장한 토큰을 huggingface_hub 가 더 이상
# 찾지 못한다. 그래서 홈의 토큰 파일을 읽어 HF_TOKEN 환경변수로 올려준다.
# (이미 HF_TOKEN 이 설정돼 있으면 그것을 그대로 쓴다.)
if [ -z "${HF_TOKEN:-}" ]; then
    for _tf in "${HOME}/.cache/huggingface/token" "${HF_HOME}/token" "${HOME}/.huggingface/token"; do
        if [ -s "${_tf}" ]; then
            HF_TOKEN=$(tr -d '[:space:]' < "${_tf}")
            export HF_TOKEN
            echo "  [i] HF 토큰 로드: ${_tf}"
            break
        fi
    done
    unset _tf
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
# -----------------------------------------------------------------------------

# PYTHON: 서버마다 다르므로 자동 탐지한다. 환경변수로 덮어쓸 수 있다.
#   예)  PYTHON=/path/to/python bash run.sh
# 후보를 순서대로 보고 openslide + torch 가 있는 첫 python 을 고른다.
if [ -z "${PYTHON:-}" ]; then
    for _p in \
        "$(command -v python 2>/dev/null)" \
        "$(command -v python3 2>/dev/null)" \
        ${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"} ; do
        [ -x "${_p:-}" ] || continue
        if "$_p" -c "import torch, openslide" >/dev/null 2>&1; then
            PYTHON="$_p"; break
        fi
    done
    unset _p
fi
if [ -z "${PYTHON:-}" ] || ! [ -x "$PYTHON" ]; then
    echo "[X] torch+openslide 가 있는 python 을 찾지 못했습니다."
    echo "    PYTHON=/path/to/python bash run.sh  형태로 직접 지정하세요."
    exit 1
fi
echo "  [i] PYTHON = $PYTHON"
MODEL=paige-ai/Virchow2
EMBED_MODE=concat        # concat(2560) | cls(1280)
DTYPE=fp16

TERT_ROOT=/path/to/Thyroid_TERT_dataset
OUT_ROOT=/path/to/embeddings/tert/virchow2_40x512_${EMBED_MODE}

SOURCE=svs
SVS_DIRS="${TERT_ROOT}/thyroid"      # 201개 svs 가 평평하게 들어있다

# BATCH: 실측 Virchow2 fp16 = 1.18GB + 9.64MB/타일.
#   GPU6(3080, 10GB) 를 혼자 쓰므로 512 -> 피크 약 6.1GB 로 여유가 있다.
#   (BRAF 추출에서 384 -> 4.8GB 로 검증된 값에서 선형 환산)
#   OOM 이 나면 BATCH=384 로 낮춘다.
BATCH="${BATCH:-512}"
WORKERS="${WORKERS:-16}"

# fd 한도를 올린다. rank x worker 가 슬라이드마다 Queue/Lock 을 만들어
# 기본값(1024)으로는 금방 바닥난다 (BRAF 추출에서 27장째 Errno 24 발생 이력).
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# 단편화 완화 (BRAF 추출에서 V100S OOM 후 추가)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run_$(date +%m%d_%H%M).log"
echo "[INFO] model : ${MODEL} (${EMBED_MODE})"
echo "[INFO] out   : ${OUT_ROOT}"
echo "[INFO] log   : ${LOG}"

# 기존 TERT 임베딩과 동일한 {class}/npy/ 구조로 저장한다
# (extract_features.py 가 out_dir 아래 npy/ 와 json/ 을 만든다)
FAILED=""
for CLS in Wild C228T C250T; do
    echo "========== TERT ${CLS} ==========" | tee -a "${LOG}"
    # 주의: tee 로 파이프하면 기본적으로 tee 의 종료코드가 반환되어
    # python 이 죽어도 성공으로 보인다. pipefail 로 실제 실패를 전파한다.
    set +e
    set -o pipefail
    "${PYTHON}" -m torch.distributed.run \
        --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
        extract_features.py \
        --model_name ${MODEL} \
        --embed_mode ${EMBED_MODE} \
        --dtype ${DTYPE} \
        --tile_dir "${TERT_ROOT}/40x_patch/${CLS}" \
        --out_dir "${OUT_ROOT}/${CLS}" \
        --batch_size ${BATCH} \
        --num_workers ${WORKERS} \
        --source ${SOURCE} \
        --svs_dirs ${SVS_DIRS} 2>&1 | tee -a "${LOG}"
    rc=$?
    set +o pipefail
    set -e
    if [ $rc -ne 0 ]; then
        echo "[X] ${CLS} 실패 (rc=$rc) — 나머지 클래스는 계속 진행한다." | tee -a "${LOG}"
        echo "    다시 돌리면 이미 끝난 슬라이드는 건너뛴다(resume)." | tee -a "${LOG}"
        FAILED="${FAILED} ${CLS}"
    fi
done

# 결과 집계: 클래스별 기대 개수와 실제 npy 개수를 비교한다
echo | tee -a "${LOG}"
echo "========== 결과 ==========" | tee -a "${LOG}"
TOTAL=0
for CLS in Wild C228T C250T; do
    exp=$(find "${TERT_ROOT}/40x_patch/${CLS}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    got=$(find "${OUT_ROOT}/${CLS}/npy" -name '*.npy' 2>/dev/null | wc -l)
    TOTAL=$((TOTAL + got))
    mark="[O]"; [ "$got" != "$exp" ] && mark="[!]"
    printf "  %s %-6s %s / %s\n" "$mark" "${CLS}" "${got}" "${exp}" | tee -a "${LOG}"
done
echo "  합계: ${TOTAL} / 201" | tee -a "${LOG}"
if [ -n "${FAILED}" ]; then
    echo "[X] 실패한 클래스:${FAILED} — 같은 명령을 다시 실행하면 이어서 한다." | tee -a "${LOG}"
fi

echo "========== Done ==========" | tee -a "${LOG}"
echo
echo "다음 단계: CV split 생성 (2class + 3class 한 번에)"
echo "  cd /path/to/Thyroid_TERT_Mutation_Model"
echo "  bash scripts/make_virchow2_splits.sh"
