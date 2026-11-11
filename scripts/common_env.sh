#!/bin/bash
# ============================================================
# 서버 공통 환경 설정 — 학습 스크립트에서 source 해서 쓴다.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/common_env.sh"
#
# 해결하는 문제
#   1) PYTHON 경로가 서버마다 다르다.
#      서버별로 홈/conda 경로가 달라 한쪽 경로는 다른 쪽에 아예 없다.
#      후보를 순회하며 torch+sklearn+numpy+pandas 가 import 되는 python 을 고른다.
#   2) nvidia-smi 와 PyTorch 의 GPU 번호가 다르다.
#      nvidia-smi 는 PCI_BUS_ID 순, PyTorch 는 기본 FASTEST_FIRST 순이라
#      서로 다른 GPU 가 섞인 서버(예: RTX 3080 x4 + V100S x3)에서는 번호가 뒤섞인다.
#      CUDA_DEVICE_ORDER=PCI_BUS_ID 로 고정해 nvidia-smi 번호와 일치시킨다.
#   3) 같은 GPU 에 이미 다른 작업이 돌고 있으면 경고한다.
#   4) (선택) 특정 서버에서만 "학습용 GPU 제한"과 "전력 캡"을 건다.
#
# 환경변수 (전부 선택)
#   PYTHON                   인터프리터 직접 지정
#   TE_PYTHON_CANDIDATES     추가 후보 경로 (콜론 구분)
#   TE_GUARD_HOST_PATTERN    가드를 켤 호스트명 glob (공백 구분, 예: "*gpu01* *mixed*")
#                            비워 두면(기본) 어떤 서버에서도 가드를 켜지 않는다
#   TE_TRAINING_GPUS         가드가 켜진 서버에서 학습에 허용할 GPU 번호 (기본 "5")
#   TE_ALLOW_BUSY_GPU=1      가드 우회
# ============================================================

# --- (2) GPU 번호를 nvidia-smi 와 일치시킨다 (CUDA_VISIBLE_DEVICES 보다 먼저) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- 서버 식별 ---
# 외부에서 TE_HOST 를 주면 그것을 쓴다 (가드 동작을 다른 서버에서 검증할 때 필요).
TE_HOST="${TE_HOST:-$(hostname -s 2>/dev/null || echo unknown)}"
TE_GUARD_HOST_PATTERN="${TE_GUARD_HOST_PATTERN:-}"

# 현재 호스트가 가드 대상인지 판정한다.
te_is_guarded_host () {
    [ -n "$TE_GUARD_HOST_PATTERN" ] || return 1
    local pats pat
    read -r -a pats <<< "$TE_GUARD_HOST_PATTERN"
    for pat in "${pats[@]}"; do
        # shellcheck disable=SC2254
        case "$TE_HOST" in $pat) return 0 ;; esac
    done
    return 1
}

# --- (1) PYTHON 자동 탐지 -------------------------------------------------
# 환경변수로 덮어쓸 수 있다:  PYTHON=/path/to/python bash train.sh
# 필요 패키지: torch(+CUDA), sklearn, numpy, pandas
# nystrom_attention 은 TransMIL 전용이라 여기서는 필수로 보지 않고 경고만 한다.
te_find_python () {
    local p cands=()
    [ -n "${PYTHON:-}" ] && cands+=("$PYTHON")
    if [ -n "${TE_PYTHON_CANDIDATES:-}" ]; then
        local extra
        IFS=':' read -r -a extra <<< "$TE_PYTHON_CANDIDATES"
        cands+=("${extra[@]}")
    fi
    [ -n "${CONDA_PREFIX:-}" ] && cands+=("$CONDA_PREFIX/bin/python")
    cands+=("$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)")

    for p in "${cands[@]}"; do
        [ -n "$p" ] && [ -x "$p" ] || continue
        if "$p" -c "import torch, sklearn, numpy, pandas" >/dev/null 2>&1; then
            echo "$p"; return 0
        fi
    done
    return 1
}

if ! PYTHON="$(te_find_python)"; then
    echo "[X] torch+sklearn+numpy+pandas 가 있는 python 을 찾지 못했습니다. (host=$TE_HOST)" >&2
    echo "    PYTHON=/path/to/python bash <script>  형태로 직접 지정하세요." >&2
    return 1 2>/dev/null || exit 1
fi
export PYTHON

# --- CUDA 사용 가능 여부 + 선택한 GPU 확인 --------------------------------
# CUDA_VISIBLE_DEVICES 는 이 파일을 source 하기 *전에* 설정돼 있어야 한다.
te_check_cuda () {
    "$PYTHON" - <<'PYCHK'
import os, sys
try:
    import torch
except Exception as e:
    print(f"[X] torch import 실패: {e}", file=sys.stderr); sys.exit(1)
if not torch.cuda.is_available():
    print(f"[X] CUDA 사용 불가 (torch {torch.__version__}). "
          "드라이버와 torch 빌드가 맞는지 확인하세요.", file=sys.stderr)
    sys.exit(1)
n = torch.cuda.device_count()
if n == 0:
    print("[X] 보이는 GPU 가 0개입니다. CUDA_VISIBLE_DEVICES 를 확인하세요.",
          file=sys.stderr)
    sys.exit(1)
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
print(f"[CHECK] torch {torch.__version__} / CUDA_VISIBLE_DEVICES={cvd} / {n} GPU")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"        cuda:{i} = {p.name}  sm_{p.major}{p.minor}  "
          f"{total/1024**3:.1f}GB (free {free/1024**3:.1f}GB)")
    # Volta(sm_70)는 FlashAttention 불가. MIL 모델은 보통 SDPA 를 쓰지 않지만
    # TransMIL 등이 쓸 수 있어 알려만 둔다.
    if p.major < 8:
        print(f"        [i] sm_{p.major}{p.minor} < sm_80 → FlashAttention 미지원 "
              "(TransMIL 사용 시 mem_efficient/math SDPA 로 폴백됨)")
PYCHK
}

# --- (3) 선택한 GPU 에 다른 프로세스가 있으면 경고 -------------------------
te_warn_busy_gpu () {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local cvd="${CUDA_VISIBLE_DEVICES:-}"
    [ -n "$cvd" ] || return 0
    local idx used
    for idx in ${cvd//,/ }; do
        used=$(nvidia-smi -i "$idx" --query-gpu=memory.used \
               --format=csv,noheader,nounits 2>/dev/null)
        [ -n "$used" ] || continue
        if [ "$used" -gt 1000 ]; then
            echo "[!] GPU $idx 에 이미 ${used}MiB 사용 중인 프로세스가 있습니다."
            nvidia-smi -i "$idx" --query-compute-apps=pid,used_memory,process_name \
                --format=csv,noheader 2>/dev/null | sed 's/^/      /'
        fi
    done
}

# --- (4a) 가드 대상 서버: 추출 전용 GPU 사용 차단 ---------------------------
# 임베딩 추출과 MIL 학습을 한 서버에서 같이 돌릴 때의 GPU 배분 예 (7장: 0~6):
#   0~4  BRAF virchow2 추출   (DDP 5장)
#   5    MIL 학습 전용        ← 이 스크립트들
#   6    TERT virchow2 추출   (src/data/virchow2/run.sh)
#
# 학습이 추출 GPU 를 고르면
#   - 추출과 연산 자원을 경쟁해 추출이 느려지고
#   - 10GB 급 GPU 는 VRAM 여유가 몇 GB 뿐이라 피크에서 OOM 으로 죽는다
# 그래서 여기서 막는다.
#
# 추출이 끝났거나 일부러 써야 하면 아래로 우회한다:
#   TE_ALLOW_BUSY_GPU=1 GPU_ID=6 bash <script>
TE_TRAINING_GPUS="${TE_TRAINING_GPUS:-5}"

te_guard_gpu_choice () {
    te_is_guarded_host || return 0     # 가드 대상이 아니면 제한하지 않는다
    [ "${TE_ALLOW_BUSY_GPU:-0}" = "1" ] && return 0

    local idx bad=""
    for idx in ${CUDA_VISIBLE_DEVICES//,/ }; do
        case " $TE_TRAINING_GPUS " in
            *" $idx "*) ;;
            *) bad="$bad $idx" ;;
        esac
    done
    [ -n "$bad" ] || return 0

    echo "[X] GPU${bad} 는 임베딩 추출이 쓰는 GPU 다." >&2
    echo "    학습은 ${TE_TRAINING_GPUS// /, } 번만 쓴다." >&2
    echo >&2
    echo "    GPU_ID=${TE_TRAINING_GPUS%% *} bash <script>" >&2
    echo >&2
    echo "    추출이 끝났거나 일부러 써야 하면:" >&2
    echo "    TE_ALLOW_BUSY_GPU=1 GPU_ID=${bad# } bash <script>" >&2
    return 1
}

# --- (4b) 가드 대상 서버: 전력 가드 ------------------------------------------
# 혼종 GPU 서버 한 대에서 Xid 79 ("GPU has fallen off the bus") 로 GPU 4장이
# 동시에 버스에서 이탈한 적이 있다. 개별 GPU 고장이 아니라 공통 전원/PCIe fabric
# 문제로 추정되므로, 그 서버에서는 전력을 정격보다 낮춰 동시 피크를 줄인다.
# (V100S 250->200W / RTX3080 320->250W)
#
# 추출이 이미 여러 장을 100% 로 돌리는 상태에서 남은 GPU 까지 학습에 투입하면
# 전 장이 동시에 피크를 친다. 그래서 학습 스크립트에서도 가드를 걸어 둔다.
#
# 전력 캡은 root 권한이 필요하다. 권한이 없으면 경고만 남기고 진행한다
# (학습이 권한 때문에 아예 못 도는 것보다는 낫다).
te_power_guard () {
    local guard
    guard="$(dirname "${BASH_SOURCE[0]}")/gpu_guard.sh"
    [ -f "$guard" ] || return 0
    te_is_guarded_host || return 0     # 가드 대상이 아니면 아무것도 하지 않는다
    # shellcheck disable=SC1090
    source "$guard" || return 0
    local idx
    for idx in ${CUDA_VISIBLE_DEVICES//,/ }; do
        gpu_guard_require "$idx" || true
    done
}

# --- TransMIL 전용 의존성 안내 --------------------------------------------
te_check_transmil () {
    "$PYTHON" -c "import nystrom_attention" >/dev/null 2>&1 && return 0
    echo "[X] TransMIL 을 골랐는데 nystrom_attention 이 없습니다." >&2
    echo "    $PYTHON -m pip install nystrom-attention" >&2
    return 1
}

echo "  [i] host   = $TE_HOST"
echo "  [i] PYTHON = $PYTHON"
