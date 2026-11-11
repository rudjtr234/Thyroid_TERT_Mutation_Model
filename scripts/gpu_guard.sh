#!/bin/bash
# GPU 전력/헬스 가드 — Xid 79 장애 재발 방지용 공통 함수.
#
# 혼종 GPU 서버 한 대에서 여러 장이 동시에 "GPU has fallen off the bus"(Xid 79)로
# 이탈한 적이 있다. 개별 GPU 고장이 아니라 공통 전원(PSU/power rail) 또는 PCIe fabric
# 문제로 추정되므로, 그 서버에서는 전력을 정격보다 낮춰 동시 피크를 줄인다.
#
# 사용법 (각 run 스크립트 상단에서):
#   source /path/to/scripts/gpu_guard.sh
#   gpu_guard_require <GPU_UUID_또는_인덱스>     # 존재 확인 + 전력 캡 + 상태 출력
#   gpu_guard_watch_xid &                        # (선택) 백그라운드 Xid 감시
#
# 전력 캡은 nvidia-smi -pl 이 root 권한을 요구한다. 권한이 없으면 경고만 남기고
# 진행한다 — 임베딩이 권한 때문에 아예 못 도는 것보다는 낫다.

# 모델별 캡 (W). 정격 대비 ~70~80%로 낮춰 잡는다.
#   V100S PCIe 32GB   : 정격 250W → 200W
#   RTX 3080          : 정격 320W → 250W
#   RTX 6000 Ada 48GB : 정격 300W → 250W
#
# 장애 이력이 없는 서버에서는 캡이 예방 차원이며, 적용 실패해도 정격으로 진행한다.
GPU_GUARD_PL_V100=${GPU_GUARD_PL_V100:-200}
GPU_GUARD_PL_RTX3080=${GPU_GUARD_PL_RTX3080:-250}
GPU_GUARD_PL_RTX6000ADA=${GPU_GUARD_PL_RTX6000ADA:-250}

# nvidia-smi 한 줄 조회. $1 = GPU 지정자(UUID 또는 인덱스), $2 = 조회 필드
_gpu_guard_query() {
    nvidia-smi --id="$1" --query-gpu="$2" --format=csv,noheader,nounits 2>/dev/null
}

# GPU가 실제로 붙어 있는지 확인하고, 모델에 맞는 전력 캡을 건다.
# 실패하면(=GPU 미검출) 1을 리턴하므로 호출부에서 exit 시킬 것.
gpu_guard_require() {
    local gpu="$1"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[FATAL] nvidia-smi 를 찾을 수 없다. GPU 서버에서 실행하고 있는지 확인할 것."
        return 1
    fi

    local name
    name="$(_gpu_guard_query "$gpu" name)"
    if [ -z "$name" ]; then
        echo "[FATAL] GPU '$gpu' 를 찾을 수 없다."
        echo "        GPU 가 버스에서 이탈했을 수 있다. 아래로 현재 UUID를 확인하고"
        echo "        스크립트의 CUDA_VISIBLE_DEVICES 값을 교체할 것:"
        echo "          nvidia-smi --query-gpu=index,name,uuid --format=csv"
        return 1
    fi

    # 모델에 따라 캡 결정
    local cap=""
    case "$name" in
        *V100S*)      cap=$GPU_GUARD_PL_V100 ;;
        *"3080"*)     cap=$GPU_GUARD_PL_RTX3080 ;;
        *"6000 Ada"*) cap=$GPU_GUARD_PL_RTX6000ADA ;;
        *)            echo "[WARN] '$name' 은 캡 기준이 정의되지 않았다 — 전력 캡을 건너뛴다." ;;
    esac

    if [ -n "$cap" ]; then
        # 캡이 하드웨어 허용 범위 안인지 확인 (범위 밖이면 nvidia-smi가 실패한다)
        local min max
        min="$(_gpu_guard_query "$gpu" power.min_limit)"
        max="$(_gpu_guard_query "$gpu" power.max_limit)"
        min="${min%%.*}"; max="${max%%.*}"
        if [ -n "$min" ] && [ -n "$max" ] && { [ "$cap" -lt "$min" ] || [ "$cap" -gt "$max" ]; }; then
            echo "[WARN] 요청한 캡 ${cap}W 가 허용 범위(${min}~${max}W) 밖이다 — 캡을 건너뛴다."
        elif nvidia-smi --id="$gpu" -pl "$cap" >/dev/null 2>&1; then
            echo "[OK] $name : 전력 캡 ${cap}W 적용"
        elif sudo -n nvidia-smi --id="$gpu" -pl "$cap" >/dev/null 2>&1; then
            echo "[OK] $name : 전력 캡 ${cap}W 적용 (sudo)"
        else
            echo "[WARN] $name : 전력 캡 적용 실패 (root 권한 필요). 정격으로 진행한다."
            echo "       수동 적용: sudo nvidia-smi --id=$gpu -pl $cap"
        fi
    fi

    # 시작 시점 상태를 로그에 남겨둔다 — 다시 이탈하면 이 값이 비교 기준이 된다.
    echo "[INFO] $(_gpu_guard_query "$gpu" 'name,uuid,temperature.gpu,power.draw,power.limit,memory.used,memory.total')"
    return 0
}

# 백그라운드 감시: GPU가 조회 불가 상태가 되면(=bus에서 이탈) 즉시 알리고
# 부모 프로세스 그룹을 종료한다. NCCL watchdog 600초를 기다리지 않기 위함.
# $1 = 감시할 GPU 지정자, $2 = 감시 주기(초, 기본 60)
gpu_guard_watch_xid() {
    local gpu="$1"
    local interval="${2:-60}"
    local parent=$$

    while kill -0 "$parent" 2>/dev/null; do
        if [ -z "$(_gpu_guard_query "$gpu" name)" ]; then
            echo ""
            echo "[FATAL] $(date '+%F %T') GPU '$gpu' 가 응답하지 않는다 — Xid 79 재발 가능성."
            echo "        과거 장애와 같은 패턴이다. 작업을 중단한다."
            echo "        확인: dmesg -T | grep -i -E 'xid|fallen off'"
            kill -TERM "$parent" 2>/dev/null
            return 1
        fi
        sleep "$interval"
    done
}
