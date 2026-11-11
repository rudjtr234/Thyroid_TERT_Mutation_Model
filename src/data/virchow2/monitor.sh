#!/bin/bash
# Virchow2 TERT 추출 진행 모니터
# 사용법:
#   bash monitor.sh            # 1회 출력
#   bash monitor.sh -w         # 30초마다 갱신 (watch 모드)
#   bash monitor.sh -w 10      # 10초 간격
#
# === TERT 전용 ===
# BRAF 추출 모니터는 Thyroid_Mutation_Prediction_Model/src/data/virchow2/monitor.sh 를 쓴다.
set -u

TARGET=tert
WATCH=0
INTERVAL=30
for a in "$@"; do
    case "$a" in
        -w|--watch) WATCH=1 ;;
        braf)
            echo "[X] 이 monitor.sh 는 TERT 전용입니다." >&2
            echo "    BRAF: cd /path/to/Thyroid_Mutation_Prediction_Model/src/data/virchow2 && bash monitor.sh" >&2
            exit 1 ;;
        tert)       TARGET=tert ;;
        [0-9]*)     INTERVAL="$a" ;;
    esac
done

EMBED_MODE=${EMBED_MODE:-concat}
EXP_DIM=$([ "$EMBED_MODE" = "cls" ] && echo 1280 || echo 2560)

# GROUPS/EXPECT는 같은 인덱스로 짝을 이룬다. set -u 와의 충돌을 피해 미리 선언한다.
# "그룹명:기대개수" 형식. 배열 2개를 인덱스로 짝짓는 방식은 local 과 충돌해 버그를 냈다.
ROOT=/path/to/embeddings/tert/virchow2_40x512_${EMBED_MODE}
PAIRS="Wild:113 C228T:69 C250T:19"

show () {
    echo "=========================================================="
    echo " Virchow2 / ${TARGET} / ${EMBED_MODE} (dim ${EXP_DIM})   $(date '+%m-%d %H:%M:%S')"
    echo " host: $(hostname -s 2>/dev/null || echo unknown)"
    echo " out: ${ROOT}"
    echo "=========================================================="

    if [ ! -d "$ROOT" ]; then
        echo " [대기] 출력 디렉터리가 아직 없습니다 (추출 미시작)"
        return
    fi

    local total=0 texp=0 pair g exp n pct filled bar k
    for pair in $PAIRS; do
        g="${pair%%:*}"
        exp="${pair##*:}"
        n=0
        if [ -d "$ROOT/$g/npy" ]; then
            n=$(find "$ROOT/$g/npy" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
        fi
        total=$((total + n)); texp=$((texp + exp))
        pct=0; [ "$exp" -gt 0 ] && pct=$((n * 100 / exp))
        filled=$((pct / 5)); bar=""
        for ((k = 0; k < 20; k++)); do
            if [ "$k" -lt "$filled" ]; then bar="${bar}#"; else bar="${bar}."; fi
        done
        printf "  %-10s [%s] %5d / %-5d (%3d%%)\n" "$g" "$bar" "$n" "$exp" "$pct"
    done
    echo "  ----------------------------------------------------------"
    local tpct=0; [ "$texp" -gt 0 ] && tpct=$((total*100/texp))
    printf "  %-10s %18d / %-5d (%3d%%)\n" "합계" "$total" "$texp" "$tpct"

    # 용량 / 속도 / ETA
    local sz; sz=$(du -sh "$ROOT" 2>/dev/null | cut -f1)
    echo "  용량: ${sz:-0}"

    # 최근 10분간 생성된 npy 로 속도 추정
    local recent; recent=$(find "$ROOT" -name '*.npy' -newermt '10 minutes ago' 2>/dev/null | wc -l)
    if [ "$recent" -gt 0 ]; then
        local per_h=$((recent*6))
        local left=$((texp-total))
        echo "  속도: 최근 10분 ${recent}장 (≈ ${per_h}장/h)"
        if [ "$per_h" -gt 0 ] && [ "$left" -gt 0 ]; then
            local eta_h=$((left/per_h))
            echo "  남은: ${left}장 → ETA 약 ${eta_h}h ($(date -d "+${eta_h} hours" '+%m-%d %H:%M' 2>/dev/null))"
        fi
    else
        echo "  속도: 최근 10분간 신규 npy 없음"
    fi

    # 프로세스 / GPU
    # 주의: 아래 두 섹션은 "이 셸이 떠 있는 호스트" 를 본다. npy/로그는 공유
    # 마운트라 어디서 봐도 정확하지만, 프로세스와 GPU 는 추출을 실행한 서버에서
    # 봐야 의미가 있다. 다른 호스트면 오해를 막기 위해 표시만 하고 넘긴다.
    local me; me=$(hostname -s 2>/dev/null || echo unknown)
    local is_runner=1
    # RUN_HOST_PATTERN(glob) 을 주면 그 호스트에서만 프로세스/GPU 를 본다.
    if [ -n "${RUN_HOST_PATTERN:-}" ]; then
        # shellcheck disable=SC2254
        case "$me" in $RUN_HOST_PATTERN) ;; *) is_runner=0 ;; esac
    fi

    local nproc; nproc=$(pgrep -f "extract_features.py" 2>/dev/null | wc -l | tr -d " ")
    nproc=${nproc:-0}
    if [ "$nproc" -gt 0 ]; then
        echo "  프로세스: extract_features.py ${nproc}개 실행 중 (${me})"
    elif [ "$total" -ge "$texp" ] && [ "$texp" -gt 0 ]; then
        echo "  프로세스: 없음 — 완료된 것으로 보입니다"
    elif [ "$is_runner" = "1" ]; then
        echo "  프로세스: 없음 — ⚠ 중단되었을 수 있습니다 (로그 확인)"
    else
        echo "  프로세스: 여기(${me})서는 알 수 없음 — 추출 서버에서 확인할 것"
        echo "            추출 서버에서: bash monitor.sh"
    fi
    if [ "$is_runner" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
        echo "  GPU (${me}):"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
            | awk -F', ' '{printf "    GPU%s  util %-5s mem %s\n",$1,$2,$3}'
    fi

    # 최신 로그 끝줄 (tqdm 캐리지리턴 정리)
    local lg; lg=$(ls -t "$ROOT"/run_*.log 2>/dev/null | head -1)
    if [ -n "${lg:-}" ]; then
        echo "  로그: $(basename "$lg")"
        tr '\r' '\n' < "$lg" | grep -vE '^\s*$' | tail -2 | sed 's/^/    /'
        local err; err=$(grep -icE "error|traceback|out of memory" "$lg" 2>/dev/null | head -1 | tr -d " ")
        err=${err:-0}
        [ "$err" -gt 0 ] && echo "    ⚠ 로그에 에러 ${err}건 — grep -iE 'error|traceback' 로 확인"
    fi
}

if [ "$WATCH" = "1" ]; then
    echo "(${INTERVAL}초 간격 갱신 / Ctrl-C 로 종료)"
    while true; do clear; show; sleep "$INTERVAL"; done
else
    show
fi
