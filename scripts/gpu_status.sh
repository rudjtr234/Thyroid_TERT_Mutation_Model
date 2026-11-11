#!/bin/bash
# ============================================================
# 서버 공용 GPU + 작업 현황 보기
#
#   bash gpu_status.sh        한 번 출력
#   bash gpu_status.sh -w     5초마다 갱신
# ============================================================
WATCH=0
[ "${1:-}" = "-w" ] && WATCH=1

show () {
    echo "=============================================================="
    echo " $(hostname -s)   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================="
    printf "%-4s %-28s %-14s %-6s %s\n" IDX NAME MEM UTIL TEMP
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx name used total util temp; do
        name=$(echo "$name" | sed 's/^ *//;s/NVIDIA //;s/GeForce //')
        printf "%-4s %-28s %5s/%-6s %4s%% %4s C\n" \
            "$idx" "$name" "${used// /}" "${total// /}" "${util// /}" "${temp// /}"
    done

    echo "-------------------------------------------------------------- "
    echo " GPU 를 쓰는 프로세스"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r uuid pid mem pname; do
        # uuid -> index 매핑
        idx=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader 2>/dev/null \
              | awk -F', *' -v u="${uuid// /}" '$2==u{print $1}')
        # 커맨드라인에서 뭘 돌리는지 요약
        cmd=$(tr '\0' ' ' < /proc/${pid// /}/cmdline 2>/dev/null)
        tag="-"
        case "$cmd" in
            *extract_features.py*) tag="virchow2 추출" ;;
            *main.py*)
                sd=$(echo "$cmd" | grep -oE '\-\-model_save_dir [^ ]+' | awk '{print $2}')
                # TERT 는 --model_type / BRAF(v2) 는 --model_name
                mt=$(echo "$cmd" | grep -oE '\-\-model_(type|name) [^ ]+' | awk '{print $2}')
                bs=$(echo "$cmd" | grep -oE '\-\-bag_size [^ ]+'        | awk '{print $2}')
                tag="학습 ${mt:-?} bag=${bs:-?} $(basename "${sd:-?}")" ;;
            *) tag=$(basename "$(echo "$cmd" | awk '{print $1}')") ;;
        esac
        printf "  GPU %-3s pid %-8s %-10s %s\n" "${idx:-?}" "${pid// /}" "${mem// /}" "$tag"
    done
    echo "=============================================================="
}

if [ "$WATCH" = 1 ]; then
    while true; do clear; show; sleep 5; done
else
    show
fi
