#!/bin/bash
# ============================================================
# bag_size (또는 임의 설정) 스윕을 여러 GPU 에 나눠 병렬 실행한다.
#
# 왜 DDP 가 아니라 이 방식인가
#   MIL 학습은 이미 뽑아 둔 임베딩 벡터만 쓰므로 모델이 작고
#   GPU 1장에서 VRAM 몇 GB 면 끝난다. 한 번의 학습을 DDP 로 2장에
#   쪼개도 통신 비용 때문에 거의 빨라지지 않는다.
#   반면 bag_size 스윕은 서로 독립적인 run 이 여러 개이므로
#   GPU 마다 다른 run 을 붙이면 장수만큼 선형으로 빨라진다.
#
# 사용법
#   bash sweep.sh --gpus 5,6 --bags 200,400,800,1600 \
#                 --model abmil --prefix thyroid_tert_v9.9
#
#   --gpus    쉼표 구분 GPU 번호 (nvidia-smi 번호와 동일)
#   --bags    쉼표 구분 bag_size 목록
#   --model   MODEL_TYPE (abmil|transmil|acmil|dtfd|mhim|clam)
#   --prefix  출력 디렉토리 접두사. 실제 경로는 <prefix>.<i> 로 붙는다
#             예) thyroid_tert_v9.9 + bags 200,400 -> ...v9.9.0, ...v9.9.1
#   --split   CV_SPLIT_FILE (생략 시 train.sh 의 기본값 사용)
#   --classes NUM_CLASSES (생략 시 train.sh 기본값)
#   --dry-run 실행하지 않고 계획만 출력
#
# 로그: <outputs>/sweep_logs/<prefix>.<i>_bag<N>_gpu<G>.log
# 중지: 각 GPU 작업은 독립 프로세스다. kill 하려면
#         pkill -f 'main.py --cv_split_file'
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SH="$REPO_ROOT/src/training/train.sh"

GPUS=""; BAGS=""; MODEL=""; PREFIX=""; SPLIT=""; CLASSES=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)    GPUS="$2"; shift 2 ;;
        --bags)    BAGS="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --prefix)  PREFIX="$2"; shift 2 ;;
        --split)   SPLIT="$2"; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        *) echo "[X] 모르는 옵션: $1" >&2; exit 1 ;;
    esac
done

[ -n "$GPUS" ]   || { echo "[X] --gpus 필요" >&2; exit 1; }
[ -n "$BAGS" ]   || { echo "[X] --bags 필요" >&2; exit 1; }
[ -n "$PREFIX" ] || { echo "[X] --prefix 필요" >&2; exit 1; }
[ -f "$TRAIN_SH" ] || { echo "[X] train.sh 없음: $TRAIN_SH" >&2; exit 1; }

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a BAG_ARR <<< "$BAGS"
NG=${#GPU_ARR[@]}
NB=${#BAG_ARR[@]}

# 출력 디렉토리 중복 검사 — 기존 run 을 덮어쓰면 결과가 섞인다
CONFLICT=0
for i in $(seq 0 $((NB-1))); do
    d="$REPO_ROOT/outputs/${PREFIX}.${i}"
    if [ -e "$d" ]; then
        echo "[X] 이미 존재: $d"
        CONFLICT=1
    fi
done
if [ "$CONFLICT" = 1 ]; then
    echo "[X] --prefix 를 바꾸거나 위 디렉토리를 먼저 치우세요. (덮어쓰기 방지)" >&2
    exit 1
fi

LOGDIR="$REPO_ROOT/outputs/sweep_logs"
mkdir -p "$LOGDIR"

echo "============================================================"
echo " sweep : ${NB} run 을 ${NG} GPU 에 분배"
echo " GPUs  : ${GPUS}"
echo " bags  : ${BAGS}"
echo " model : ${MODEL:-(train.sh 기본값)}"
echo " prefix: ${PREFIX}  ->  outputs/${PREFIX}.0 ...${PREFIX}.$((NB-1))"
echo " logs  : ${LOGDIR}"
echo "============================================================"

# GPU 당 작업 큐를 만들어 순차 실행한다 (라운드로빈).
# 한 GPU 에 여러 run 을 동시에 올리면 VRAM 이 터질 수 있어 직렬로 돈다.
for g in $(seq 0 $((NG-1))); do
    GPU="${GPU_ARR[$g]}"
    # 이 GPU 가 맡을 인덱스 목록
    MY_IDX=""
    for i in $(seq 0 $((NB-1))); do
        [ $((i % NG)) -eq "$g" ] && MY_IDX="$MY_IDX $i"
    done
    [ -n "$MY_IDX" ] || continue

    echo "  GPU ${GPU} <-${MY_IDX}"
    [ "$DRY" = 1 ] && continue

    (
        for i in $MY_IDX; do
            BAG="${BAG_ARR[$i]}"
            OUT="$REPO_ROOT/outputs/${PREFIX}.${i}"
            LOG="$LOGDIR/${PREFIX}.${i}_bag${BAG}_gpu${GPU}.log"
            echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} start  bag=${BAG}  -> ${PREFIX}.${i}"
            # train.sh 의 변수를 환경변수로 덮어쓴다.
            # (train.sh 안에서 BAG_SIZE=... 처럼 대입하므로 그대로는 안 먹는다.
            #  그래서 아래 TE_* 를 train.sh 가 읽도록 해 뒀다.)
            env GPU_ID="$GPU" \
                TE_BAG_SIZE="$BAG" \
                TE_MODEL_SAVE_DIR="$OUT" \
                ${MODEL:+TE_MODEL_TYPE="$MODEL"} \
                ${SPLIT:+TE_CV_SPLIT_FILE="$SPLIT"} \
                ${CLASSES:+TE_NUM_CLASSES="$CLASSES"} \
                bash "$TRAIN_SH" > "$LOG" 2>&1
            rc=$?
            if [ $rc -eq 0 ]; then
                echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} done   bag=${BAG}  (${PREFIX}.${i})"
            else
                echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} FAIL rc=$rc bag=${BAG}  로그: $LOG"
            fi
        done
    ) &
done

[ "$DRY" = 1 ] && { echo "(dry-run: 실행 안 함)"; exit 0; }

echo "------------------------------------------------------------"
echo "모든 GPU 작업 시작됨. 완료까지 대기 중..."
echo "진행 확인:  tail -f ${LOGDIR}/${PREFIX}.*.log"
echo "------------------------------------------------------------"
wait
echo "============================================================"
echo " sweep 완료"
for i in $(seq 0 $((NB-1))); do
    f="$REPO_ROOT/outputs/${PREFIX}.${i}/results_cv_summary_optimal.json"
    if [ -f "$f" ]; then
        printf "  %-28s bag=%-6s OK\n" "${PREFIX}.${i}" "${BAG_ARR[$i]}"
    else
        printf "  %-28s bag=%-6s 결과 없음(실패 의심)\n" "${PREFIX}.${i}" "${BAG_ARR[$i]}"
    fi
done
echo "============================================================"
