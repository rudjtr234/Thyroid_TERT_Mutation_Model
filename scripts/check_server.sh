#!/bin/bash
# ============================================================
# 이 서버에서 학습을 돌릴 수 있는지 사전 점검.
# 새 서버에서 학습을 처음 돌리기 전에 반드시 한 번 실행할 것.
#
#   bash scripts/check_server.sh
#   GPU_ID=5 bash scripts/check_server.sh    특정 GPU 지정
# ============================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FAIL=0
ok ()   { echo "  [O] $*"; }
bad ()  { echo "  [X] $*"; FAIL=1; }
warn () { echo "  [!] $*"; }

echo "=============================================================="
echo " 학습 환경 점검   host=$(hostname -s)   $(date '+%Y-%m-%d %H:%M')"
echo " repo: $REPO_ROOT"
echo "=============================================================="

# --- 1. python ---
echo; echo "[1] Python 인터프리터"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
if source "$SCRIPT_DIR/common_env.sh"; then
    ok "PYTHON = $PYTHON"
else
    bad "torch 가 있는 python 을 못 찾음"
    echo
    echo "  이 서버의 conda env 목록:"
    for base in ${TE_CONDA_ENV_DIRS:-} \
                "$HOME/anaconda3/envs" "$HOME/miniconda3/envs"; do
        [ -d "$base" ] && ls -1 "$base" 2>/dev/null | sed "s|^|    $base/|"
    done
    echo
    echo "  해결: 위 목록에서 torch 가 든 env 를 골라"
    echo "        PYTHON=<경로>/bin/python bash scripts/check_server.sh"
    echo "        맞는 env 가 없으면 새로 만들어야 한다:"
    echo "        conda create -n thyroid_mil python=3.10"
    echo "        pip install torch --index-url https://download.pytorch.org/whl/cu128"
    echo "        pip install scikit-learn pandas numpy mlflow nystrom-attention matplotlib seaborn"
    exit 1
fi

# --- 2. 필수 패키지 ---
echo; echo "[2] 필수 패키지"
"$PYTHON" - <<'PY'
import importlib, sys
req = ["torch","numpy","pandas","sklearn","matplotlib"]
opt = {"mlflow":"MLflow 로깅", "nystrom_attention":"TransMIL",
       "seaborn":"플롯", "transformers":"일부 모델"}
bad = 0
for m in req:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  [O] {m:<20} {v}")
    except Exception as e:
        print(f"  [X] {m:<20} 없음 ({e.__class__.__name__})"); bad = 1
for m, why in opt.items():
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  [O] {m:<20} {v}")
    except Exception:
        print(f"  [!] {m:<20} 없음 — {why} 사용 시 필요")
sys.exit(bad)
PY
[ $? -eq 0 ] || FAIL=1

# --- 3. GPU ---
echo; echo "[3] GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
if te_check_cuda; then ok "CUDA OK"; else bad "CUDA 사용 불가"; fi
te_warn_busy_gpu

echo; echo "[3b] 이 서버의 전체 GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/      /'

# --- 4. 데이터 경로 ---
echo; echo "[4] 임베딩 / split 경로"
# 확인할 데이터 마운트 지점 (공백 구분). 예: TE_DATA_MOUNTS="/mnt/wsi /mnt/embeddings"
for d in ${TE_DATA_MOUNTS:-}; do
    [ -d "$d" ] && ok "마운트 $d" || bad "마운트 $d 없음 (학습 데이터 접근 불가)"
done

# train.sh 가 현재 고른 split 파일을 확인한다
SPLIT=$(grep -m1 '^CV_SPLIT_FILE="' "$REPO_ROOT/src/training/train.sh" 2>/dev/null \
        | sed 's/^CV_SPLIT_FILE="//; s/"$//')
SPLIT="${SPLIT/\$PROJECT_ROOT/$REPO_ROOT}"   # train.sh 는 $PROJECT_ROOT 기준 경로를 쓴다
if [ -n "$SPLIT" ]; then
    if [ -f "$SPLIT" ]; then
        ok "split  $(basename "$SPLIT")"
        "$PYTHON" - "$SPLIT" <<'PY'
import json, os, sys
f = sys.argv[1]
j = json.load(open(f))
print(f"      embed_dim = {j.get('embed_dim', '(없음 -> 1536 기본값)')}")
print(f"      encoder   = {j.get('encoder', '?')}")
print(f"      WSI       = {j.get('total_wsis', j.get('total_samples', '?'))}")

# 임베딩 루트와 실제 npy 파일 존재 확인.
# 디렉토리만 있고 npy 가 없으면 학습이 즉시 죽으므로 개수까지 센다.
rc = 0
roots = [j[k] for k in ("embedding_root", "data_root", "embedding_dir", "npy_dir")
         if j.get(k)]
if not roots:
    print("      [!] split JSON 에 임베딩 경로 키가 없다 — 수동 확인 필요")
for r in roots:
    if not os.path.isdir(r):
        print(f"      [X] {r}  (없음)"); rc = 1; continue
    n = 0
    for dirpath, _dirnames, files in os.walk(r):
        n += sum(1 for f in files if f.endswith(".npy"))
        if n > 20000: break
    if n == 0:
        print(f"      [X] {r}  (npy 0개)"); rc = 1
    else:
        print(f"      [O] {r}  (npy {n}개)")
sys.exit(rc)
PY
        [ $? -eq 0 ] || bad "임베딩 경로 문제 — 위 [X] 참고"
    else
        bad "split 파일 없음: $SPLIT"
    fi
fi

# --- 5. 쓰기 권한 ---
echo; echo "[5] 출력 디렉토리 쓰기 권한"
OUTD="$REPO_ROOT/outputs"
mkdir -p "$OUTD" 2>/dev/null
T="$OUTD/.write_test_$$"
if touch "$T" 2>/dev/null; then
    rm -f "$T"; ok "쓰기 가능  $OUTD"
else
    bad "쓰기 불가  $OUTD   (소유자: $(stat -c '%U:%G' "$OUTD" 2>/dev/null), 현재: $(whoami))"
    echo "      해결: sudo chown -R \$(whoami) $OUTD"
fi

echo; echo "=============================================================="
if [ "$FAIL" = 0 ]; then
    echo " 결과: 학습 가능"
    echo "   단일 :  cd src/training && GPU_ID=${GPU_ID:-5} bash train.sh"
    echo "   스윕 :  cd scripts && bash sweep.sh --gpus 5,6 \\"
    echo "             --bags 200,400,800,1600 --model abmil --prefix <이름>"
else
    echo " 결과: 문제 있음 — 위 [X] 항목을 해결해야 한다"
fi
echo "=============================================================="
exit $FAIL
