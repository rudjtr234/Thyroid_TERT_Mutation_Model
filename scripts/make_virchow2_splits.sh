#!/bin/bash
# ============================================================
# Virchow2 TERT CV split 생성
#
# 선행 조건: Virchow2 임베딩 추출이 끝나 있어야 한다.
#   cd src/data/virchow2 && bash run.sh
#
# 사용법:
#   bash scripts/make_virchow2_splits.sh        # 2class 만 (기본)
#   CLASSES="2 3" bash scripts/make_virchow2_splits.sh   # 둘 다
# ============================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Virchow2 = CLS(1280) ⊕ mean(patch tokens)(1280) = 2560
EMBED_DIM=2560
ENCODER=Virchow2
NEW_ROOT="${NEW_ROOT:-/path/to/embeddings/tert/virchow2_40x512_concat}"
# 만들 split 종류. 기본은 2class 만. 3class 도 필요하면 CLASSES="2 3" 으로 준다.
CLASSES="${CLASSES:-2}"

source "$SCRIPT_DIR/common_env.sh" || exit 1

echo "=============================================================="
echo " Virchow2 TERT CV split 생성"
echo "   embedding_root : $NEW_ROOT"
echo "   encoder        : $ENCODER  (embed_dim=$EMBED_DIM)"
echo "   생성 대상      : ${CLASSES}-class"
echo "=============================================================="

# --- 임베딩 존재 확인 -------------------------------------------------------
if [ ! -d "$NEW_ROOT" ]; then
    echo "[X] 임베딩 디렉토리가 없습니다: $NEW_ROOT" >&2
    echo "    먼저 추출을 끝내세요:" >&2
    echo "      cd src/data/virchow2 && bash run.sh" >&2
    exit 1
fi

TOTAL=0
for CLS in Wild C228T C250T; do
    n=$(find "$NEW_ROOT/$CLS/npy" -name '*.npy' 2>/dev/null | wc -l)
    TOTAL=$((TOTAL + n))
    printf "   %-6s %s\n" "$CLS" "$n"
done
echo "   합계   $TOTAL / 201"
if [ "$TOTAL" -ne 201 ]; then
    echo "[X] 201장이 아닙니다. 추출이 끝나지 않았습니다." >&2
    echo "    추출을 다시 돌리면 이어서 합니다 (resume)." >&2
    exit 1
fi

# --- 차원 확인 (split 에 적는 embed_dim 과 실제가 다르면 학습이 죽는다) ----
echo
echo "[1] 임베딩 차원 확인"
"$PYTHON" - "$NEW_ROOT" "$EMBED_DIM" <<'PY' || exit 1
import glob, sys
import numpy as np
root, want = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(f"{root}/*/npy/*.npy"))[:3]
if not files:
    print("[X] npy 파일을 찾지 못했습니다.", file=sys.stderr); sys.exit(1)
bad = 0
for f in files:
    a = np.load(f, mmap_mode="r")
    ok = a.ndim == 2 and a.shape[1] == want
    print(f"    {'[O]' if ok else '[X]'} {a.shape}  {f.split('/')[-1]}")
    if not ok: bad = 1
if bad:
    print(f"[X] 기대한 차원({want})과 다릅니다. EMBED_MODE 를 확인하세요 "
          "(concat=2560 / cls=1280).", file=sys.stderr)
sys.exit(bad)
PY

# --- split 생성 -------------------------------------------------------------
for NC in ${CLASSES}; do
    if [ "$NC" = 2 ]; then
        IN_JSON="config/cv_splits_tert_5fold_seed42.json"
    else
        IN_JSON="config/cv_splits_tert_5fold_seed42_hoptimus_3class.json"
    fi
    OUT_JSON="config/cv_splits_tert_5fold_seed42_virchow2_40x_${NC}class.json"

    echo
    echo "[2] ${NC}-class split 생성"
    if [ ! -f "$IN_JSON" ]; then
        echo "    [X] 원본 split 이 없습니다: $IN_JSON — 건너뜁니다." >&2
        continue
    fi
    if [ -f "$OUT_JSON" ]; then
        echo "    [!] 이미 존재: $OUT_JSON — 건너뜁니다."
        echo "        다시 만들려면 먼저 지우세요."
        continue
    fi

    # 원본 split 의 embedding_root 를 읽어 old_root 로 넘긴다
    OLD_ROOT=$("$PYTHON" -c \
        "import json,sys; print(json.load(open(sys.argv[1]))['embedding_root'])" \
        "$IN_JSON")
    echo "    old_root = $OLD_ROOT"

    "$PYTHON" src/data/create_cv_splits_hoptimus.py \
        --input_json  "$IN_JSON" \
        --output_json "$OUT_JSON" \
        --old_root    "$OLD_ROOT" \
        --new_root    "$NEW_ROOT" \
        --encoder     "$ENCODER" \
        --embed_dim   "$EMBED_DIM" \
        --verify || { echo "    [X] ${NC}-class 생성 실패" >&2; continue; }
    echo "    [O] $OUT_JSON"
done

echo
echo "=============================================================="
echo " 다음 단계: src/training/train.sh 에서 아래 두 줄만 맞춘다"
echo "   CV_SPLIT_FILE = config/cv_splits_tert_5fold_seed42_virchow2_40x_2class.json"
echo "   NUM_CLASSES   = 2"
echo " IN_DIM 은 split 의 embed_dim(2560) 에서 자동으로 읽는다."
echo
echo " 학습 (GPU 5):"
echo "   cd src/training && bash train.sh"
echo "=============================================================="
