#!/bin/bash
# ============================================================
# sweep.sh 로 돌린 run 들의 진행/결과 요약
#
#   bash sweep_status.sh                   outputs 전체
#   bash sweep_status.sh thyroid_tert_v9   이름에 해당 문자열이 든 것만
#
# AUC 기준 내림차순 정렬해서 보여준다.
# ============================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 - "$REPO_ROOT" "${1:-}" <<'PY'
import json, os, re, subprocess, sys
root, filt = sys.argv[1], sys.argv[2]
outs = os.path.join(root, "outputs")
logdir = os.path.join(outs, "sweep_logs")

# 지금 돌고 있는 학습 프로세스의 model_save_dir 집합
running = set()
try:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        if "main.py" in line:
            m = re.search(r"--model_save_dir\s+(\S+)", line)
            if m:
                running.add(os.path.realpath(m.group(1)))
except Exception:
    pass

rows = []
for name in sorted(os.listdir(outs)):
    d = os.path.join(outs, name)
    if not os.path.isdir(d) or name == "sweep_logs":
        continue
    if filt and filt not in name:
        continue

    res = os.path.join(d, "results_cv_summary_optimal.json")
    if os.path.isfile(res):
        try:
            j = json.load(open(res))
            t = j["summary_statistics"]["test"]
            get = lambda k: t.get(k, {}).get("mean")
            rows.append(dict(
                name=name, auc=get("auc"), f1=get("f1"), acc=get("accuracy"),
                sens=get("sensitivity"), spec=get("specificity"),
                # TERT 는 model_type / BRAF(v2) 는 model_name 으로 기록한다
                model=j.get("model_type") or j.get("model_name") or "?",
                nc=j.get("num_classes", 2),
                state="완료",
            ))
            continue
        except Exception as e:
            rows.append(dict(name=name, state=f"파싱실패({e.__class__.__name__})",
                             auc=None, f1=None, acc=None, sens=None, spec=None,
                             model="?", nc="?"))
            continue

    # 결과 없음 -> 진행중 / 중단 / 미시작
    state = "진행중" if os.path.realpath(d) in running else "미완료"
    ep = "-"
    try:
        logs = [os.path.join(logdir, f) for f in os.listdir(logdir)
                if f.startswith(name + "_")] if os.path.isdir(logdir) else []
        if logs:
            log = max(logs, key=os.path.getmtime)
            txt = open(log, errors="ignore").read()
            hits = re.findall(r"Epoch[ :]+(\d+)", txt)
            if hits:
                ep = hits[-1]
            if state == "미완료":
                state = "중단/실패"
    except Exception:
        pass
    rows.append(dict(name=name, state=state, auc=None, f1=None, acc=None,
                     sens=None, spec=None, model="?", nc="?", epoch=ep))

if not rows:
    print("해당하는 run 이 없습니다.")
    sys.exit(0)

# AUC 있는 것 먼저, 그 안에서 내림차순
rows.sort(key=lambda r: (r["auc"] is None, -(r["auc"] or 0)))

fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "   -  "
print(f"{'RUN':<34} {'MODEL':<9} {'C':<2} {'AUC':<7} {'F1':<7} "
      f"{'ACC':<7} {'SENS':<7} {'SPEC':<7} 상태")
print("-" * 105)
for r in rows:
    st = r["state"]
    if st in ("진행중", "중단/실패") and r.get("epoch", "-") != "-":
        st = f"{st} (ep {r['epoch']})"
    print(f"{r['name']:<34} {str(r['model']):<9} {str(r['nc']):<2} "
          f"{fmt(r['auc']):<7} {fmt(r['f1']):<7} {fmt(r['acc']):<7} "
          f"{fmt(r['sens']):<7} {fmt(r['spec']):<7} {st}")

done = [r for r in rows if r["auc"] is not None]
if len(done) > 1:
    b = done[0]
    print("-" * 105)
    print(f"최고 AUC: {b['name']}  AUC={b['auc']:.4f}  F1={fmt(b['f1'])}")
PY
