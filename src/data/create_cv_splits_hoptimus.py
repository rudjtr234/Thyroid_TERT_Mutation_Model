"""
기존 UNI2-H CV split JSON의 경로를 H-Optimus-0 경로로 변환하여 새 JSON 생성.
fold 구조(train/val/test 샘플 구성)는 그대로 유지하고 경로만 변경.

Usage:
    python create_cv_splits_hoptimus.py

    # 파일 존재 여부까지 확인하려면 (임베딩 추출 완료 후):
    python create_cv_splits_hoptimus.py --verify
"""

import json
import argparse
from pathlib import Path

OLD_ROOT = "/path/to/dataset/embedding"
NEW_ROOT = "/path/to/dataset/h_optimus_embeddings"

INPUT_JSON = str(
    Path(__file__).parents[2] / "config" / "cv_splits_tert_5fold_seed42.json"
)
OUTPUT_JSON = str(
    Path(__file__).parents[2] / "config" / "cv_splits_tert_5fold_seed42_hoptimus.json"
)


def remap_path(old_path: str, old_root: str, new_root: str) -> str:
    try:
        relative = Path(old_path).relative_to(old_root)
        return str(Path(new_root) / relative)
    except ValueError:
        print(f"[WARN] 경로 변환 실패 (예상치 못한 prefix): {old_path}")
        return old_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json",  type=str, default=INPUT_JSON)
    parser.add_argument("--output_json", type=str, default=OUTPUT_JSON)
    parser.add_argument("--old_root",    type=str, default=OLD_ROOT)
    parser.add_argument("--new_root",    type=str, default=NEW_ROOT)
    parser.add_argument("--verify", action="store_true",
                        help="변환된 경로의 npy 파일 존재 여부 확인")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    new_data = dict(data)
    new_data["embedding_root"] = args.new_root
    new_data["encoder"] = "H-Optimus-0"
    new_data["embed_dim"] = 1536
    new_data["source_json"] = args.input_json

    missing = []
    new_folds = []
    for fold in data["folds"]:
        new_fold = dict(fold)
        for split in ("train_wsis", "val_wsis", "test_wsis"):
            remapped = [remap_path(p, args.old_root, args.new_root) for p in fold[split]]
            new_fold[split] = remapped
            if args.verify:
                for p in remapped:
                    if not Path(p).exists():
                        missing.append(p)
        new_folds.append(new_fold)

    new_data["folds"] = new_folds

    # 고유 WSI 수 집계 (fold 중복 제거, test_wsis 기준)
    all_wsis = set()
    for fold in new_folds:
        for s in ("train_wsis", "val_wsis", "test_wsis"):
            all_wsis.update(fold[s])
    new_data["total_wsis"] = len(all_wsis)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"[+] 저장 완료: {out_path}")

    total = sum(
        len(fold[s])
        for fold in new_folds
        for s in ("train_wsis", "val_wsis", "test_wsis")
    )
    print(f"[INFO] {len(new_folds)} folds, 총 {total}개 경로 변환")

    if args.verify:
        if missing:
            print(f"\n[WARN] 존재하지 않는 파일 {len(missing)}개:")
            for p in missing[:20]:
                print(f"  {p}")
            if len(missing) > 20:
                print(f"  ... 외 {len(missing)-20}개")
        else:
            print(f"[OK] 변환된 경로 {total}개 모두 존재 확인")


if __name__ == "__main__":
    main()
