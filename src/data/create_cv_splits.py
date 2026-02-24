# -*- coding: utf-8 -*-
"""
Create K-Fold Cross-Validation Splits for TERT Mutation Prediction

Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
- Wild (no mutation) = 0
- Mutant (C228T or C250T) = 1

This script creates stratified K-fold splits based on TERT mutation labels.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

try:
    from .tert_common import load_tert_labels_from_excel, set_seed
except ImportError:  # pragma: no cover - script execution fallback
    from tert_common import load_tert_labels_from_excel, set_seed


def find_npy_files(data_root: str, labels_dict: dict) -> list:
    """
    데이터 디렉토리에서 NPY 파일을 찾고 label이 있는 것만 반환

    Args:
        data_root: NPY 파일이 있는 루트 디렉토리
        labels_dict: {sample_id: label} dictionary

    Returns:
        list of (filepath, sample_id, label) tuples
    """
    npy_files = []

    # Recursively find all .npy files
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.npy'):
                filepath = os.path.join(root, file)
                sample_id = file.replace('.npy', '')

                if sample_id in labels_dict:
                    label = labels_dict[sample_id]
                    npy_files.append((filepath, sample_id, label))

    print(f"Found {len(npy_files)} NPY files with labels")
    return npy_files


def create_stratified_kfold_splits(
    npy_files: list,
    n_splits: int = 5,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> dict:
    """
    Create stratified K-fold splits with explicit train/val/test ratios.

    Args:
        npy_files: list of (filepath, sample_id, label) tuples
        n_splits: number of folds
        seed: random seed
        train_ratio: overall train ratio
        val_ratio: overall val ratio
        test_ratio: overall test ratio (must match 1 / n_splits)

    Returns:
        cv_splits dict
    """
    set_seed(seed)

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.4f}")

    expected_test_ratio = 1.0 / n_splits
    if abs(test_ratio - expected_test_ratio) > 1e-6:
        raise ValueError(
            f"test_ratio {test_ratio:.4f} must match 1 / n_splits ({expected_test_ratio:.4f}). "
            "Adjust n_splits or test_ratio."
        )

    train_val_ratio = train_ratio + val_ratio
    val_ratio_from_trainval = val_ratio / train_val_ratio

    # Extract arrays
    filepaths = np.array([f[0] for f in npy_files])
    labels = np.array([f[2] for f in npy_files])

    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(filepaths, labels)):
        # Split train_val into train and val (stratified)
        train_val_labels = labels[train_val_idx]

        # Stratified split for validation
        from sklearn.model_selection import train_test_split
        train_idx_local, val_idx_local = train_test_split(
            np.arange(len(train_val_idx)),
            test_size=val_ratio_from_trainval,
            stratify=train_val_labels,
            random_state=seed
        )

        train_idx = train_val_idx[train_idx_local]
        val_idx = train_val_idx[val_idx_local]

        # Get file paths
        train_files = filepaths[train_idx].tolist()
        val_files = filepaths[val_idx].tolist()
        test_files = filepaths[test_idx].tolist()

        # Count labels
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        fold_info = {
            "fold": fold_idx + 1,
            "train_wsis": train_files,
            "val_wsis": val_files,
            "test_wsis": test_files,
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "train_pos_count": int(np.sum(train_labels == 1)),
            "train_neg_count": int(np.sum(train_labels == 0)),
            "val_pos_count": int(np.sum(val_labels == 1)),
            "val_neg_count": int(np.sum(val_labels == 0)),
            "test_pos_count": int(np.sum(test_labels == 1)),
            "test_neg_count": int(np.sum(test_labels == 0)),
        }

        folds.append(fold_info)

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_files)} (Mutant: {fold_info['train_pos_count']}, Wild: {fold_info['train_neg_count']})")
        print(f"  Val:   {len(val_files)} (Mutant: {fold_info['val_pos_count']}, Wild: {fold_info['val_neg_count']})")
        print(f"  Test:  {len(test_files)} (Mutant: {fold_info['test_pos_count']}, Wild: {fold_info['test_neg_count']})")

    cv_splits = {
        "task": "TERT Mutation Prediction",
        "n_splits": n_splits,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "val_ratio_from_trainval": val_ratio_from_trainval,
        "total_samples": len(npy_files),
        "total_mutant": int(np.sum(labels == 1)),
        "total_wild": int(np.sum(labels == 0)),
        "folds": folds
    }

    return cv_splits


def main():
    parser = argparse.ArgumentParser(description='Create K-Fold CV splits for TERT prediction')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing NPY embedding files')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Excel file with TERT mutation labels')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for CV split JSON')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Overall train ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Overall validation ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Overall test ratio (default: 0.2, must match 1/n_splits)')

    args = parser.parse_args()

    print("="*80)
    print("Creating K-Fold CV Splits for TERT Mutation Prediction")
    print("="*80)

    # Load labels
    print(f"\nLoading labels from: {args.label_file}")
    labels_dict = load_tert_labels_from_excel(args.label_file)
    print(f"  Total labels: {len(labels_dict)}")
    print(f"  Mutant: {sum(1 for v in labels_dict.values() if v == 1)}")
    print(f"  Wild: {sum(1 for v in labels_dict.values() if v == 0)}")

    # Find NPY files
    print(f"\nSearching for NPY files in: {args.data_root}")
    npy_files = find_npy_files(args.data_root, labels_dict)

    if len(npy_files) == 0:
        print("ERROR: No NPY files found with matching labels!")
        return

    # Create splits
    print(f"\nCreating {args.n_splits}-fold stratified splits (train/val/test = "
          f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f})...")
    cv_splits = create_stratified_kfold_splits(
        npy_files,
        n_splits=args.n_splits,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"cv_splits_tert_k{args.n_splits}_seed{args.seed}_v0.1.0.json"
    with open(output_file, 'w') as f:
        json.dump(cv_splits, f, indent=2)

    print(f"\n{'='*80}")
    print(f"[+] CV splits saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
