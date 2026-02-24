# src/data/datasets.py
"""
Thyroid TERT Mutation Prediction Dataset

Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
- Wild (no mutation) = 0
- Mutant (C228T or C250T) = 1

Label mapping from Excel file based on 'TERT mutation' column.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    from .tert_common import load_tert_labels_from_excel as _load_tert_labels_from_excel
    from .tert_common import set_seed as _set_seed_common
except ImportError:  # pragma: no cover - script execution fallback
    from tert_common import load_tert_labels_from_excel as _load_tert_labels_from_excel
    from tert_common import set_seed as _set_seed_common


class TERTWSIDataset(Dataset):
    """
    Thyroid TERT Mutation WSI Dataset
    - Binary classification: Wild (0) vs Mutant (1)
    - Label mapping: Wild = 0, C228T/C250T = 1
    """
    def __init__(self, wsi_files, labels_dict, bag_size=2000, use_variance=False):
        """
        Args:
            wsi_files: WSI 파일 경로 리스트
            labels_dict: {sample_id: label} 딕셔너리 (0=Wild, 1=Mutant)
            bag_size: 각 WSI에서 사용할 타일 개수
            use_variance: True면 분산 기준, False면 랜덤 (기본)
        """
        self.wsi_list = []
        self.bag_size = bag_size
        self.use_variance = use_variance

        for filepath in wsi_files:
            # Extract sample ID from filename (e.g., TC_04_8001.npy -> TC_04_8001)
            filename = os.path.basename(filepath)
            sample_id = filename.replace('.npy', '').replace('.pt', '')

            # Get label from labels_dict
            if sample_id in labels_dict:
                label = labels_dict[sample_id]
            else:
                print(f"Warning: Sample ID '{sample_id}' not found in labels_dict, skipping...")
                continue

            self.wsi_list.append({
                'filepath': filepath,
                'label': label,
                'filename': filename,
                'sample_id': sample_id
            })

        labels = [wsi['label'] for wsi in self.wsi_list]
        print(f"Dataset: {len(self.wsi_list)} WSIs (Mutant: {sum(labels)}, Wild: {len(labels) - sum(labels)})")

    def _select_tiles(self, features):
        """타일 선택: Random (use_variance=False) or Top-K Variance"""
        if len(features) <= self.bag_size:
            return features

        if self.use_variance:
            variances = np.var(features, axis=1)
            top_k_indices = np.argsort(variances)[::-1][:self.bag_size]
            return features[top_k_indices]
        else:
            # Random Sampling
            indices = np.random.choice(len(features), self.bag_size, replace=False)
            return features[indices]

    def __len__(self):
        return len(self.wsi_list)

    def __getitem__(self, idx):
        wsi = self.wsi_list[idx]
        features = np.load(wsi['filepath'])
        features = self._select_tiles(features)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(wsi['label'], dtype=torch.long)

        return features, label, wsi['filename']


def load_tert_labels_from_excel(excel_path: str) -> dict:
    """Excel 파일에서 TERT mutation label을 로드."""
    return _load_tert_labels_from_excel(excel_path, verbose=True)


def load_tert_labels_from_cv_splits(cv_split_file: str) -> dict:
    """
    CV split JSON 파일의 경로에서 TERT mutation label을 추출

    경로 구조: .../embedding/{C228T|C250T|Wild}/npy/{sample_id}.npy
    - Wild 폴더 → label 0
    - C228T, C250T 폴더 → label 1

    Args:
        cv_split_file: CV split JSON 파일 경로

    Returns:
        labels_dict: {sample_id: label} (0=Wild, 1=Mutant)
    """
    with open(cv_split_file, 'r') as f:
        cv_data = json.load(f)

    labels_dict = {}

    # 모든 fold에서 경로 수집
    all_paths = set()
    for fold_data in cv_data['folds']:
        for split in ['train_wsis', 'val_wsis', 'test_wsis']:
            all_paths.update(fold_data.get(split, []))

    # 경로에서 라벨 추출
    for filepath in all_paths:
        # 경로에서 sample_id와 class 추출
        # 예: /path/to/.../embedding/C228T/npy/TC_04_8001.npy
        parts = Path(filepath).parts
        filename = Path(filepath).stem  # TC_04_8001

        # npy 폴더의 상위 폴더가 클래스명
        try:
            npy_idx = parts.index('npy')
            class_name = parts[npy_idx - 1]  # C228T, C250T, or Wild
        except (ValueError, IndexError):
            print(f"Warning: Cannot extract class from path '{filepath}', skipping...")
            continue

        # Label mapping
        if class_name == 'Wild':
            label = 0
        elif class_name in ['C228T', 'C250T']:
            label = 1
        else:
            print(f"Warning: Unknown class '{class_name}' for {filename}, skipping...")
            continue

        labels_dict[filename] = label

    print(f"Loaded {len(labels_dict)} labels from CV split paths")
    print(f"  - Wild (0): {sum(1 for v in labels_dict.values() if v == 0)}")
    print(f"  - Mutant (1): {sum(1 for v in labels_dict.values() if v == 1)}")

    return labels_dict


def load_json_splits(json_path: str, labels_dict: dict, bag_size: int = 2000):
    """
    JSON 파일에서 미리 정의된 K-Fold split 로드

    Args:
        json_path: CV split JSON 파일 경로
        labels_dict: {sample_id: label} 딕셔너리
        bag_size: bag size for dataset

    Returns:
        fold_datasets: list of fold dataset dictionaries
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)

    fold_datasets = []

    for fold_info in split_data['folds']:
        fold_idx = fold_info['fold']

        # JSON에 전체 경로가 저장되어 있으므로 그대로 사용
        train_files = fold_info['train_wsis']
        val_files = fold_info['val_wsis']
        test_files = fold_info['test_wsis']

        # 파일 존재 여부 확인
        train_files_exist = [f for f in train_files if os.path.exists(f)]
        val_files_exist = [f for f in val_files if os.path.exists(f)]
        test_files_exist = [f for f in test_files if os.path.exists(f)]

        # 누락된 파일 경고
        if len(train_files_exist) != len(train_files):
            print(f"Warning: Fold {fold_idx} - {len(train_files) - len(train_files_exist)} train files not found")
        if len(val_files_exist) != len(val_files):
            print(f"Warning: Fold {fold_idx} - {len(val_files) - len(val_files_exist)} val files not found")
        if len(test_files_exist) != len(test_files):
            print(f"Warning: Fold {fold_idx} - {len(test_files) - len(test_files_exist)} test files not found")

        # Dataset 생성
        train_dataset = TERTWSIDataset(train_files_exist, labels_dict, bag_size=bag_size, use_variance=False)
        val_dataset = TERTWSIDataset(val_files_exist, labels_dict, bag_size=bag_size, use_variance=False)
        test_dataset = TERTWSIDataset(test_files_exist, labels_dict, bag_size=bag_size, use_variance=False)

        fold_datasets.append({
            'fold': fold_idx,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        })

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_dataset)} WSIs")
        print(f"  Val: {len(val_dataset)} WSIs")
        print(f"  Test: {len(test_dataset)} WSIs")

    return fold_datasets


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    _set_seed_common(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ========================================
# 사용 예시
# ========================================
if __name__ == "__main__":
    set_seed(42)

    # Excel에서 label 로드
    excel_path = "/path/to/Thyroid_TERT_labels.xlsx"
    labels_dict = load_tert_labels_from_excel(excel_path)

    print(f"\nSample labels:")
    for i, (k, v) in enumerate(list(labels_dict.items())[:5]):
        print(f"  {k}: {v} ({'Mutant' if v == 1 else 'Wild'})")
