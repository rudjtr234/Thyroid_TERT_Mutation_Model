# -*- coding: utf-8 -*-
"""
Shared utilities for TERT data preprocessing/splitting scripts.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


TERT_BINARY_CLASS_NAMES = ["Wild", "Mutant"]
TERT_MULTICLASS_CLASS_NAMES = ["Wild", "C228T", "C250T"]


def set_seed(seed=42):
    """Set random seed for reproducibility (python/numpy)."""
    random.seed(seed)
    np.random.seed(seed)


def get_tert_class_names(num_classes: int = 2) -> List[str]:
    """Return canonical class names for the requested TERT classification mode."""
    if num_classes <= 2:
        return list(TERT_BINARY_CLASS_NAMES)
    if num_classes == 3:
        return list(TERT_MULTICLASS_CLASS_NAMES)
    raise ValueError(f"Unsupported num_classes for TERT task: {num_classes}")


def get_tert_task_label(num_classes: int = 2) -> str:
    """Return a human-readable task label for logging and metadata."""
    if num_classes <= 2:
        return "TERT Mutation Prediction (Wild vs Mutant)"
    return "TERT Mutation Prediction (Wild/C228T/C250T)"


def get_tert_label_mapping(num_classes: int = 2) -> Dict[str, int]:
    """Return class-name to index mapping for binary or 3-class TERT tasks."""
    if num_classes <= 2:
        return {"Wild": 0, "C228T": 1, "C250T": 1}
    if num_classes == 3:
        return {"Wild": 0, "C228T": 1, "C250T": 2}
    raise ValueError(f"Unsupported num_classes for TERT task: {num_classes}")


def map_tert_class_name_to_label(class_name: str, num_classes: int = 2) -> int:
    """Map a class folder/value name to its training label index."""
    mapping = get_tert_label_mapping(num_classes=num_classes)
    if class_name not in mapping:
        raise KeyError(f"Unknown TERT class name: {class_name}")
    return mapping[class_name]


def summarize_label_counts(labels_dict: Dict[str, int], num_classes: int = 2) -> Dict[str, int]:
    """Return class-wise counts keyed by display name."""
    class_names = get_tert_class_names(num_classes=num_classes)
    counts = Counter(labels_dict.values())
    return {
        class_name: int(counts.get(class_idx, 0))
        for class_idx, class_name in enumerate(class_names)
    }


def infer_embedding_label_from_cv_split(cv_split_file: str) -> str:
    """Infer encoder / patch setting label from a CV split filename or path."""
    lower_path = str(cv_split_file).lower()
    lower_name = Path(cv_split_file).name.lower()

    if "hoptimus_20x" in lower_name or "h_optimus_embeddings_20x" in lower_path:
        return "H-Optimus-0 20x"
    if "hoptimus" in lower_name or "h_optimus_embeddings" in lower_path:
        return "H-Optimus-0 40x"
    return "UNI2-H 40x"


def load_tert_labels_from_excel(excel_path: str, verbose: bool = False, num_classes: int = 2) -> dict:
    """
    Load TERT mutation labels from Excel.

    Returns:
        labels_dict: {sample_id: label}
    """
    df = pd.read_excel(excel_path)

    tert_col = None
    for col in df.columns:
        if "TERT" in col.upper() and "MUTATION" in col.upper():
            tert_col = col
            break

    if tert_col is None:
        raise ValueError(f"TERT mutation column not found in {excel_path}")

    id_col = "NO. (부여번호)"
    if id_col not in df.columns:
        for col in df.columns:
            if "NO" in col.upper() or "부여번호" in col:
                id_col = col
                break

    labels_dict = {}
    for _, row in df.iterrows():
        sample_id = str(row[id_col]).strip()
        tert_value = str(row[tert_col]).strip().upper()

        normalized_value = "Wild" if tert_value == "WILD" else tert_value

        try:
            label = map_tert_class_name_to_label(normalized_value, num_classes=num_classes)
        except KeyError:
            print(f"Warning: Unknown TERT value '{tert_value}' for {sample_id}, skipping...")
            continue

        labels_dict[sample_id] = label

    if verbose:
        print(f"Loaded {len(labels_dict)} labels from Excel")
        for idx, class_name in enumerate(get_tert_class_names(num_classes=num_classes)):
            count = sum(1 for value in labels_dict.values() if value == idx)
            print(f"  - {class_name} ({idx}): {count}")

    return labels_dict
