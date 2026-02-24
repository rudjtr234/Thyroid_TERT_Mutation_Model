# -*- coding: utf-8 -*-
"""
Shared utilities for TERT data preprocessing/splitting scripts.
"""

import random

import numpy as np
import pandas as pd


def set_seed(seed=42):
    """Set random seed for reproducibility (python/numpy)."""
    random.seed(seed)
    np.random.seed(seed)


def load_tert_labels_from_excel(excel_path: str, verbose: bool = False) -> dict:
    """
    Load TERT mutation labels from Excel.

    Returns:
        labels_dict: {sample_id: label} (0=Wild, 1=Mutant)
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

        if tert_value == "WILD":
            label = 0
        elif tert_value in ["C228T", "C250T"]:
            label = 1
        else:
            print(f"Warning: Unknown TERT value '{tert_value}' for {sample_id}, skipping...")
            continue

        labels_dict[sample_id] = label

    if verbose:
        print(f"Loaded {len(labels_dict)} labels from Excel")
        print(f"  - Wild (0): {sum(1 for v in labels_dict.values() if v == 0)}")
        print(f"  - Mutant (1): {sum(1 for v in labels_dict.values() if v == 1)}")

    return labels_dict
