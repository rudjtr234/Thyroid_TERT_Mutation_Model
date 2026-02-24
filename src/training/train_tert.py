# -*- coding: utf-8 -*-
"""
Thyroid TERT Mutation Prediction Model Training Script
K-Fold Cross-Validation with Fixed Threshold (0.5)
Test evaluation with FULL WSI (all patches)

Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
- Wild (no mutation) = 0
- Mutant (C228T or C250T) = 1

Version: thyroid_tert_v0.1.0
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import warnings
import sys
from multiprocessing import cpu_count

# GPU 설정 강제 (코드 최상단에서 설정)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"CUDA_VISIBLE_DEVICES not set, forcing GPU 0")
else:
    print(f"CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

warnings.filterwarnings('ignore')

# =========================
# 경로 설정
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

# =========================
# Models & Utils import
# =========================
from models.abmil import ABMILTERTModel, ABMILTERTConfig
from models.transmil import TransMILTERTModel, TransMILTERTConfig
from data.datasets import TERTWSIDataset, load_tert_labels_from_cv_splits, set_seed
from evaluation.metric import (
    compute_metrics_with_confusion,
    compute_roc_curve_data,
    compute_precision_recall_curve_data,
    compute_summary_statistics,
    print_fold_table,
    print_summary_statistics,
    print_confusion_matrix_summary,
    generate_all_plots
)
from evaluation.visualization import generate_attention_heatmaps_from_results
import torch.nn.functional as F
from typing import Dict, List
from torch.utils.data import DataLoader


# =========================
# EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001, restore_best_weights=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        if mode not in ('max', 'min'):
            raise ValueError("mode must be either 'max' or 'min'")
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, score, model=None, epoch=None):
        improved = False
        if self.best_score is None:
            self.best_score = score
            improved = True
        elif self.mode == 'max' and score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            improved = True
        elif self.mode == 'min' and score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            improved = True
        else:
            self.counter += 1

        if improved and model is not None:
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch is not None:
                self.best_epoch = epoch

        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# =========================
# Model Checkpoint
# =========================
def _resolve_model_hparams(args) -> Dict:
    """Build model hyper-parameter payload for checkpoint serialization."""
    model_type = getattr(args, 'model_type', 'abmil')
    use_layer_norm = not getattr(args, 'disable_layer_norm', False)

    if model_type == 'transmil':
        return {
            'model_type': 'transmil',
            'in_dim': int(getattr(args, 'in_dim', 1536)),
            'embed_dim': int(getattr(args, 'transmil_embed_dim', 512)),
            'num_heads': int(getattr(args, 'transmil_num_heads', 8)),
            'num_layers': int(getattr(args, 'transmil_num_layers', 2)),
            'num_landmarks': int(getattr(args, 'transmil_num_landmarks', 256)),
            'pinv_iterations': int(getattr(args, 'transmil_pinv_iterations', 6)),
            'dropout': float(getattr(args, 'dropout', 0.25)),
            'num_classes': 2,
            'use_layer_norm': use_layer_norm,
        }

    return {
        'model_type': 'abmil',
        'gate': True,
        'in_dim': int(getattr(args, 'in_dim', 1536)),
        'embed_dim': int(getattr(args, 'embed_dim', 512)),
        'attn_dim': int(getattr(args, 'attn_dim', 384)),
        'num_fc_layers': int(getattr(args, 'num_fc_layers', 2)),
        'dropout': float(getattr(args, 'dropout', 0.25)),
        'num_classes': 2,
        'use_layer_norm': use_layer_norm,
    }


def build_model_from_args(args):
    """Instantiate ABMIL/TransMIL from CLI args."""
    model_hparams = _resolve_model_hparams(args)
    model_type = model_hparams['model_type']

    if model_type == 'transmil':
        config = TransMILTERTConfig(
            in_dim=model_hparams['in_dim'],
            embed_dim=model_hparams['embed_dim'],
            num_heads=model_hparams['num_heads'],
            num_layers=model_hparams['num_layers'],
            num_landmarks=model_hparams['num_landmarks'],
            pinv_iterations=model_hparams['pinv_iterations'],
            dropout=model_hparams['dropout'],
            num_classes=model_hparams['num_classes'],
            use_layer_norm=model_hparams['use_layer_norm'],
        )
        return TransMILTERTModel(config), config, model_hparams

    config = ABMILTERTConfig(
        gate=model_hparams['gate'],
        in_dim=model_hparams['in_dim'],
        embed_dim=model_hparams['embed_dim'],
        attn_dim=model_hparams['attn_dim'],
        num_fc_layers=model_hparams['num_fc_layers'],
        dropout=model_hparams['dropout'],
        num_classes=model_hparams['num_classes'],
        use_layer_norm=model_hparams['use_layer_norm'],
    )
    return ABMILTERTModel(config), config, model_hparams


def save_model_checkpoint(model, fold_idx, fold_result, save_dir, args, is_best=False):
    save_dir = Path(save_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'fold': fold_idx + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': fold_result['test_metrics_optimal']['accuracy'],
        'auc': fold_result['test_metrics_optimal']['auc'],
        'optimal_threshold': fold_result['optimal_threshold'],
        'config': {
            'lr': args.lr,
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': f"{getattr(args, 'model_type', 'abmil').upper()}_TERT",
            'model_type': getattr(args, 'model_type', 'abmil'),
            'model_hparams': _resolve_model_hparams(args),
        }
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx+1}_auc{fold_result['test_metrics_optimal']['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)

    print(f"[+] Model saved: {checkpoint_path}")
    return checkpoint_path


# =========================
# CV Split Loading
# =========================
def load_cv_splits_with_paths(cv_split_file, labels_dict, debug=False):
    """Load CV splits from JSON file"""
    with open(cv_split_file, 'r') as f:
        cv_splits = json.load(f)

    # Validate that all samples in splits have labels
    for fold_data in cv_splits['folds']:
        for split_name in ['train_wsis', 'val_wsis', 'test_wsis']:
            valid_paths = []
            for filepath in fold_data[split_name]:
                if os.path.exists(filepath):
                    # Check if sample has label
                    sample_id = os.path.basename(filepath).replace('.npy', '').replace('.pt', '')
                    if sample_id in labels_dict:
                        valid_paths.append(filepath)
                    elif debug:
                        print(f"Debug: {sample_id} not in labels_dict")
            fold_data[f'{split_name}_paths'] = valid_paths

    return cv_splits


def check_data_leakage(cv_splits):
    """CV split data leakage check"""
    print("\n" + "="*80)
    print("Checking Data Leakage")
    print("="*80)

    all_folds_ok = True

    for fold_data in cv_splits['folds']:
        fold_num = fold_data['fold']

        train_files = set([os.path.basename(p) for p in fold_data['train_wsis_paths']])
        val_files = set([os.path.basename(p) for p in fold_data['val_wsis_paths']])
        test_files = set([os.path.basename(p) for p in fold_data['test_wsis_paths']])

        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_files):3d} files")
        print(f"  Val:   {len(val_files):3d} files")
        print(f"  Test:  {len(test_files):3d} files")
        print(f"  Total: {len(train_files) + len(val_files) + len(test_files):3d} files")

        train_val_overlap = train_files & val_files
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files

        has_overlap = False

        if train_val_overlap:
            print(f"  [X] Train-Val overlap: {len(train_val_overlap)} files")
            has_overlap = True
            all_folds_ok = False

        if train_test_overlap:
            print(f"  [X] Train-Test overlap: {len(train_test_overlap)} files")
            has_overlap = True
            all_folds_ok = False

        if val_test_overlap:
            print(f"  [X] Val-Test overlap: {len(val_test_overlap)} files")
            has_overlap = True
            all_folds_ok = False

        if not has_overlap:
            print(f"  [OK] No overlap detected")

    print("\n" + "="*80)
    if all_folds_ok:
        print("[OK] All folds passed leakage check")
    else:
        print("[X] DATA LEAKAGE DETECTED!")
    print("="*80 + "\n")

    return all_folds_ok


def check_label_distribution(cv_splits, labels_dict):
    """각 fold의 TERT label 분포 확인 (Wild vs Mutant)"""
    print("\n" + "="*80)
    print("TERT Label Distribution Analysis (Wild vs Mutant)")
    print("="*80)

    for fold_data in cv_splits['folds']:
        fold_num = fold_data['fold']

        # Count labels for each split
        def count_labels(paths):
            pos = sum(1 for p in paths if labels_dict.get(os.path.basename(p).replace('.npy', '').replace('.pt', ''), 0) == 1)
            neg = len(paths) - pos
            return pos, neg

        train_pos, train_neg = count_labels(fold_data['train_wsis_paths'])
        val_pos, val_neg = count_labels(fold_data['val_wsis_paths'])
        test_pos, test_neg = count_labels(fold_data['test_wsis_paths'])

        train_total = train_pos + train_neg
        val_total = val_pos + val_neg
        test_total = test_pos + test_neg

        print(f"\nFold {fold_num}:")
        print(f"  Train: Mutant={train_pos:3d} ({train_pos/train_total*100 if train_total > 0 else 0:5.1f}%), "
              f"Wild={train_neg:3d} ({train_neg/train_total*100 if train_total > 0 else 0:5.1f}%), "
              f"Total={train_total:3d}")
        print(f"  Val  : Mutant={val_pos:3d} ({val_pos/val_total*100 if val_total > 0 else 0:5.1f}%), "
              f"Wild={val_neg:3d} ({val_neg/val_total*100 if val_total > 0 else 0:5.1f}%), "
              f"Total={val_total:3d}")
        print(f"  Test : Mutant={test_pos:3d} ({test_pos/test_total*100 if test_total > 0 else 0:5.1f}%), "
              f"Wild={test_neg:3d} ({test_neg/test_total*100 if test_total > 0 else 0:5.1f}%), "
              f"Total={test_total:3d}")

    print("\n" + "="*80 + "\n")


# =========================
# Training Loop (threshold=0.5 고정)
# =========================
def run_one_epoch(model, dataloader, device, optimizer=None, train=False, threshold=0.5):
    """Training/Validation with fixed threshold=0.5"""
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    loss_fn = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)

            if train:
                optimizer.zero_grad()
                results_dict = model(h=features, loss_fn=loss_fn, label=label, return_extra=True)
                loss = results_dict['loss']
                logits = results_dict['logits']
                loss.backward()
                optimizer.step()
            else:
                results_dict = model(h=features, loss_fn=loss_fn, label=label, return_extra=True)
                loss = results_dict['loss']
                logits = results_dict['logits']

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().detach().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    if all_labels:
        metrics = compute_metrics_with_confusion(all_labels, all_preds, all_probs)
    else:
        metrics = {
            "accuracy": 0.0, "auc": 0.5,
            "sensitivity": 0.0, "specificity": 0.0,
            "ppv": 0.0, "npv": 0.0,
            "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    metrics["loss"] = avg_loss
    return metrics


# ============================================================
# Full WSI Test Evaluation (전체 NPY 직접 로드)
# ============================================================
def evaluate_full_wsi_for_test_direct(
    model: nn.Module,
    test_wsi_paths: List[str],
    labels_dict: dict,
    device: torch.device,
    threshold: float = 0.5,
    model_type: str = "abmil",
):
    """
    Test evaluation with FULL WSI (all patches) - NPY 경로 직접 사용

    Args:
        model: trained model
        test_wsi_paths: 전체 NPY 파일 경로 리스트
        labels_dict: {sample_id: label} dictionary
        device: torch device
        threshold: classification threshold

    Returns:
        metrics: evaluation metrics
        all_probs: prediction probabilities
        all_labels: true labels
        attention_scores_dict: attention scores with metadata
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []
    attention_scores_dict = {}

    print(f"\n{'-'*80}")
    print(f"Testing with FULL WSI (all patches) - Direct NPY Loading")
    print(f"{'-'*80}")

    successful = 0
    failed = 0

    for wsi_path in test_wsi_paths:
        wsi_name = Path(wsi_path).stem
        sample_id = wsi_name

        # Get true label
        if sample_id not in labels_dict:
            print(f"[!] Label not found for {sample_id}")
            failed += 1
            continue

        true_label = labels_dict[sample_id]

        # NPY 파일 로드
        if not os.path.exists(wsi_path):
            print(f"[!] NPY not found: {wsi_path}")
            failed += 1
            continue

        try:
            full_features = np.load(wsi_path)
            n_patches = full_features.shape[0]

            # 전체 패치로 forward pass
            features_tensor = torch.from_numpy(full_features).float().unsqueeze(0).to(device)
            use_transmil_attr = (
                str(model_type).lower() == "transmil"
                and hasattr(model, "get_patch_attribution")
            )

            if use_transmil_attr:
                with torch.enable_grad():
                    attr_dict = model.get_patch_attribution(
                        h=features_tensor,
                        target_class=None,
                        use_pred_class=True,
                        positive_only=True,
                    )
                    logits = attr_dict["logits"].detach()
                    attr_scores = attr_dict["attribution_scores"].detach()[0]
            else:
                with torch.no_grad():
                    results_dict = model(
                        h=features_tensor,
                        loss_fn=None,
                        label=None,
                        return_attention=True,
                        return_extra=True
                    )
                    logits = results_dict['logits']

            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs >= threshold).long()

            # Attention/Attribution 저장
            if use_transmil_attr:
                attn_scores = attr_scores.cpu().numpy()
                attention_scores_dict[wsi_name] = {
                    'scores': attn_scores.tolist(),
                    'n_patches': len(attn_scores),
                    'predicted_label': int(pred.cpu().numpy()[0]),
                    'pred_prob': float(probs.cpu().numpy()[0]),
                    'true_label': int(true_label),
                    'model_type': str(model_type).lower(),
                    'score_type': 'transmil_b_attribution',
                }
            elif 'attention' in results_dict and results_dict['attention'] is not None:
                attention_weights = results_dict['attention']

                # Accept tensor outputs from ABMIL/TransMIL and dict-based custom outputs.
                if isinstance(attention_weights, dict):
                    if 'A' in attention_weights and attention_weights['A'] is not None:
                        attn_raw = attention_weights['A']
                    elif 'cls_attn_logits' in attention_weights and attention_weights['cls_attn_logits'] is not None:
                        attn_raw = attention_weights['cls_attn_logits']
                    else:
                        print(f"[!] Unsupported attention dict keys for {wsi_name}: {list(attention_weights.keys())}")
                        failed += 1
                        continue
                else:
                    attn_raw = attention_weights

                if not isinstance(attn_raw, torch.Tensor):
                    attn_raw = torch.tensor(attn_raw, device=features_tensor.device)

                if attn_raw.dim() == 3:
                    attn_raw = attn_raw[0, 0, :]
                elif attn_raw.dim() == 2:
                    attn_raw = attn_raw[0, :]
                elif attn_raw.dim() == 1:
                    attn_raw = attn_raw
                else:
                    print(f"[!] Unexpected attention shape for {wsi_name}: {tuple(attn_raw.shape)}")
                    failed += 1
                    continue

                attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()

                attention_scores_dict[wsi_name] = {
                    'scores': attn_scores.tolist(),
                    'n_patches': len(attn_scores),
                    'predicted_label': int(pred.cpu().numpy()[0]),
                    'pred_prob': float(probs.cpu().numpy()[0]),
                    'true_label': int(true_label),
                    'model_type': str(model_type).lower(),
                    'score_type': 'attention',
                }

            # Metrics 계산용
            all_probs.append(float(probs.cpu().numpy()[0]))
            all_labels.append(int(true_label))
            all_preds.append(int(pred.cpu().numpy()[0]))

            label_str = "Mutant" if true_label == 1 else "Wild"
            pred_str = "Mutant" if pred.item() == 1 else "Wild"
            print(f"[+] {wsi_name}: {n_patches} patches, prob={probs.item():.4f}, pred={pred_str}, true={label_str}")
            successful += 1

        except Exception as e:
            print(f"[-] Failed to process {wsi_name}: {e}")
            failed += 1
            continue

    print(f"{'-'*80}")
    print(f"[+] Successfully processed: {successful}/{len(test_wsi_paths)} WSIs")
    if failed > 0:
        print(f"[-] Failed: {failed}/{len(test_wsi_paths)} WSIs")
    print(f"{'-'*80}\n")

    # Metrics 계산
    if all_labels:
        metrics = compute_metrics_with_confusion(all_labels, all_preds, all_probs)
    else:
        metrics = {
            "accuracy": 0.0, "auc": 0.5,
            "sensitivity": 0.0, "specificity": 0.0,
            "ppv": 0.0, "npv": 0.0,
            "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    return metrics, all_probs, all_labels, attention_scores_dict


# ============================================================
# Full WSI Validation Evaluation (전체 NPY 직접 로드, Attention 없음)
# ============================================================
def evaluate_full_wsi_for_validation(
    model: nn.Module,
    val_wsi_paths: List[str],
    labels_dict: dict,
    device: torch.device,
    threshold: float = 0.5,
    verbose: bool = False
):
    """
    Validation evaluation with FULL WSI (all patches) - 간소화 버전

    Args:
        model: trained model
        val_wsi_paths: 전체 NPY 파일 경로 리스트
        labels_dict: {sample_id: label} dictionary
        device: torch device
        threshold: classification threshold
        verbose: 상세 출력 여부

    Returns:
        metrics: evaluation metrics
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []
    total_loss = 0
    processed_count = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for wsi_path in val_wsi_paths:
            wsi_name = Path(wsi_path).stem
            sample_id = wsi_name

            if sample_id not in labels_dict:
                continue

            true_label = labels_dict[sample_id]

            if not os.path.exists(wsi_path):
                continue

            try:
                full_features = np.load(wsi_path)
                features_tensor = torch.from_numpy(full_features).float().unsqueeze(0).to(device)
                label_tensor = torch.tensor([true_label], dtype=torch.long).to(device)

                results_dict = model(
                    h=features_tensor,
                    loss_fn=loss_fn,
                    label=label_tensor,
                    return_extra=True
                )

                logits = results_dict['logits']
                loss = results_dict['loss']
                probs = torch.softmax(logits, dim=1)[:, 1]
                pred = (probs >= threshold).long()

                total_loss += loss.item()
                all_probs.append(float(probs.cpu().numpy()[0]))
                all_labels.append(int(true_label))
                all_preds.append(int(pred.cpu().numpy()[0]))
                processed_count += 1

            except Exception as e:
                if verbose:
                    print(f"[-] Failed to process {wsi_name}: {e}")
                continue

    # Metrics 계산
    avg_loss = total_loss / processed_count if processed_count > 0 else 0.0

    if all_labels:
        metrics = compute_metrics_with_confusion(all_labels, all_preds, all_probs)
    else:
        metrics = {
            "accuracy": 0.0, "auc": 0.5,
            "sensitivity": 0.0, "specificity": 0.0,
            "ppv": 0.0, "npv": 0.0,
            "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    metrics["loss"] = avg_loss
    return metrics


# =========================
# K-Fold CV (Fixed Threshold, AUC-based Early Stopping)
# =========================
def run_k_fold_cv(cv_splits, labels_dict, args, device):
    all_fold_results = []
    all_predictions, all_true_labels = [], []
    saved_model_paths = []
    model_hparams = _resolve_model_hparams(args)

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1

        if hasattr(args, 'test_fold') and args.test_fold is not None and fold_data['fold'] != args.test_fold:
            continue

        print(f"\n{'='*80}")
        print(f"Fold {fold_data['fold']}/{len(cv_splits['folds'])}")
        print(f"Fixed threshold (0.5) - Loss-based Early Stopping")
        print(f"Model type: {model_hparams['model_type']}")
        if hasattr(args, 'test_fold') and args.test_fold is not None:
            print(f"[!] TEST MODE: Running only Fold {args.test_fold}")
        print(f"{'='*80}")

        # DataLoader (Training만 샘플링 사용, Validation은 FULL WSI)
        train_dataset = TERTWSIDataset(fold_data['train_wsis_paths'], labels_dict, bag_size=args.bag_size, use_variance=False)

        print(f"Train size: {len(train_dataset)}, Val size: {len(fold_data['val_wsis_paths'])} (FULL WSI), Test size: {len(fold_data['test_wsis_paths'])} (FULL WSI)")

        # Calculate optimal num_workers
        n_cpus = cpu_count()
        train_workers = max(8, int(n_cpus * 0.4))

        print(f"DataLoader workers: train={train_workers} (Total CPUs: {n_cpus})")
        print(f"Validation/Test: FULL WSI evaluation (all patches)")

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=train_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        # Model
        model, _, _ = build_model_from_args(args)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=8, min_delta=0.001, mode='min')

        best_train_metrics, best_val_metrics = {}, {}
        history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [],
                   'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []}

        print(f"\n{'-'*80}")
        print(f"Starting training for Fold {fold_data['fold']}")
        print(f"{'-'*80}\n")

        # Training
        for epoch in range(args.epochs):
            train_metrics = run_one_epoch(model, train_loader, device, optimizer, train=True, threshold=0.5)
            # Validation with FULL WSI (all patches)
            val_metrics = evaluate_full_wsi_for_validation(
                model, fold_data['val_wsis_paths'], labels_dict, device, threshold=0.5
            )
            scheduler.step(val_metrics["loss"])

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_auc'].append(train_metrics['auc'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_f1'].append(val_metrics['f1'])

            print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_metrics["loss"] < best_val_metrics.get("loss", float('inf')):
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                print(f"  ^ Best model updated (Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f})")

            if early_stopping(val_metrics["loss"], model, epoch+1):
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch was {early_stopping.best_epoch} with Val Loss: {best_val_metrics['loss']:.4f}")
                early_stopping.restore_best(model)
                break

        # ============================================================
        # Test Set 평가 (FULL WSI - 전체 패치 사용)
        # ============================================================
        print(f"\n{'='*80}")
        print(f"Testing with FULL WSI (all patches) - Threshold=0.5")
        print(f"{'='*80}")

        optimal_threshold = 0.5

        # 전체 NPY로 Test 평가 + Attention 추출
        test_metrics_optimal, test_probs, test_labels, full_attention_scores = \
            evaluate_full_wsi_for_test_direct(
                model,
                fold_data['test_wsis_paths'],
                labels_dict,
                device,
                threshold=optimal_threshold,
                model_type=getattr(args, 'model_type', 'abmil'),
            )

        if len(test_labels) == 0 or len(test_probs) == 0:
            raise RuntimeError(
                "No test predictions were produced for this fold. "
                "Check model forward/attribution extraction errors in the test logs."
            )

        # ROC/PR curve
        fpr, tpr, _ = compute_roc_curve_data(test_labels, test_probs)
        precision, recall, _ = compute_precision_recall_curve_data(test_labels, test_probs)

        # 결과 출력
        print(f"\n[*] Test Results with FULL WSI (Threshold={optimal_threshold:.2f}):")
        print(f"{'-'*80}")
        print(f"{'Metric':<15} {'Value':<10}")
        print(f"{'-'*80}")
        print(f"{'AUC':<15} {test_metrics_optimal['auc']:<10.4f}")
        print(f"{'Accuracy':<15} {test_metrics_optimal['accuracy']:<10.4f}")
        print(f"{'F1-Score':<15} {test_metrics_optimal['f1']:<10.4f}")
        print(f"{'Sensitivity':<15} {test_metrics_optimal['sensitivity']:<10.4f}")
        print(f"{'Specificity':<15} {test_metrics_optimal['specificity']:<10.4f}")
        print(f"{'PPV':<15} {test_metrics_optimal['ppv']:<10.4f}")
        print(f"{'NPV':<15} {test_metrics_optimal['npv']:<10.4f}")
        print(f"{'-'*80}")
        print(f"{'TP':<15} {test_metrics_optimal['tp']:<10}")
        print(f"{'TN':<15} {test_metrics_optimal['tn']:<10}")
        print(f"{'FP':<15} {test_metrics_optimal['fp']:<10}")
        print(f"{'FN':<15} {test_metrics_optimal['fn']:<10}")
        print(f"{'-'*80}")
        print(f"[+] Used FULL WSI for {len(full_attention_scores)} test samples\n")

        # Fold 결과 출력
        print_fold_table(fold_data['fold'], best_train_metrics, best_val_metrics, test_metrics_optimal)

        # Fold 결과 저장
        fold_result = {
            "fold": fold_data['fold'],
            "train_size": len(train_dataset),
            "val_size": len(fold_data['val_wsis_paths']),
            "test_size": len(fold_data['test_wsis_paths']),
            "best_epoch": early_stopping.best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics_optimal": test_metrics_optimal,
            "optimal_threshold": optimal_threshold,
            "optimal_val_f1": best_val_metrics.get("f1", 0.0),
            "history": history,
            "test_fpr": fpr.tolist(),
            "test_tpr": tpr.tolist(),
            "test_precision": precision.tolist(),
            "test_recall": recall.tolist(),
            "full_attention_scores": full_attention_scores
        }

        all_fold_results.append(fold_result)
        all_predictions.extend(test_probs)
        all_true_labels.extend(test_labels)

        # Model saving
        if args.save_model:
            if args.save_best_only:
                if fold_idx == 0 or test_metrics_optimal['auc'] > max(r['test_metrics_optimal']['auc'] for r in all_fold_results[:-1]):
                    path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args, is_best=True)
                    saved_model_paths.append(path)
                    print(f"[+] Best model saved")
            else:
                path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args)
                saved_model_paths.append(path)
                print(f"[+] Model checkpoint saved")

    # Summary 계산
    summary_stats_optimal = compute_summary_statistics(
        [{**r, 'test_metrics': r['test_metrics_optimal']} for r in all_fold_results]
    )

    return all_fold_results, all_predictions, all_true_labels, saved_model_paths, summary_stats_optimal


def convert_numpy(obj):
    """NumPy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


# =========================
# Training Pipeline
# =========================
def run_training(args):
    """Run the complete training pipeline."""
    set_seed(args.seed)
    if not hasattr(args, 'model_type'):
        args.model_type = 'abmil'

    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    if not Path(args.cv_split_file).exists():
        raise ValueError(f"CV split file not found: {args.cv_split_file}")

    # Load labels from CV split paths
    labels_dict = load_tert_labels_from_cv_splits(args.cv_split_file)

    # Data validation
    cv_splits = load_cv_splits_with_paths(args.cv_split_file, labels_dict, debug=args.debug)
    leakage_check_passed = check_data_leakage(cv_splits)
    check_label_distribution(cv_splits, labels_dict)

    if not leakage_check_passed:
        raise RuntimeError("Data leakage detected. Training aborted.")

    # Training
    fold_results, predictions, true_labels, model_paths, summary_stats_optimal = \
        run_k_fold_cv(cv_splits, labels_dict, args, device)

    # Summary 출력
    print(f"\n{'='*80}")
    print(f"Summary Statistics - FULL WSI TEST (Threshold=0.5)")
    print(f"Model Type: {args.model_type}")
    print(f"{'='*80}\n")
    print_summary_statistics(summary_stats_optimal)
    print_confusion_matrix_summary([{**r, 'test_metrics': r['test_metrics_optimal']} for r in fold_results])

    # Results 저장
    print(f"\n{'='*80}")
    print(f"Saving Results")
    print(f"{'='*80}")

    cv_summary_optimal = {
        "task": "TERT Mutation Prediction (Wild vs Mutant)",
        "model_type": getattr(args, 'model_type', 'abmil'),
        "threshold": "fixed (0.5)",
        "fixed_threshold": 0.5,
        "test_mode": "FULL_WSI (all patches)",
        "summary_statistics": summary_stats_optimal,
        "folds": []
    }

    for fold_result in fold_results:
        fold_summary = {
            "fold": fold_result['fold'],
            "train_size": fold_result['train_size'],
            "val_size": fold_result['val_size'],
            "test_size": fold_result['test_size'],
            "best_epoch": fold_result['best_epoch'],
            "threshold": fold_result['optimal_threshold'],
            "best_train_metrics": fold_result['best_train_metrics'],
            "best_val_metrics": fold_result['best_val_metrics'],
            "test_metrics": fold_result['test_metrics_optimal'],
            "history": fold_result['history'],
            "test_fpr": fold_result['test_fpr'],
            "test_tpr": fold_result['test_tpr'],
            "test_precision": fold_result['test_precision'],
            "test_recall": fold_result['test_recall']
        }
        cv_summary_optimal["folds"].append(fold_summary)

    # Create output directory
    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    cv_summary_optimal_path = Path(args.model_save_dir) / "results_cv_summary_optimal.json"
    with open(cv_summary_optimal_path, "w") as f:
        json.dump(convert_numpy(cv_summary_optimal), f, indent=2)

    print(f"[+] Optimal CV Summary saved: {cv_summary_optimal_path}")

    # Attention Scores 저장
    attention_dir = Path(args.model_save_dir) / "attention_scores"
    attention_dir.mkdir(parents=True, exist_ok=True)

    for fold_result in fold_results:
        fold_num = fold_result['fold']
        attention_data = {
            "fold": fold_num,
            "model_type": getattr(args, 'model_type', 'abmil'),
            "attention_scores": fold_result['full_attention_scores']
        }
        attention_path = attention_dir / f"attention_scores_fold{fold_num}.json"
        with open(attention_path, "w") as f:
            json.dump(convert_numpy(attention_data), f, indent=2)
        print(f"[+] Fold {fold_num} attention scores saved: {attention_path}")

    print(f"\n{'='*80}")
    print(f"Summary of Saved Files:")
    print(f"  - 1 FULL WSI test summary (threshold=0.5)")
    print(f"  - {len(fold_results)} attention score files")
    print(f"{'='*80}")
    print(f"\n[*] Final Results: 'results_cv_summary_optimal.json'")
    print(f"   Test Mode: FULL WSI (all patches)")
    print(f"   Fixed threshold: 0.5")
    print(f"{'='*80}")

    # Visualization
    if args.generate_plots:
        viz_dir = Path(args.model_save_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Generating Visualization Plots")
        print(f"{'='*80}\n")

        generate_all_plots(fold_results, viz_dir)
        print(f"Plots generated using FULL WSI results")

        # Attention Heatmap Generation
        print(f"\n{'='*80}")
        print(f"Generating Attention Heatmaps")
        print(f"{'='*80}\n")

        try:
            # Extract embedding base dir from cv_split_file paths
            embedding_base_dir = None
            with open(args.cv_split_file, 'r') as f:
                cv_data = json.load(f)
            sample_path = cv_data['folds'][0]['train_wsis'][0]
            # Path: .../embedding/{class}/npy/{sample}.npy -> .../embedding
            parts = Path(sample_path).parts
            npy_idx = parts.index('npy')
            embedding_base_dir = str(Path(*parts[:npy_idx-1]))

            # SVS 경로: embedding base dir의 상위에서 thyroid/ 디렉토리
            svs_base_dir = str(Path(embedding_base_dir).parent / "thyroid")

            generate_attention_heatmaps_from_results(
                results_dir=args.model_save_dir,
                embedding_base_dir=embedding_base_dir,
                save_dir=str(Path(args.model_save_dir) / "heatmaps"),
                fold_num="best",
                n_per_class=5,
                interpolation="gaussian",
                dpi=200,
                svs_base_dir=svs_base_dir,
                thumbnail_max_side=2048,
            )
        except Exception as e:
            print(f"[!] Attention heatmap generation failed: {e}")
            import traceback
            traceback.print_exc()

    if args.save_model and model_paths:
        print(f"[+] Saved {len(model_paths)} model checkpoints")

    print(f"\n[+] Training completed!")
    print(f"   Best results saved in: results_cv_summary_optimal.json")
    print(f"   Test Mode: FULL WSI (all patches)")

    return {
        'model_save_dir': args.model_save_dir,
        'json_path': str(cv_summary_optimal_path),
        'cv_summary_path': str(cv_summary_optimal_path),
        'model_checkpoint_path': str(model_paths[0]) if model_paths else None,
        'args': args
    }
