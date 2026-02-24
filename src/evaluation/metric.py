"""
Thyroid TERT Mutation Evaluation Metrics

Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
- Wild (no mutation) = 0
- Mutant (C228T or C250T) = 1

Metrics computation and visualization for thyroid_tert_v0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import torch

matplotlib.use('Agg')


# =========================
# Metrics Computation
# =========================
def compute_metrics_with_confusion(y_true, y_pred, y_prob):
    """
    혼동행렬 기반 종합 지표 계산 (Thyroid TERT)

    Positive class: Mutant (C228T/C250T)
    Negative class: Wild
    """
    # Force 2x2 order even if one class is absent in predictions.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc  = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "auc": auc_score,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "precision": ppv,      # precision alias for ppv
        "npv": npv,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }



def evaluate_model_with_attention(model, dataloader, device):
    """
    모델 평가 + attention scores 추출
    """
    model.eval()
    all_probs, all_labels, all_preds, all_filenames = [], [], [], []
    attention_scores_dict = {}

    with torch.no_grad():
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)

            # Attention weights 추출
            results_dict = model(
                h=features,
                loss_fn=None,
                label=None,
                return_attention=True,
                return_extra=True
            )

            logits = results_dict['logits']
            attention_weights = results_dict['attention']
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_filenames.extend(filename)

            # Attention scores 저장
            wsi_name = filename[0] if isinstance(filename, (list, tuple)) else filename
            wsi_name = wsi_name.replace('.pt', '').replace('.npy', '')

            # Attention weights 처리
            if attention_weights is None:
                continue

            if isinstance(attention_weights, dict):
                if 'A' not in attention_weights or attention_weights['A'] is None:
                    continue
                attn_scores = attention_weights['A'].cpu().numpy().flatten()
            elif isinstance(attention_weights, torch.Tensor):
                attn_scores = attention_weights.cpu().numpy().flatten()
            else:
                continue

            attention_scores_dict[wsi_name] = {
                'scores': attn_scores.tolist(),
                'n_patches': len(attn_scores),
                'true_label': int(label.cpu().numpy()[0]),
                'predicted_label': int(preds[-1]),
                'pred_prob': float(probs[-1].cpu().numpy())
            }

    return all_probs, all_labels, all_preds, all_filenames, attention_scores_dict


def evaluate_model(model, dataloader, device):
    """
    기본 모델 평가 (attention scores 제외)
    """
    all_probs, all_labels, all_preds, _, _ = evaluate_model_with_attention(model, dataloader, device)
    return all_probs, all_labels, all_preds


def compute_roc_curve_data(y_true, y_prob):
    """ROC curve 데이터 계산"""
    return roc_curve(y_true, y_prob)


def compute_precision_recall_curve_data(y_true, y_prob):
    """Precision-Recall curve 데이터 계산"""
    return precision_recall_curve(y_true, y_prob)


def get_test_metrics(fold_result):
    """Return test metrics regardless of the stored key name."""
    if 'test_metrics' in fold_result:
        return fold_result['test_metrics']
    if 'test_metrics_optimal' in fold_result:
        return fold_result['test_metrics_optimal']
    raise KeyError("Fold result missing 'test_metrics'/'test_metrics_optimal'")


def compute_summary_statistics(fold_results):
    """
    전체 fold 결과의 요약 통계 계산
    """
    metrics_order = ['accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    summary_stats = {}

    for set_name, metric_key in [('train', 'best_train_metrics'),
                                   ('val', 'best_val_metrics'),
                                   ('test', 'test_metrics')]:
        summary_stats[set_name] = {}
        for metric in metrics_order:
            if set_name == 'test':
                values = [get_test_metrics(r)[metric] for r in fold_results]
            else:
                values = [r[metric_key][metric] for r in fold_results]
            summary_stats[set_name][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }

    return summary_stats


# =========================
# Printing Functions
# =========================
def print_fold_table(fold_num, train_metrics, val_metrics, test_metrics):
    """각 fold별 결과를 표 형태로 출력"""
    print(f"\n{'='*80}")
    print(f"Fold {fold_num} Results")
    print(f"{'='*80}")

    print(f"| {'Set':8s} | {'Accuracy':8s} | {'AUC':8s} | {'Sensitivity':11s} | "
          f"{'Specificity':11s} | {'Precision':9s} | {'NPV':8s} | {'F1-score':8s} |")
    print("|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*13 + "|" +
          "-"*13 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|")

    print(f"| {'Train':8s} | {train_metrics['accuracy']:8.2f} | "
          f"{train_metrics['auc']:8.2f} | {train_metrics['sensitivity']:11.2f} | "
          f"{train_metrics['specificity']:11.2f} | {train_metrics['ppv']:9.2f} | "
          f"{train_metrics['npv']:8.2f} | {train_metrics['f1']:8.2f} |")

    print(f"| {'Val':8s} | {val_metrics['accuracy']:8.2f} | "
          f"{val_metrics['auc']:8.2f} | {val_metrics['sensitivity']:11.2f} | "
          f"{val_metrics['specificity']:11.2f} | {val_metrics['ppv']:9.2f} | "
          f"{val_metrics['npv']:8.2f} | {val_metrics['f1']:8.2f} |")

    print(f"| {'Test':8s} | {test_metrics['accuracy']:8.2f} | "
          f"{test_metrics['auc']:8.2f} | {test_metrics['sensitivity']:11.2f} | "
          f"{test_metrics['specificity']:11.2f} | {test_metrics['ppv']:9.2f} | "
          f"{test_metrics['npv']:8.2f} | {test_metrics['f1']:8.2f} |")


def print_summary_statistics(summary_stats):
    """전체 fold의 요약 통계 출력"""
    print(f"\n\n{'='*80}")
    print(f"WSI Instance Results (Mean +/- Std Across All Folds)")
    print(f"{'='*80}\n")

    metrics_order = ['accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'auc': 'AUC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'ppv': 'PPV',
        'npv': 'NPV',
        'f1': 'F1-score'
    }

    # 헤더
    header = f"| {'Set':8s} |"
    for metric in metrics_order:
        header += f" {metric_names[metric]:17s} |"
    print(header)
    print("|" + "-"*10 + "|" + "-"*19*len(metrics_order) + "|")

    # Train/Val/Test 결과
    for set_name, set_label in [('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
        row = f"| **{set_label}** |"
        for metric in metrics_order:
            mean_val = summary_stats[set_name][metric]['mean']
            std_val = summary_stats[set_name][metric]['std']
            row += f" **{mean_val:.3f} +/- {std_val:.3f}** |"
        print(row)


def print_confusion_matrix_summary(fold_results):
    """전체 fold의 confusion matrix 합계 출력"""
    print(f"\n{'-'*80}")
    print(f"Total Confusion Matrix (Test):")
    print(f"{'-'*80}")

    total_tp = sum(get_test_metrics(r)['tp'] for r in fold_results)
    total_tn = sum(get_test_metrics(r)['tn'] for r in fold_results)
    total_fp = sum(get_test_metrics(r)['fp'] for r in fold_results)
    total_fn = sum(get_test_metrics(r)['fn'] for r in fold_results)

    print(f"    TP: {total_tp:4d}  |  FN: {total_fn:4d}")
    print(f"    FP: {total_fp:4d}  |  TN: {total_tn:4d}")


# =========================
# Visualization Functions
# =========================
def plot_roc_curves(fold_results, save_dir):
    """각 fold별 ROC curve와 평균 ROC curve 그리기"""
    plt.figure(figsize=(10, 8))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_result in fold_results:
        fold_num = fold_result['fold']
        fpr = fold_result['test_fpr']
        tpr = fold_result['test_tpr']
        roc_auc = get_test_metrics(fold_result)['auc']

        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold_num} (AUC = {roc_auc:.3f})')

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
             label=f'Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                     label='+/- 1 std. dev.')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - TERT Mutation Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)

    save_path = Path(save_dir) / 'roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] ROC curves saved: {save_path}")


def plot_precision_recall_curves(fold_results, save_dir):
    """각 fold별 Precision-Recall curve 그리기"""
    plt.figure(figsize=(10, 8))

    for fold_result in fold_results:
        fold_num = fold_result['fold']
        precision = fold_result['test_precision']
        recall = fold_result['test_recall']
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, alpha=0.5, linewidth=2,
                label=f'Fold {fold_num} (AUC = {pr_auc:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - TERT Mutation Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    save_path = Path(save_dir) / 'precision_recall_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Precision-Recall curves saved: {save_path}")


def plot_training_curves(fold_results, save_dir):
    """각 fold별 학습 곡선 (Loss, AUC, Accuracy) 그리기"""
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds, 3, figsize=(18, 4*n_folds))

    if n_folds == 1:
        axes = axes.reshape(1, -1)

    for idx, fold_result in enumerate(fold_results):
        fold_num = fold_result['fold']
        history = fold_result['history']
        best_epoch = fold_result['best_epoch']

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[idx, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[idx, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[idx, 0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
        axes[idx, 0].set_xlabel('Epoch', fontsize=10)
        axes[idx, 0].set_ylabel('Loss', fontsize=10)
        axes[idx, 0].set_title(f'Fold {fold_num} - Loss', fontsize=11, fontweight='bold')
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(alpha=0.3)

        # AUC
        axes[idx, 1].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
        axes[idx, 1].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
        test_auc = get_test_metrics(fold_result)['auc']
        axes[idx, 1].scatter([best_epoch], [test_auc],
                           color='green', s=100, marker='*', label=f'Test AUC ({test_auc:.3f})', zorder=5)
        axes[idx, 1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 1].set_xlabel('Epoch', fontsize=10)
        axes[idx, 1].set_ylabel('AUC', fontsize=10)
        axes[idx, 1].set_title(f'Fold {fold_num} - AUC', fontsize=11, fontweight='bold')
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(alpha=0.3)
        axes[idx, 1].set_ylim([0, 1.05])

        # Accuracy
        axes[idx, 2].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[idx, 2].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        test_acc = get_test_metrics(fold_result)['accuracy']
        axes[idx, 2].scatter([best_epoch], [test_acc],
                           color='green', s=100, marker='*', label=f'Test Acc ({test_acc:.3f})', zorder=5)
        axes[idx, 2].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 2].set_xlabel('Epoch', fontsize=10)
        axes[idx, 2].set_ylabel('Accuracy', fontsize=10)
        axes[idx, 2].set_title(f'Fold {fold_num} - Accuracy', fontsize=11, fontweight='bold')
        axes[idx, 2].legend(fontsize=8)
        axes[idx, 2].grid(alpha=0.3)
        axes[idx, 2].set_ylim([0, 1.05])

    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Training curves saved: {save_path}")


def plot_metric_comparison(fold_results, save_dir):
    """각 fold별 최종 train/val/test metrics 비교 bar plot"""
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # 안전한 metric 접근 함수
    def safe_get(m, key):
        """precision / ppv 키 혼용 지원"""
        if key in m:
            return m[key]
        elif key == 'ppv' and 'precision' in m:
            return m['precision']
        elif key == 'precision' and 'ppv' in m:
            return m['ppv']
        else:
            return 0.0

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        fold_nums = [r['fold'] for r in fold_results]

        train_values = [safe_get(r['best_train_metrics'], metric) for r in fold_results]
        val_values   = [safe_get(r['best_val_metrics'], metric) for r in fold_results]
        test_values  = [safe_get(get_test_metrics(r), metric) for r in fold_results]

        x = np.arange(len(fold_nums))
        width = 0.25

        axes[idx].bar(x - width, train_values, width, label='Train', color='skyblue', alpha=0.8)
        axes[idx].bar(x, val_values, width, label='Val', color='orange', alpha=0.8)
        bars3 = axes[idx].bar(x + width, test_values, width, label='Test', color='lightgreen', alpha=0.8)

        axes[idx].axhline(y=np.mean(train_values), color='blue', linestyle='--', alpha=0.5, linewidth=1)
        axes[idx].axhline(y=np.mean(val_values), color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[idx].axhline(y=np.mean(test_values), color='green', linestyle='--', alpha=0.5, linewidth=1,
                         label=f'Test Mean: {np.mean(test_values):.3f}')

        axes[idx].set_xlabel('Fold', fontsize=10)
        axes[idx].set_ylabel(name, fontsize=10)
        axes[idx].set_title(f'{name} Comparison', fontsize=11, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(fold_nums)
        axes[idx].set_ylim([0, 1.05])
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3, axis='y')

        for bar, val in zip(bars3, test_values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.delaxes(axes[-1])

    plt.tight_layout()
    save_path = Path(save_dir) / 'metric_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Metric comparison saved: {save_path}")


def plot_confusion_matrices(fold_results, save_dir):
    """각 fold별 confusion matrix 시각화"""
    n_folds = len(fold_results)
    cols = min(3, n_folds)
    rows = (n_folds + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, fold_result in enumerate(fold_results):
        fold_num = fold_result['fold']
        metrics = get_test_metrics(fold_result)

        cm = np.array([[metrics['tn'], metrics['fp']],
                       [metrics['fn'], metrics['tp']]])

        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(f'Fold {fold_num} Confusion Matrix')

        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Pred Wild', 'Pred Mutant'])
        axes[idx].set_yticklabels(['True Wild', 'True Mutant'])

        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black",
                             fontsize=14, fontweight='bold')

    for idx in range(n_folds, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    save_path = Path(save_dir) / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Confusion matrices saved: {save_path}")


def generate_all_plots(fold_results, save_dir):
    """
    모든 시각화 plot을 한번에 생성
    """
    print(f"\n{'='*80}")
    print(f"Generating Visualization Plots")
    print(f"{'='*80}\n")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(fold_results, save_dir)
    plot_precision_recall_curves(fold_results, save_dir)
    plot_training_curves(fold_results, save_dir)
    plot_metric_comparison(fold_results, save_dir)
    plot_confusion_matrices(fold_results, save_dir)

    print(f"\n[+] All visualizations saved in: {save_dir}")
