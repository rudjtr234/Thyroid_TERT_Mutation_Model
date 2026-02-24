#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
register_model.py

학습 완료 후 checkpoints/ 디렉토리에서 best .pt를 선택해
MLflow Registered Models (thyr-tert)에 등록하는 독립 스크립트.

실행 방법 (src/training/ 에서):

    # [1] checkpoints/ 에서 AUC 기준 자동 선택
    cd /path/to/thyroid_tert_project/src/training
    python register_model.py \
        --model_save_dir /path/to/thyroid_tert_project/outputs/thyroid_tert_model_v0.2.4

    # [2] .pt 직접 지정 (원하는 fold 고를 때)
    cd /path/to/thyroid_tert_project/src/training
    python register_model.py \
        --model_save_dir /path/to/thyroid_tert_project/outputs/thyroid_tert_model_v0.2.4 \
        --checkpoint_path /path/to/thyroid_tert_project/outputs/thyroid_tert_model_v0.2.4/checkpoints/best_model_fold2_auc0.9949.pt

    # [3] 모델 타입 명시 (checkpoint에서 자동 감지 안 될 때)
    cd /path/to/thyroid_tert_project/src/training
    python register_model.py \
        --model_save_dir /path/to/thyroid_tert_project/outputs/thyroid_tert_model_v0.2.4 \
        --model_type abmil
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "thyroid_tert")
MLFLOW_TRACKING_INSECURE_TLS = os.getenv("MLFLOW_TRACKING_INSECURE_TLS")
if MLFLOW_TRACKING_INSECURE_TLS is not None:
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = MLFLOW_TRACKING_INSECURE_TLS

REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "thyr-tert")


def find_best_checkpoint(ckpt_dir: Path) -> tuple:
    """checkpoints/ 에서 AUC 기준 best .pt 반환. (path, auc)"""
    pt_files = sorted(ckpt_dir.glob("*.pt"))
    if not pt_files:
        return None, -1.0

    best_path = None
    best_auc = -1.0
    for pt in pt_files:
        try:
            ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
            auc = float(ckpt.get("auc", -1.0))
            print(f"  {pt.name}  AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_path = pt
        except Exception as e:
            print(f"  [!] Failed to load {pt.name}: {e}")

    return best_path, best_auc


def register(checkpoint_path: Path, run_id: str, model_type: str,
             bag_size, lr, seed, test_auc_mean, test_acc_mean, test_f1_mean,
             best_auc: float, model_hparams: dict):
    """MLflow Registered Models에 best .pt 등록."""
    client = MlflowClient()
    model_name = REGISTERED_MODEL_NAME
    pt_name = checkpoint_path.name

    model_description = (
        f"TERT Mutation 5-Fold CV\n"
        f"Backbone: {model_type.upper()} + UNI2-H (1536-dim)\n"
        f"Best fold AUC: {best_auc:.4f} | Test AUC(mean): {test_auc_mean}, "
        f"Acc: {test_acc_mean}, F1: {test_f1_mean}\n"
        f"bag_size={bag_size}, lr={lr}, seed={seed}"
    )

    # Registered Model 없으면 생성
    try:
        client.get_registered_model(model_name)
        client.update_registered_model(model_name, description=model_description)
    except Exception:
        client.create_registered_model(model_name, description=model_description)

    source = f"runs:/{run_id}/model/{pt_name}"
    mv = client.create_model_version(
        name=model_name, source=source, run_id=run_id,
        description=f"Best fold AUC: {best_auc:.4f} | Test AUC(mean): {test_auc_mean}"
    )

    # 버전 태그
    client.set_model_version_tag(model_name, mv.version, "embedding", "UNI2-H (1536-dim)")
    client.set_model_version_tag(model_name, mv.version, "best_fold_auc", str(round(best_auc, 4)))
    client.set_model_version_tag(model_name, mv.version, "test_auc_mean", str(test_auc_mean))
    client.set_model_version_tag(model_name, mv.version, "test_acc_mean", str(test_acc_mean))
    client.set_model_version_tag(model_name, mv.version, "test_f1_mean", str(test_f1_mean))
    client.set_model_version_tag(model_name, mv.version, "bag_size", str(bag_size))
    client.set_model_version_tag(model_name, mv.version, "model_arch", f"{model_type.upper()}_Gated")
    if model_hparams:
        client.set_model_version_tag(model_name, mv.version, "embed_dim", str(model_hparams.get("embed_dim", "")))
        client.set_model_version_tag(model_name, mv.version, "attn_dim", str(model_hparams.get("attn_dim", "")))

    # alias
    client.set_registered_model_alias(model_name, "production", mv.version)
    client.set_registered_model_alias(model_name, "staging", mv.version)

    print(f"[+] Registered: {model_name} version {mv.version} (production/staging alias 설정 완료)")
    return mv.version


def main():
    parser = argparse.ArgumentParser(description="Register best TERT checkpoint to MLflow Model Registry")
    parser.add_argument("--model_save_dir", type=str, required=True,
                        help="학습 결과 디렉토리 (outputs/thyroid_tert_model_vX.X.X)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="직접 지정할 .pt 경로 (생략 시 checkpoints/에서 AUC 기준 자동 선택)")
    parser.add_argument("--model_type", type=str, default=None, choices=["abmil", "transmil"],
                        help="모델 타입 (생략 시 checkpoint에서 자동 감지)")
    args = parser.parse_args()

    save_dir = Path(args.model_save_dir)
    if not save_dir.exists():
        print(f"[!] Directory not found: {save_dir}")
        sys.exit(1)

    # --- Best checkpoint 결정 ---
    if args.checkpoint_path:
        best_path = Path(args.checkpoint_path)
        if not best_path.exists():
            print(f"[!] Checkpoint not found: {best_path}")
            sys.exit(1)
        ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
        best_auc = float(ckpt.get("auc", -1.0))
        print(f"[+] Using specified checkpoint: {best_path.name}  AUC={best_auc:.4f}")
    else:
        ckpt_dir = save_dir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"[!] checkpoints/ not found in {save_dir}")
            sys.exit(1)
        print(f"[*] Scanning checkpoints in {ckpt_dir} ...")
        best_path, best_auc = find_best_checkpoint(ckpt_dir)
        if best_path is None:
            print("[!] No .pt files found")
            sys.exit(1)
        print(f"[+] Best checkpoint: {best_path.name}  AUC={best_auc:.4f}")
        ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)

    # --- config 추출 ---
    ckpt_cfg = ckpt.get("config", {}) or {}
    model_hparams = ckpt_cfg.get("model_hparams", {}) or {}
    model_type = args.model_type or ckpt_cfg.get("model_type") or model_hparams.get("model_type", "abmil")
    lr = ckpt_cfg.get("lr", "N/A")
    bag_size = ckpt_cfg.get("bag_size") or model_hparams.get("bag_size", "N/A")
    seed = ckpt_cfg.get("seed", "N/A")

    # --- JSON에서 5-fold 평균 메트릭 추출 ---
    json_candidates = list(save_dir.glob("*results*.json"))
    test_auc_mean = test_acc_mean = test_f1_mean = "N/A"
    if json_candidates:
        import json
        with open(json_candidates[0], "r") as f:
            summary = json.load(f)
        folds = summary.get("folds", [])

        def _get_metric(fold, split, key):
            for k in (f"{split}_metrics", f"best_{split}_metrics"):
                if k in fold:
                    return fold[k].get(key)
            return None

        aucs = [_get_metric(fd, "test", "auc") for fd in folds]
        accs = [_get_metric(fd, "test", "accuracy") for fd in folds]
        f1s  = [_get_metric(fd, "test", "f1") for fd in folds]
        import numpy as np
        aucs = [v for v in aucs if v is not None]
        accs = [v for v in accs if v is not None]
        f1s  = [v for v in f1s  if v is not None]
        if aucs: test_auc_mean = round(float(np.mean(aucs)), 4)
        if accs: test_acc_mean = round(float(np.mean(accs)), 4)
        if f1s:  test_f1_mean  = round(float(np.mean(f1s)),  4)

    print(f"\n{'='*60}")
    print(f"  Model type : {model_type}")
    print(f"  Best AUC   : {best_auc:.4f}  ({best_path.name})")
    print(f"  Test AUC   : {test_auc_mean} (mean), Acc: {test_acc_mean}, F1: {test_f1_mean}")
    print(f"  bag_size={bag_size}, lr={lr}, seed={seed}")
    print(f"{'='*60}\n")
    # --- MLflow run에 best .pt 업로드 후 등록 ---
    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("Set MLFLOW_TRACKING_URI before running register_model.py")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    version_tag = save_dir.name.replace("thyroid_tert_model_", "").replace("thyroid_tert_", "")
    run_name = f"tert_register_{version_tag}"

    with mlflow.start_run(run_name=run_name):
        # best .pt → artifacts/model/
        mlflow.log_artifact(str(best_path), artifact_path="model")
        run_id = mlflow.active_run().info.run_id

        # 파라미터 기록
        mlflow.log_params({
            "version": version_tag,
            "model_type": model_type,
            "bag_size": bag_size,
            "lr": lr,
            "seed": seed,
            "best_fold_auc": round(best_auc, 4),
        })
        mlflow.log_metrics({
            "test_auc_mean": float(test_auc_mean) if test_auc_mean != "N/A" else 0.0,
            "test_acc_mean": float(test_acc_mean) if test_acc_mean != "N/A" else 0.0,
            "test_f1_mean":  float(test_f1_mean)  if test_f1_mean  != "N/A" else 0.0,
        })

        version = register(
            checkpoint_path=best_path,
            run_id=run_id,
            model_type=model_type,
            bag_size=bag_size,
            lr=lr,
            seed=seed,
            test_auc_mean=test_auc_mean,
            test_acc_mean=test_acc_mean,
            test_f1_mean=test_f1_mean,
            best_auc=best_auc,
            model_hparams=model_hparams,
        )

    print(f"\n[+] Done. thyr-tert version {version} registered.")


if __name__ == "__main__":
    main()
