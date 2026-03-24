#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
register_model.py

학습 완료 후 checkpoints/ 디렉토리에서 best .pt를 선택해
MLflow Registered Models (thyr-tert)에 등록하는 독립 스크립트.

실행 방법 (프로젝트 루트에서):

    # [1] checkpoints/ 에서 AUC 기준 자동 선택
    python src/training/register_model.py \
        --model_save_dir outputs/thyroid_tert_model_v0.2.2

    # [2] TorchScript fp16도 함께 등록
    python src/training/register_model.py \
        --model_save_dir outputs/thyroid_tert_model_v0.2.2 \
        --with_torchscript

    # [3] TorchScript fp16 경로 직접 지정
    python src/training/register_model.py \
        --model_save_dir outputs/thyroid_tert_model_v0.2.2 \
        --torchscript_path exports/abmil_torchscript/thyroid_tert_abmil.pt

    # [4] .pt 직접 지정 (원하는 fold 고를 때)
    python src/training/register_model.py \
        --model_save_dir outputs/thyroid_tert_model_v0.2.2 \
        --checkpoint_path outputs/thyroid_tert_model_v0.2.2/checkpoints/best_model_fold2_auc1.0000.pt

    # [5] 모델 타입 명시 (checkpoint에서 자동 감지 안 될 때)
    python src/training/register_model.py \
        --model_save_dir outputs/thyroid_tert_model_v0.2.2 \
        --model_type abmil
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import mlflow
from mlflow.tracking import MlflowClient

# src/ 경로 추가 (export_torchscript import용)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Set MLFLOW_TRACKING_INSECURE_TLS=true in your environment if using self-signed certificates

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "thyr-tert"


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


def _get_latest_version(client: MlflowClient, model_name: str) -> Optional[int]:
    """현재 등록된 최신 버전 번호 반환. 없으면 None."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        return max(int(v.version) for v in versions)
    except Exception:
        return None


def _move_aliases(client: MlflowClient, model_name: str, from_version: int, to_version: int) -> None:
    """production/staging alias를 from_version → to_version으로 이동."""
    for alias in ("production", "staging"):
        try:
            current = client.get_model_version_by_alias(model_name, alias)
            if int(current.version) == from_version:
                client.set_registered_model_alias(model_name, alias, str(to_version))
                print(f"[+] alias @{alias}: v{from_version} → v{to_version}")
        except Exception:
            # alias가 없거나 from_version과 다를 경우 그냥 새 버전에 설정
            client.set_registered_model_alias(model_name, alias, str(to_version))
            print(f"[+] alias @{alias}: → v{to_version}")


def register(checkpoint_path: Path, run_id: str, model_type: str,
             bag_size, lr, seed, test_auc_mean, test_acc_mean, test_f1_mean,
             best_auc: float, model_hparams: dict,
             torchscript_fp16_path: Optional[Path] = None):
    """MLflow Registered Models에 best .pt (+ 선택적 TorchScript fp16) 등록."""
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

    # 등록 전 현재 최신 버전 기록 (alias 이동용)
    prev_version = _get_latest_version(client, model_name)

    source = f"runs:/{run_id}/model/{pt_name}"
    mv = client.create_model_version(
        name=model_name, source=source, run_id=run_id,
        description=f"Best fold AUC: {best_auc:.4f} | Test AUC(mean): {test_auc_mean}"
    )
    new_version = int(mv.version)

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

    # TorchScript fp16 artifact 업로드 (같은 run에)
    if torchscript_fp16_path and torchscript_fp16_path.exists():
        mlflow.log_artifact(str(torchscript_fp16_path), artifact_path="model_torchscript_fp16")
        client.set_model_version_tag(model_name, mv.version, "torchscript_fp16", torchscript_fp16_path.name)
        client.set_model_version_tag(model_name, mv.version, "formats", "pytorch_checkpoint, torchscript_fp16")
        print(f"[+] TorchScript fp16 uploaded: {torchscript_fp16_path.name}")
    else:
        client.set_model_version_tag(model_name, mv.version, "formats", "pytorch_checkpoint")

    # alias 이동: prev_version → new_version
    if prev_version is not None and prev_version != new_version:
        print(f"[*] Moving aliases from v{prev_version} → v{new_version}")
        _move_aliases(client, model_name, prev_version, new_version)
    else:
        client.set_registered_model_alias(model_name, "production", str(new_version))
        client.set_registered_model_alias(model_name, "staging", str(new_version))
        print(f"[+] alias @production, @staging → v{new_version}")

    print(f"[+] Registered: {model_name} version {new_version}")
    return new_version


def main():
    parser = argparse.ArgumentParser(description="Register best TERT checkpoint to MLflow Model Registry")
    parser.add_argument("--model_save_dir", type=str, required=True,
                        help="학습 결과 디렉토리 (outputs/thyroid_tert_model_vX.X.X)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="직접 지정할 .pt 경로 (생략 시 checkpoints/에서 AUC 기준 자동 선택)")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["abmil", "transmil", "acmil", "dtfd", "mhim", "clam"],
                        help="모델 타입 (생략 시 checkpoint에서 자동 감지)")
    parser.add_argument("--with_torchscript", action="store_true",
                        help="TorchScript fp16을 생성하여 함께 등록")
    parser.add_argument("--torchscript_path", type=str, default=None,
                        help="이미 생성된 TorchScript fp16 .pt 경로 (생략 시 --with_torchscript가 자동 생성)")
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

    # --- TorchScript fp16 준비 ---
    torchscript_fp16_path: Optional[Path] = None

    if args.torchscript_path:
        torchscript_fp16_path = Path(args.torchscript_path)
        if not torchscript_fp16_path.exists():
            print(f"[!] TorchScript path not found: {torchscript_fp16_path}")
            sys.exit(1)
        print(f"[+] Using existing TorchScript fp16: {torchscript_fp16_path.name}")

    elif args.with_torchscript:
        from inference.export_torchscript import ABMILTERTForExport, load_weights, _export_one
        print(f"\n[*] Generating TorchScript fp16 from checkpoint ...")
        ts_dir = ROOT / "exports" / "abmil_torchscript"
        ts_dir.mkdir(parents=True, exist_ok=True)
        ts_stem = best_path.stem  # e.g. best_model_fold2_auc1.0000
        ts_out = ts_dir / f"{ts_stem}_fp16.pt"

        hp = model_hparams
        export_model = ABMILTERTForExport(
            in_dim=int(hp.get("in_dim", 1536)),
            embed_dim=int(hp.get("embed_dim", 768)),
            attn_dim=int(hp.get("attn_dim", 512)),
            num_fc_layers=int(hp.get("num_fc_layers", 2)),
            dropout=float(hp.get("dropout", 0.25)),
            num_classes=int(hp.get("num_classes", 2)),
        )
        load_weights(export_model, str(best_path))
        export_model = export_model.half().eval()
        _export_one(export_model, str(ts_out), num_patches=2000, fp16=True)
        torchscript_fp16_path = ts_out

    # --- MLflow run에 best .pt 업로드 후 등록 ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("thyroid_tert")

    version_tag = save_dir.name.replace("thyroid_tert_model_", "").replace("thyroid_tert_", "")
    run_name = f"tert_register_{version_tag}"

    with mlflow.start_run(run_name=run_name):
        # Registry source로 등록할 파일 결정:
        # --torchscript_path 있으면 TorchScript fp16을 source로, 없으면 일반 checkpoint
        if torchscript_fp16_path and torchscript_fp16_path.exists():
            registry_path = torchscript_fp16_path
        else:
            registry_path = best_path

        mlflow.log_artifact(str(registry_path), artifact_path="model")
        run_id = mlflow.active_run().info.run_id

        # 일반 checkpoint도 함께 artifact로 보관 (torchscript 등록 시)
        if registry_path != best_path:
            mlflow.log_artifact(str(best_path), artifact_path="model_checkpoint")

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
            checkpoint_path=registry_path,
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
            torchscript_fp16_path=torchscript_fp16_path if registry_path != torchscript_fp16_path else None,
        )

    print(f"\n[+] Done. thyr-tert version {version} registered.")


if __name__ == "__main__":
    main()
