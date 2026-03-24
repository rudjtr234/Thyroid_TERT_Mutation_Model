#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KPI evaluation for TERT inference latency.

- Loads trained .pt checkpoints directly (no TorchScript conversion required)
- Runs batch inference over sampled slides (.npy embeddings)
- Computes p50 / p95 timing metrics
- Optionally uploads KPI metrics to MLflow

Examples:
    # Ensemble (all .pt in directory)
    python src/inference/kpi_eval.py \
        --checkpoint_dir outputs/thyroid_tert_model_v0.2.4/checkpoints \
        --embedding_dir /path/to/dataset/embedding \
        --n_slides 50

    # Single model (best AUC file by filename pattern)
    python src/inference/kpi_eval.py \
        --checkpoint_dir outputs/thyroid_tert_model_v0.2.4/checkpoints \
        --embedding_dir /path/to/dataset/embedding \
        --n_slides 50 \
        --single_model

    # With MLflow upload
    python src/inference/kpi_eval.py \
        --checkpoint_dir outputs/thyroid_tert_model_v0.2.4/checkpoints \
        --embedding_dir /path/to/dataset/embedding \
        --n_slides 50 \
        --upload_mlflow
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _ensure_torch_available() -> None:
    if importlib.util.find_spec("torch") is None:
        print("[!] ModuleNotFoundError: No module named 'torch'")
        print("[!] PyTorch is required to run this KPI script.")
        print(f"[!] Python executable: {sys.executable}")
        print("[!] Example:")
        print(f"    {sys.executable} -m pip install torch torchvision torchaudio")
        sys.exit(1)


_ensure_torch_available()

import numpy as np
import torch
import torch.nn.functional as F

# Add project src/ to import path
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from models.abmil import ABMILTERTConfig, ABMILTERTModel
from models.transmil import TransMILTERTConfig, TransMILTERTModel


DEFAULT_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_MLFLOW_EXPERIMENT = "thyroid_tert"


@dataclass
class LoadedModel:
    path: Path
    model_type: str
    model: torch.nn.Module
    auc: Optional[float]
    in_dim: int


def _parse_auc_from_filename(path: Path) -> Optional[float]:
    m = re.search(r"auc([0-9]+(?:\.[0-9]+)?)", path.name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_model_info_from_ckpt(ckpt: Dict) -> Tuple[str, Dict, Optional[float]]:
    config = ckpt.get("config", {}) or {}
    model_hparams = config.get("model_hparams", {}) or {}
    model_type = (
        config.get("model_type")
        or model_hparams.get("model_type")
        or "abmil"
    )
    model_type = str(model_type).lower()
    auc = ckpt.get("auc")
    try:
        auc = float(auc) if auc is not None else None
    except (TypeError, ValueError):
        auc = None
    return model_type, model_hparams, auc


def _build_model(model_type: str, model_hparams: Dict) -> Tuple[torch.nn.Module, int]:
    in_dim = int(model_hparams.get("in_dim", 1536))

    if model_type == "transmil":
        config = TransMILTERTConfig(
            in_dim=in_dim,
            embed_dim=int(model_hparams.get("embed_dim", 512)),
            num_heads=int(model_hparams.get("num_heads", 8)),
            num_layers=int(model_hparams.get("num_layers", 2)),
            num_landmarks=int(model_hparams.get("num_landmarks", 256)),
            pinv_iterations=int(model_hparams.get("pinv_iterations", 6)),
            dropout=float(model_hparams.get("dropout", 0.1)),
            num_classes=int(model_hparams.get("num_classes", 2)),
            use_layer_norm=bool(model_hparams.get("use_layer_norm", False)),
        )
        return TransMILTERTModel(config), in_dim

    config = ABMILTERTConfig(
        gate=bool(model_hparams.get("gate", True)),
        in_dim=in_dim,
        embed_dim=int(model_hparams.get("embed_dim", 512)),
        attn_dim=int(model_hparams.get("attn_dim", 384)),
        num_fc_layers=int(model_hparams.get("num_fc_layers", 2)),
        dropout=float(model_hparams.get("dropout", 0.25)),
        num_classes=int(model_hparams.get("num_classes", 2)),
        use_layer_norm=bool(model_hparams.get("use_layer_norm", True)),
    )
    return ABMILTERTModel(config), in_dim


def _load_state_dict_with_fallback(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    # Standard checkpoint from this repo: keys start with "model."
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    # Fallback 1: load into wrapped model without prefix
    if hasattr(model, "model"):
        stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        if stripped:
            try:
                model.model.load_state_dict(stripped, strict=True)
                return
            except RuntimeError:
                pass
        try:
            model.model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            pass

    # Fallback 2: add "model." prefix and retry
    prefixed = {f"model.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(prefixed, strict=True)


def load_model_from_checkpoint(pt_path: Path, device: torch.device) -> LoadedModel:
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model_type, model_hparams, auc = _extract_model_info_from_ckpt(ckpt)

    model, in_dim = _build_model(model_type=model_type, model_hparams=model_hparams)

    raw_sd = ckpt.get("model_state_dict", ckpt)
    if not isinstance(raw_sd, dict):
        raise ValueError(f"Invalid state_dict format in checkpoint: {pt_path}")

    _load_state_dict_with_fallback(model, raw_sd)
    model.eval()
    model.to(device)

    return LoadedModel(
        path=pt_path,
        model_type=model_type,
        model=model,
        auc=auc,
        in_dim=in_dim,
    )


def _pick_single_checkpoint(pt_files: Sequence[Path]) -> Path:
    best_path = None
    best_score = float("-inf")
    for path in pt_files:
        score = _parse_auc_from_filename(path)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_path = path
    return best_path or sorted(pt_files)[0]


def _collect_embedding_files(embedding_dir: Path) -> List[Path]:
    # Works for both:
    # 1) flat folder: .../npy/*.npy
    # 2) nested folder: .../embedding/{Wild|C228T|C250T}/npy/*.npy
    return sorted(embedding_dir.rglob("*.npy"))


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_one_slide(models: Sequence[LoadedModel], npy_path: Path, device: torch.device) -> Dict:
    t_total_start = time.perf_counter()

    # Load embeddings
    t_load_start = time.perf_counter()
    embeddings = np.load(str(npy_path))
    t_load = time.perf_counter() - t_load_start

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape={embeddings.shape}")

    expected_dim = models[0].in_dim
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dim mismatch for {npy_path.name}: "
            f"expected {expected_dim}, got {embeddings.shape[1]}"
        )

    # Inference
    _synchronize_if_cuda(device)
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        h = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)  # [1, N, D]
        pos_probs = []
        for loaded in models:
            outputs = loaded.model(
                h=h,
                return_attention=False,
                return_extra=True,
            )
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            probs = F.softmax(logits, dim=-1)[0]
            pos_probs.append(float(probs[1].detach().cpu().item()))
    _synchronize_if_cuda(device)
    t_infer = time.perf_counter() - t_infer_start

    t_total = time.perf_counter() - t_total_start

    return {
        "slide_id": npy_path.stem,
        "source_npy_path": str(npy_path),
        "num_patches": int(embeddings.shape[0]),
        "tert_mutant_prob": float(np.mean(pos_probs)),
        "sec_load": round(float(t_load), 6),
        "sec_infer": round(float(t_infer), 6),
        "sec_total": round(float(t_total), 6),
    }


def _compute_kpi(timings: Sequence[Dict], n_errors: int, n_models: int) -> Dict:
    total_secs = np.array([x["sec_total"] for x in timings], dtype=np.float64)
    infer_secs = np.array([x["sec_infer"] for x in timings], dtype=np.float64)
    load_secs = np.array([x["sec_load"] for x in timings], dtype=np.float64)

    success = len(timings)
    attempts = success + n_errors
    success_rate = (success / attempts) if attempts > 0 else 0.0

    return {
        "n_slides_success": int(success),
        "n_errors": int(n_errors),
        "n_models": int(n_models),
        "success_rate": round(float(success_rate), 6),
        "total_p50_sec": round(float(np.percentile(total_secs, 50)), 6),
        "total_p95_sec": round(float(np.percentile(total_secs, 95)), 6),
        "total_mean_sec": round(float(np.mean(total_secs)), 6),
        "infer_p50_sec": round(float(np.percentile(infer_secs, 50)), 6),
        "infer_p95_sec": round(float(np.percentile(infer_secs, 95)), 6),
        "infer_mean_sec": round(float(np.mean(infer_secs)), 6),
        "load_p50_sec": round(float(np.percentile(load_secs, 50)), 6),
        "load_p95_sec": round(float(np.percentile(load_secs, 95)), 6),
        "load_mean_sec": round(float(np.mean(load_secs)), 6),
    }


def _upload_kpi_to_mlflow(
    result_path: Path,
    kpi: Dict,
    args: argparse.Namespace,
    model_type: str,
    model_paths: Sequence[Path],
    device: torch.device,
) -> None:
    if importlib.util.find_spec("mlflow") is None:
        print("[!] mlflow not installed. Skipping MLflow upload.")
        return

    import mlflow

    # Set MLFLOW_TRACKING_INSECURE_TLS=true in your environment if using self-signed certificates
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    mode = "single" if args.single_model else "ensemble"
    run_name = args.mlflow_run_name or (
        f"tert_kpi_{mode}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "task": "TERT mutation inference KPI",
                "mode": mode,
                "model_type": model_type,
                "n_models": len(model_paths),
                "n_slides_requested": args.n_slides,
                "seed": args.seed,
                "device": device.type,
                "checkpoint_dir": args.checkpoint_dir,
                "embedding_dir": args.embedding_dir,
            }
        )

        for k, v in kpi.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # Keep model list as tag for quick traceability
        mlflow.set_tag(
            "checkpoint_files",
            ",".join(path.name for path in model_paths),
        )

        mlflow.log_artifact(str(result_path), artifact_path="kpi_eval")

    print(
        f"[KPI] MLflow upload completed "
        f"(experiment={args.mlflow_experiment}, run_name={run_name})"
    )


def run_kpi_eval(args: argparse.Namespace) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[KPI] Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")

    pt_files = sorted(checkpoint_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")

    if args.single_model:
        selected = _pick_single_checkpoint(pt_files)
        pt_files = [selected]
        print(f"[KPI] Single-model mode, selected checkpoint: {selected.name}")
    else:
        print(f"[KPI] Ensemble mode, checkpoints: {len(pt_files)}")

    loaded_models: List[LoadedModel] = []
    for path in pt_files:
        loaded = load_model_from_checkpoint(path, device=device)
        loaded_models.append(loaded)
        auc_txt = f"{loaded.auc:.4f}" if loaded.auc is not None else "N/A"
        print(f"  - loaded {path.name} | type={loaded.model_type} | auc={auc_txt}")

    # Safety: mixed model types in one ensemble is likely accidental.
    model_types = {x.model_type for x in loaded_models}
    if len(model_types) > 1:
        raise ValueError(f"Mixed model types detected in checkpoint_dir: {sorted(model_types)}")

    model_type = loaded_models[0].model_type

    npy_files = _collect_embedding_files(embedding_dir)
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {embedding_dir}")

    rng = random.Random(args.seed)
    sample_n = min(args.n_slides, len(npy_files))
    slides = rng.sample(npy_files, sample_n)
    print(f"[KPI] Sampled slides: {sample_n}/{len(npy_files)}")

    timings: List[Dict] = []
    errors: List[Dict] = []

    for idx, npy_path in enumerate(slides, start=1):
        try:
            result = infer_one_slide(loaded_models, npy_path=npy_path, device=device)
            timings.append(result)
            print(
                f"  [{idx:03d}/{sample_n}] {npy_path.stem} "
                f"total={result['sec_total']:.3f}s infer={result['sec_infer']:.3f}s"
            )
        except Exception as exc:
            errors.append({"slide_id": npy_path.stem, "error": str(exc)})
            print(f"  [{idx:03d}/{sample_n}] [!] {npy_path.stem}: {exc}")

    if not timings:
        raise RuntimeError("No successful inference timing records were collected.")

    kpi = _compute_kpi(timings=timings, n_errors=len(errors), n_models=len(loaded_models))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device.type,
        "checkpoint_dir": str(checkpoint_dir),
        "embedding_dir": str(embedding_dir),
        "sampled_slides": sample_n,
        "model_type": model_type,
        "single_model": bool(args.single_model),
        "kpi": kpi,
        "timings": timings,
        "errors": errors,
    }

    result_path = out_dir / "kpi_timing.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[KPI] Results")
    print(f"  Total  p50={kpi['total_p50_sec']:.6f}s  p95={kpi['total_p95_sec']:.6f}s")
    print(f"  Infer  p50={kpi['infer_p50_sec']:.6f}s  p95={kpi['infer_p95_sec']:.6f}s")
    print(f"  Load   p50={kpi['load_p50_sec']:.6f}s  p95={kpi['load_p95_sec']:.6f}s")
    print(f"  Errors {kpi['n_errors']}/{sample_n}")
    print(f"[KPI] Saved: {result_path}")

    if args.upload_mlflow:
        _upload_kpi_to_mlflow(
            result_path=result_path,
            kpi=kpi,
            args=args,
            model_type=model_type,
            model_paths=[x.path for x in loaded_models],
            device=device,
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    default_output_dir = Path(__file__).resolve().parent / "kpi_eval_results"

    parser = argparse.ArgumentParser(
        description="KPI evaluation for TERT inference latency (p50/p95)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing .pt checkpoints",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Embedding root directory (.npy searched recursively)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_dir),
        help="Output directory for KPI JSON",
    )
    parser.add_argument(
        "--n_slides",
        type=int,
        default=50,
        help="Number of slides to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for slide sampling",
    )
    parser.add_argument(
        "--single_model",
        action="store_true",
        help="Use one checkpoint only (best AUC in filename pattern if available)",
    )
    parser.add_argument(
        "--upload_mlflow",
        action="store_true",
        help="Upload KPI metrics/artifact to MLflow",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_TRACKING_URI})",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_MLFLOW_EXPERIMENT})",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default=None,
        help="Optional MLflow run name override",
    )
    return parser


if __name__ == "__main__":
    cli_args = _build_parser().parse_args()
    run_kpi_eval(cli_args)
