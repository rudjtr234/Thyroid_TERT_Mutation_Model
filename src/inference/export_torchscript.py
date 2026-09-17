#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_torchscript.py

TERT MIL 모델을 TorchScript로 변환하는 스크립트.
register_model.py --with_torchscript 에서 호출되거나 단독 실행 가능.

구조:
  Input  [batch, num_patches, in_dim]
  Output (logits [batch, 2], attention [batch, num_patches])

실행 방법:

    cd .

    # [1] checkpoint에서 직접 export
    python src/inference/export_torchscript.py \
        --checkpoint outputs/thyroid_tert_model_v0.8.5/checkpoints/best_model_fold2_auc1.0000.pt

    # [2] MLflow Registry의 현재 production 모델에서 export
    python src/inference/export_torchscript.py \
        --model_uri models:/thyr-tert@production
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

MLFLOW_URI = "http://localhost:5000"
REGISTERED_MODEL_NAME = "thyr-tert"


# =============================================================================
# Export 전용 래퍼
# =============================================================================

class MILTERTForExport(nn.Module):
    """
    TorchScript 변환용 MIL 래퍼.

    실제 모델의 `return_attention=True` 경로를 고정 사용하고,
    TorchScript 출력은 `(logits, attention)` 2-tensor tuple로 정규화한다.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    @staticmethod
    def _canonicalize_attention(attention: Optional[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        if attention is None:
            return h.new_empty((h.shape[0], 0))
        if attention.dim() == 3 and attention.shape[1] == 1:
            return attention[:, 0, :]
        if attention.dim() == 2:
            return attention
        if attention.dim() == 1:
            return attention.unsqueeze(0)
        return attention.reshape(attention.shape[0], -1)

    def forward(self, h: torch.Tensor):
        slide_feat, log_dict = self.base_model.model.forward_features(
            h=h,
            attn_mask=None,
            return_attention=True,
        )
        logits = self.base_model.model.forward_head(slide_feat)
        attention = self._canonicalize_attention(log_dict.get("attention"), h)
        return logits, attention


# Backward compatibility for older imports.
ABMILTERTForExport = MILTERTForExport


def _extract_checkpoint_payload(checkpoint_path: str) -> Tuple[dict, dict, dict, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {}) or {}
    model_hparams = ckpt_cfg.get("model_hparams", {}) or {}
    fold_auc = float(ckpt.get("auc", -1.0))
    return ckpt, ckpt_cfg, model_hparams, fold_auc


def _build_base_model(model_hparams: Dict) -> nn.Module:
    model_type = str(model_hparams.get("model_type", "abmil")).lower()
    common = {
        "in_dim": int(model_hparams.get("in_dim", 1536)),
        "embed_dim": int(model_hparams.get("embed_dim", 512)),
        "attn_dim": int(model_hparams.get("attn_dim", 384)),
        "num_fc_layers": int(model_hparams.get("num_fc_layers", 2)),
        "dropout": float(model_hparams.get("dropout", 0.25)),
        "num_classes": int(model_hparams.get("num_classes", 2)),
        "use_layer_norm": bool(model_hparams.get("use_layer_norm", True)),
    }

    if model_type == "abmil":
        from models.abmil.abmil import ABMILTERTConfig, ABMILTERTModel

        config = ABMILTERTConfig(
            gate=bool(model_hparams.get("gate", True)),
            **common,
        )
        return ABMILTERTModel(config)

    if model_type == "acmil":
        from models.acmil.acmil import ACMILTERTConfig, ACMILTERTModel

        config = ACMILTERTConfig(
            n_token=int(model_hparams.get("n_token", 5)),
            n_masked_patch=int(model_hparams.get("n_masked_patch", 10)),
            mask_drop=float(model_hparams.get("mask_drop", 0.6)),
            **common,
        )
        return ACMILTERTModel(config)

    if model_type == "dtfd":
        from models.dtfd.dtfd import DTFDMILTERTConfig, DTFDMILTERTModel

        config = DTFDMILTERTConfig(
            n_pseudo_bags=int(model_hparams.get("n_pseudo_bags", 4)),
            **common,
        )
        return DTFDMILTERTModel(config)

    if model_type == "mhim":
        from models.mhim.mhim import MHIMMILTERTConfig, MHIMMILTERTModel

        config = MHIMMILTERTConfig(
            mask_ratio=float(model_hparams.get("mask_ratio", 0.5)),
            ema_decay=float(model_hparams.get("ema_decay", 0.999)),
            **common,
        )
        return MHIMMILTERTModel(config)

    if model_type == "clam":
        from models.clam.clam import CLAMTERTConfig, CLAMTERTModel

        config = CLAMTERTConfig(
            k_sample=int(model_hparams.get("k_sample", 8)),
            **common,
        )
        return CLAMTERTModel(config)

    if model_type == "transmil":
        from models.transmil.transmil import TransMILTERTConfig, TransMILTERTModel

        config = TransMILTERTConfig(
            in_dim=int(model_hparams.get("in_dim", 1536)),
            embed_dim=int(model_hparams.get("embed_dim", model_hparams.get("transmil_embed_dim", 512))),
            num_heads=int(model_hparams.get("num_heads", 8)),
            num_layers=int(model_hparams.get("num_layers", 2)),
            num_landmarks=int(model_hparams.get("num_landmarks", 256)),
            pinv_iterations=int(model_hparams.get("pinv_iterations", 6)),
            dropout=float(model_hparams.get("dropout", 0.25)),
            num_classes=int(model_hparams.get("num_classes", 2)),
            use_layer_norm=bool(model_hparams.get("use_layer_norm", False)),
        )
        return TransMILTERTModel(config)

    raise ValueError(f"Unsupported model_type for export: {model_type}")


def get_model_signature(model_hparams: Dict) -> Dict[str, str]:
    in_dim = int(model_hparams.get("in_dim", 1536))
    num_classes = int(model_hparams.get("num_classes", 2))
    return {
        "input": f"[-1, -1, {in_dim}]",
        "logits": f"[-1, {num_classes}]",
        "attention": "[-1, -1]",
    }


def build_export_model_from_checkpoint(checkpoint_path: str) -> Tuple[nn.Module, dict, dict, float]:
    _, ckpt_cfg, model_hparams, fold_auc = _extract_checkpoint_payload(checkpoint_path)
    if not model_hparams:
        raise ValueError(
            f"Checkpoint missing model_hparams: {checkpoint_path}. "
            "Old checkpoints without config metadata are not supported by the generic exporter."
        )
    base_model = _build_base_model(model_hparams)
    load_weights(base_model, checkpoint_path)
    export_model = MILTERTForExport(base_model).eval()
    return export_model, ckpt_cfg, model_hparams, fold_auc


def _parse_model_uri(model_uri: str) -> Tuple[str, str, str]:
    if not model_uri.startswith("models:/"):
        raise ValueError(f"Unsupported model URI: {model_uri}")

    spec = model_uri[len("models:/"):]
    if "@" in spec:
        name, alias = spec.split("@", 1)
        return name, "alias", alias
    if "/" in spec:
        name, ref = spec.split("/", 1)
        if ref.isdigit():
            return name, "version", ref
        return name, "alias", ref
    raise ValueError(
        f"Could not parse model URI: {model_uri}. "
        "Use models:/<name>@<alias> or models:/<name>/<version>."
    )


def download_checkpoint_from_model_uri(model_uri: str) -> Tuple[Path, Dict[str, str]]:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    model_name, ref_type, ref_value = _parse_model_uri(model_uri)
    if ref_type == "alias":
        mv = client.get_model_version_by_alias(model_name, ref_value)
    else:
        mv = client.get_model_version(model_name, ref_value)

    source_name = Path(mv.source).name.lower()
    artifact_uri = mv.source

    # Registry source가 TorchScript인 경우, 같은 run에 함께 저장된 원본 checkpoint를 우선 사용.
    if "fp16" in source_name or "torchscript" in source_name:
        checkpoint_artifacts = client.list_artifacts(mv.run_id, "model_checkpoint")
        pt_candidates = sorted(a.path for a in checkpoint_artifacts if a.path.endswith(".pt"))
        if not pt_candidates:
            raise RuntimeError(
                f"Registered model {model_uri} points to TorchScript ({mv.source}), "
                "but no original checkpoint was found under model_checkpoint/."
            )
        artifact_uri = f"runs:/{mv.run_id}/{pt_candidates[0]}"

    local_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri))
    meta = {
        "model_name": model_name,
        "version": str(mv.version),
        "run_id": str(mv.run_id),
        "source": str(mv.source),
        "resolved_artifact_uri": artifact_uri,
    }
    return local_path, meta


# =============================================================================
# 가중치 로드
# =============================================================================

def load_weights(model: nn.Module, checkpoint_path: str) -> dict:
    """체크포인트에서 가중치 로드. config dict 반환."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        # Export 전용 구형 wrapper 호환
        new_sd = {}
        for k, v in sd.items():
            new_k = k[len("model."):] if k.startswith("model.") else k
            new_sd[new_k] = v
        model.load_state_dict(new_sd, strict=True)

    print(f"[+] Loaded weights: {checkpoint_path}")
    return ckpt.get("config", {}) or {}


# =============================================================================
# TorchScript 변환 + 검증
# =============================================================================

def _export_one(model: nn.Module, output_path: str, num_patches: int, fp16: bool) -> str:
    """fp16=True면 half()로 변환 후 trace."""
    m = model.half() if fp16 else model.float()
    m.eval()
    example = torch.randn(1, num_patches, 1536)
    example = example.half() if fp16 else example
    with torch.no_grad():
        traced = torch.jit.trace(m, example)
    traced = torch.jit.optimize_for_inference(traced)
    traced.save(output_path)
    label = "fp16" if fp16 else "fp32"
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"[+] TorchScript ({label}) saved: {output_path}  ({size_mb:.1f} MB)")
    return output_path


def export(model: nn.Module, output_path: str, num_patches: int = 2000) -> str:
    """하위 호환: fp16 단일 export (기존 동작 유지)."""
    return _export_one(model, output_path, num_patches, fp16=True)


def export_both(model: nn.Module, output_dir: Path, model_name: str,
                num_patches: int = 2000) -> tuple:
    """fp32 + fp16 두 파일 생성. (fp32_path, fp16_path) 반환."""
    fp32_path = str(output_dir / f"{model_name}_fp32.pt")
    fp16_path = str(output_dir / f"{model_name}_fp16.pt")
    _export_one(model, fp32_path, num_patches, fp16=False)
    _export_one(model, fp16_path, num_patches, fp16=True)
    return fp32_path, fp16_path


def verify(original: nn.Module, ts_path: str, num_patches: int = 2000, fp16: bool = True):
    original.eval()
    in_dim = int(getattr(original, "base_model", getattr(original, "model", original)).config.in_dim)
    example = torch.randn(1, num_patches, in_dim)
    example = example.half() if fp16 else example
    m = original.half() if fp16 else original.float()

    with torch.no_grad():
        orig_logits, orig_attention = m(example)

    ts_model = torch.jit.load(ts_path)
    ts_model = ts_model.half() if fp16 else ts_model.float()
    ts_model.eval()
    with torch.no_grad():
        ts_logits, ts_attention = ts_model(example)

    ok_logits = torch.allclose(orig_logits.float(), ts_logits.float(), rtol=1e-3, atol=1e-3)
    ok_attention = torch.allclose(orig_attention.float(), ts_attention.float(), rtol=1e-3, atol=1e-3)
    ok = ok_logits and ok_attention
    label = "fp16" if fp16 else "fp32"
    if ok:
        print(f"[+] Verification ({label}) passed")
    else:
        logits_diff = (orig_logits.float() - ts_logits.float()).abs().max().item()
        attn_diff = (orig_attention.float() - ts_attention.float()).abs().max().item()
        print(
            f"[!] Verification ({label}) failed "
            f"(logits max diff: {logits_diff:.6f}, attention max diff: {attn_diff:.6f})"
        )
    return ok


def verify_dynamic_shape(ts_path: str, num_patches: int, in_dim: int, fp16: bool = True):
    alt_patches = max(1, num_patches // 2 + 137)
    ts_model = torch.jit.load(ts_path)
    ts_model = ts_model.half() if fp16 else ts_model.float()
    ts_model.eval()
    example = torch.randn(1, alt_patches, in_dim)
    example = example.half() if fp16 else example
    with torch.no_grad():
        logits, attention = ts_model(example)

    expected_logits = (1, 2)
    expected_attention = (1, alt_patches)
    ok = tuple(logits.shape) == expected_logits and tuple(attention.shape) == expected_attention
    if ok:
        print(
            f"[+] Dynamic-shape check passed "
            f"(input=[1, {alt_patches}, {in_dim}] -> logits={tuple(logits.shape)}, attention={tuple(attention.shape)})"
        )
    else:
        print(
            f"[!] Dynamic-shape check failed "
            f"(got logits={tuple(logits.shape)}, attention={tuple(attention.shape)})"
        )
    return ok


# =============================================================================
# MLflow 등록
# =============================================================================

def register_to_mlflow(ts_path: str, config: dict, model_hparams: dict,
                       fold_auc: float, version_tag: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("thyroid_tert")

    run_name = f"tert_torchscript_{version_tag}"
    pt_name = Path(ts_path).name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(ts_path, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

        mlflow.log_params({
            "version": version_tag,
            "model_type": f"{model_hparams.get('model_type', 'abmil')}_torchscript_fp16",
            "embed_dim": model_hparams.get("embed_dim", 512),
            "attn_dim": model_hparams.get("attn_dim", 384),
            "fold_auc": round(fold_auc, 4),
            "precision": "fp16",
        })

        client = MlflowClient()
        model_name = REGISTERED_MODEL_NAME

        description = (
            f"TERT Mutation — {str(model_hparams.get('model_type', 'abmil')).upper()} TorchScript FP16\n"
            f"Backbone: UNI2-H ({model_hparams.get('in_dim', 1536)}-dim)\n"
            f"Fold AUC: {fold_auc:.4f} | version: {version_tag}\n"
            f"Input: [batch, num_patches, {model_hparams.get('in_dim', 1536)}] FP16\n"
            f"Output: (logits [batch, 2], attention [batch, num_patches]) FP16"
        )

        try:
            client.get_registered_model(model_name)
            client.update_registered_model(model_name, description=description)
        except Exception:
            client.create_registered_model(model_name, description=description)

        source = f"runs:/{run_id}/model/{pt_name}"
        mv = client.create_model_version(
            name=model_name, source=source, run_id=run_id,
            description=f"TorchScript FP16 | Fold AUC: {fold_auc:.4f}"
        )

        client.set_model_version_tag(model_name, mv.version, "format", "torchscript_fp16")
        client.set_model_version_tag(
            model_name, mv.version, "embedding", f"UNI2-H ({model_hparams.get('in_dim', 1536)}-dim)"
        )
        client.set_model_version_tag(model_name, mv.version, "fold_auc", str(round(fold_auc, 4)))
        client.set_model_version_tag(model_name, mv.version, "embed_dim", str(model_hparams.get("embed_dim", 512)))
        client.set_model_version_tag(model_name, mv.version, "attn_dim", str(model_hparams.get("attn_dim", 384)))
        client.set_model_version_tag(
            model_name, mv.version, "input", f"[batch, num_patches, {model_hparams.get('in_dim', 1536)}] FP16"
        )
        client.set_model_version_tag(model_name, mv.version, "output", "(logits [batch,2], attention [batch,N]) FP16")

        client.set_registered_model_alias(model_name, "production", mv.version)
        client.set_registered_model_alias(model_name, "staging", mv.version)

        print(f"[+] Registered: {model_name} version {mv.version} (production/staging)")
        return mv.version


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export TERT MIL model to TorchScript")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=str, default=None,
                              help="학습된 .pt 체크포인트 경로")
    source_group.add_argument("--model_uri", type=str, default=None,
                              help="MLflow model URI (예: models:/thyr-tert@production)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="TorchScript .pt 저장 디렉토리 (기본: exports/<model_type>_torchscript)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="저장 파일명 stem (기본: thyroid_tert_<model_type>)")
    parser.add_argument("--num_patches", type=int, default=2000,
                        help="tracing용 dummy 패치 수 (default: 2000)")
    parser.add_argument("--register", action="store_true",
                        help="MLflow thyr-tert에 등록")
    args = parser.parse_args()

    registry_meta = None
    if args.model_uri:
        ckpt_path, registry_meta = download_checkpoint_from_model_uri(args.model_uri)
        print(f"[+] Downloaded checkpoint from MLflow: {ckpt_path}")
        print(f"    model={registry_meta['model_name']} version={registry_meta['version']} run_id={registry_meta['run_id']}")
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"[!] Checkpoint not found: {ckpt_path}")
            sys.exit(1)

    _, ckpt_cfg, model_hparams, fold_auc = _extract_checkpoint_payload(str(ckpt_path))
    if not model_hparams:
        print(f"[!] model_hparams not found in checkpoint: {ckpt_path}")
        sys.exit(1)

    model_type = str(model_hparams.get("model_type", "abmil")).lower()
    output_dir = Path(args.output_dir) if args.output_dir else (ROOT / "exports" / f"{model_type}_torchscript")
    output_dir.mkdir(parents=True, exist_ok=True)

    in_dim      = int(model_hparams.get("in_dim", 1536))
    embed_dim   = int(model_hparams.get("embed_dim", 512))
    attn_dim    = int(model_hparams.get("attn_dim", 384))
    num_fc_layers = int(model_hparams.get("num_fc_layers", 2))
    dropout     = float(model_hparams.get("dropout", 0.25))
    num_classes = int(model_hparams.get("num_classes", 2))
    signature = get_model_signature(model_hparams)

    print(f"\n{'='*60}")
    print(f"  Checkpoint  : {ckpt_path.name}")
    print(f"  Model type  : {model_type}")
    print(f"  Fold AUC    : {fold_auc:.4f}")
    print(f"  in_dim={in_dim}, embed_dim={embed_dim}, attn_dim={attn_dim}, num_fc_layers={num_fc_layers}")
    print(f"  dropout     : {dropout}")
    print(f"  Precision   : FP16")
    print(f"  Signature   : input={signature['input']} -> logits={signature['logits']}, attention={signature['attention']}")
    print(f"{'='*60}\n")

    export_model, _, _, _ = build_export_model_from_checkpoint(str(ckpt_path))

    # FP16 변환
    export_model = export_model.half().eval()

    # --- TorchScript 변환 ---
    model_name = args.model_name or f"thyroid_tert_{model_type}"
    ts_path = str(output_dir / f"{model_name}.pt")
    export(export_model, ts_path, num_patches=args.num_patches)

    # --- 검증 ---
    print("\n[*] Verifying...")
    verify(export_model, ts_path, num_patches=args.num_patches, fp16=True)
    verify_dynamic_shape(ts_path, num_patches=args.num_patches, in_dim=in_dim, fp16=True)

    # --- info 파일 저장 ---
    info_path = output_dir / f"{model_name}_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Model: {model_name}_fp16\n")
        f.write(f"Source: {ckpt_path}\n")
        if registry_meta is not None:
            f.write(f"Model URI: {args.model_uri}\n")
            f.write(f"Registry Version: {registry_meta['version']}\n")
            f.write(f"Registry Run ID: {registry_meta['run_id']}\n")
            f.write(f"Registry Source: {registry_meta['source']}\n")
            f.write(f"Resolved Artifact: {registry_meta['resolved_artifact_uri']}\n")
        f.write(f"Precision: FP16 (torch.float16)\n\n")
        f.write(
            f"Config: model_type={model_type}, in_dim={in_dim}, embed_dim={embed_dim}, attn_dim={attn_dim}, "
            f"num_fc_layers={num_fc_layers}, dropout={dropout}, num_classes={num_classes}\n"
        )
        f.write(f"Input: {signature['input']} (FP16)\n")
        f.write(f"Output logits: {signature['logits']} (FP16)\n")
        f.write(f"Output attention: {signature['attention']} (FP16)\n\n")
        f.write("Attention tensor is exported from model(return_attention=True).\n")
        f.write("Attention is shape-normalized to [batch, num_patches] without extra post-processing.\n\n")
        f.write(f"IMPORTANT: Input must be .half()\n")
        f.write(f"PyTorch: {torch.__version__}\n")
    print(f"[+] Info saved: {info_path}")

    size_mb = os.path.getsize(ts_path) / (1024 ** 2)
    print(f"[+] Size: {size_mb:.2f} MB")

    # --- MLflow 등록 ---
    if args.register:
        version_tag = ckpt_path.parts[-3] if len(ckpt_path.parts) >= 3 else ckpt_path.stem
        version_tag = version_tag.replace("thyroid_tert_model_", "").replace("thyroid_tert_", "")
        print(f"\n[*] Registering to MLflow thyr-tert ...")
        ver = register_to_mlflow(ts_path, ckpt_cfg, model_hparams, fold_auc, version_tag)
        print(f"[+] Done. thyr-tert version {ver} registered.")
    else:
        print(f"\n[i] Skipping MLflow registration (--register 플래그 없음)")

    print(f"\n[+] Export complete: {ts_path}")


if __name__ == "__main__":
    main()
