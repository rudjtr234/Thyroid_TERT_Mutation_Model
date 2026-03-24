#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_torchscript.py

ABMIL TERT 모델을 TorchScript fp32 + fp16 두 포맷으로 변환하는 스크립트.
register_model.py --with_torchscript 에서 호출되거나 단독 실행 가능.

구조:
  Input  [batch, num_patches, 1536]
  Output (logits [batch, 2], attention [batch, num_patches])

실행 방법:

    cd /path/to/project

    # 변환만 (fp32 + fp16 모두 생성)
    python src/inference/export_torchscript.py \
        --checkpoint outputs/thyroid_tert_model_v0.2.2/checkpoints/best_model_fold2_auc1.0000.pt \
        --output_dir exports/abmil_torchscript
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.layers import GlobalGatedAttention, create_mlp

# Set MLFLOW_TRACKING_INSECURE_TLS=true in your environment if using self-signed certificates

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "thyr-tert"


# =============================================================================
# Export 전용 모델 클래스
# =============================================================================

class ABMILTERTForExport(nn.Module):
    """
    TorchScript 변환용 ABMIL TERT 추론 모델 (FP16).

    v0.2.4 구조:
      LayerNorm(1536)
      PatchEmbed MLP: 1536 → 768 → 768
      GatedAttention: V/U (768→512), w (512→1)
      Classifier: 768 → 384 → 2

    Input : [batch, num_patches, 1536]  FP16
    Output: (logits [batch, 2], attention [batch, num_patches])  FP16
    """

    def __init__(self,
                 in_dim: int = 1536,
                 embed_dim: int = 768,
                 attn_dim: int = 512,
                 num_fc_layers: int = 2,
                 dropout: float = 0.25,
                 num_classes: int = 2):
        super().__init__()

        self.feature_norm = nn.LayerNorm(in_dim)

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        self.global_attn = GlobalGatedAttention(
            L=embed_dim,
            D=attn_dim,
            dropout=0.0,
            num_classes=1,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, h: torch.Tensor):
        # h: [B, N, 1536] FP16
        h = self.feature_norm(h)
        h = self.patch_embed(h)                      # [B, N, 768]

        A = self.global_attn(h)                      # [B, N, 1]
        A = torch.transpose(A, -2, -1)               # [B, 1, N]
        A = torch.softmax(A, dim=-1)                 # [B, 1, N]

        pooled = torch.bmm(A, h).squeeze(1)          # [B, 768]
        logits = self.classifier(pooled)             # [B, 2]
        attention = A.squeeze(1)                     # [B, N]

        return logits, attention


# =============================================================================
# 가중치 로드
# =============================================================================

def load_weights(model: nn.Module, checkpoint_path: str) -> dict:
    """체크포인트에서 가중치 로드. config dict 반환."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    # 'model.' prefix 제거
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


def verify(original: nn.Module, ts_path: str, num_patches: int = 2000):
    original.eval()
    # 파일명에 fp16이 포함되어 있으면 half로 검증
    fp16 = "fp16" in Path(ts_path).name
    example = torch.randn(1, num_patches, 1536)
    example = example.half() if fp16 else example
    m = original.half() if fp16 else original.float()

    with torch.no_grad():
        orig_logits, _ = m(example)

    ts_model = torch.jit.load(ts_path)
    ts_model.eval()
    with torch.no_grad():
        ts_logits, _ = ts_model(example)

    ok = torch.allclose(orig_logits.float(), ts_logits.float(), rtol=1e-3, atol=1e-3)
    label = "fp16" if fp16 else "fp32"
    if ok:
        print(f"[+] Verification ({label}) passed")
    else:
        diff = (orig_logits.float() - ts_logits.float()).abs().max().item()
        print(f"[!] Verification ({label}) failed — max diff: {diff:.6f}")
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
            "model_type": "abmil_torchscript_fp16",
            "embed_dim": model_hparams.get("embed_dim", 768),
            "attn_dim": model_hparams.get("attn_dim", 512),
            "fold_auc": round(fold_auc, 4),
            "precision": "fp16",
        })

        client = MlflowClient()
        model_name = REGISTERED_MODEL_NAME

        description = (
            f"TERT Mutation — ABMIL TorchScript FP16\n"
            f"Backbone: UNI2-H (1536-dim)\n"
            f"Fold AUC: {fold_auc:.4f} | version: {version_tag}\n"
            f"Input: [batch, num_patches, 1536] FP16\n"
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
        client.set_model_version_tag(model_name, mv.version, "embedding", "UNI2-H (1536-dim)")
        client.set_model_version_tag(model_name, mv.version, "fold_auc", str(round(fold_auc, 4)))
        client.set_model_version_tag(model_name, mv.version, "embed_dim", str(model_hparams.get("embed_dim", 768)))
        client.set_model_version_tag(model_name, mv.version, "attn_dim", str(model_hparams.get("attn_dim", 512)))
        client.set_model_version_tag(model_name, mv.version, "input", "[batch, num_patches, 1536] FP16")
        client.set_model_version_tag(model_name, mv.version, "output", "(logits [batch,2], attention [batch,N]) FP16")

        client.set_registered_model_alias(model_name, "production", mv.version)
        client.set_registered_model_alias(model_name, "staging", mv.version)

        print(f"[+] Registered: {model_name} version {mv.version} (production/staging)")
        return mv.version


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export ABMIL TERT to FP16 TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습된 .pt 체크포인트 경로")
    parser.add_argument("--output_dir", type=str, default="exports/abmil_torchscript",
                        help="TorchScript .pt 저장 디렉토리 (default: exports/abmil_torchscript)")
    parser.add_argument("--model_name", type=str, default="thyroid_tert_abmil",
                        help="저장 파일명 (default: thyroid_tert_abmil → thyroid_tert_abmil.pt)")
    parser.add_argument("--num_patches", type=int, default=2000,
                        help="tracing용 dummy 패치 수 (default: 2000)")
    parser.add_argument("--register", action="store_true",
                        help="MLflow thyr-tert에 등록")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- config 추출 ---
    ckpt_raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt_raw.get("config", {}) or {}
    model_hparams = ckpt_cfg.get("model_hparams", {}) or {}
    fold_auc = float(ckpt_raw.get("auc", -1.0))

    in_dim      = int(model_hparams.get("in_dim", 1536))
    embed_dim   = int(model_hparams.get("embed_dim", 768))
    attn_dim    = int(model_hparams.get("attn_dim", 512))
    num_fc_layers = int(model_hparams.get("num_fc_layers", 2))
    dropout     = float(model_hparams.get("dropout", 0.25))

    print(f"\n{'='*60}")
    print(f"  Checkpoint  : {ckpt_path.name}")
    print(f"  Fold AUC    : {fold_auc:.4f}")
    print(f"  in_dim={in_dim}, embed_dim={embed_dim}, attn_dim={attn_dim}, num_fc_layers={num_fc_layers}")
    print(f"  dropout     : {dropout}")
    print(f"  Precision   : FP16")
    print(f"{'='*60}\n")

    # --- 모델 생성 + 가중치 로드 ---
    model = ABMILTERTForExport(
        in_dim=in_dim,
        embed_dim=embed_dim,
        attn_dim=attn_dim,
        num_fc_layers=num_fc_layers,
        dropout=dropout,
    )
    load_weights(model, str(ckpt_path))

    # FP16 변환
    model = model.half().eval()

    # --- TorchScript 변환 ---
    ts_path = str(output_dir / f"{args.model_name}.pt")
    export(model, ts_path, num_patches=args.num_patches)

    # --- 검증 ---
    print("\n[*] Verifying...")
    verify(model, ts_path, num_patches=args.num_patches)

    # --- info 파일 저장 ---
    info_path = output_dir / f"{args.model_name}_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Model: {args.model_name}_fp16\n")
        f.write(f"Source: {ckpt_path}\n")
        f.write(f"Precision: FP16 (torch.float16)\n\n")
        f.write(
            f"Config: in_dim={in_dim}, embed_dim={embed_dim}, attn_dim={attn_dim}, "
            f"num_fc_layers={num_fc_layers}, dropout={dropout}, num_classes=2\n"
        )
        f.write(f"Input: [batch, num_patches, {in_dim}] (FP16)\n")
        f.write(f"Output: (logits [batch, 2], attention [batch, num_patches]) (FP16)\n\n")
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
