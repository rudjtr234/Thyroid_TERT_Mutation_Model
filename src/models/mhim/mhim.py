# src/models/mhim/mhim.py
"""
MHIM-MIL: Masked Hard Instance Mining for MIL
Reference: https://arxiv.org/abs/2307.15254 (ICCV 2023 Oral / IJCV 2025)
GitHub: https://github.com/DearCaat/MHIM-MIL

Key idea:
- Teacher-student framework: teacher identifies hard (low-attention) instances
- Student is trained on masked input (hard instances unmasked, easy masked)
- Forces model to learn from hard boundary patches
- Single-model inference: only student used at test time

Adapted for Thyroid TERT Mutation (Wild vs Mutant), UNI2-H 1536-dim input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

from ..mil_template import MIL
from ..layers import create_mlp

MODEL_TYPE = "mhim_tert"


class GatedAttentionPool(nn.Module):
    """Shared gated attention pooling used by both teacher and student."""

    def __init__(self, embed_dim: int, attn_dim: int, dropout: float = 0.25):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        # h: [B, N, D]
        h_drop = self.drop(h)
        A = self.attn_w(self.attn_V(h_drop) * self.attn_U(h_drop))  # [B, N, 1]
        A = A.transpose(-1, -2)                                        # [B, 1, N]
        A_soft = F.softmax(A, dim=-1)
        feat = torch.bmm(A_soft, h).squeeze(1)                        # [B, D]
        return feat, A


class MHIMMIL(MIL):
    """
    MHIM-MIL for Thyroid TERT Mutation Prediction.

    Architecture (training):
        Input (N, 1536)
        -> LayerNorm (optional)
        -> PatchEmbed MLP (1536 -> embed_dim)
        -> Teacher: attention on full input -> identify low-attention (hard) patches
        -> Student: attention on masked input (easy patches masked out)
        -> Student classifier -> main loss
        -> Teacher classifier -> auxiliary loss (EMA-updated teacher)

    Inference:
        Only student path (no masking at eval).
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.999,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.feature_norm = nn.LayerNorm(in_dim)

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )

        # Student (main, gradient-updated)
        self.student_attn = GatedAttentionPool(embed_dim, attn_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # Teacher (EMA copy, no gradient)
        self.teacher_attn = GatedAttentionPool(embed_dim, attn_dim, dropout)
        self.teacher_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        # student 먼저 Xavier 초기화
        self.initialize_weights()
        # teacher = student 복사 (initialize_weights 이후)
        self._copy_to_teacher()
        # Freeze teacher params (EMA only)
        for p in self.teacher_attn.parameters():
            p.requires_grad = False
        for p in self.teacher_classifier.parameters():
            p.requires_grad = False

    def _copy_to_teacher(self):
        for t, s in zip(self.teacher_attn.parameters(), self.student_attn.parameters()):
            t.data.copy_(s.data)
        for t, s in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
            t.data.copy_(s.data)

    @torch.no_grad()
    def update_teacher(self):
        """EMA update: call after each training step."""
        for t, s in zip(self.teacher_attn.parameters(), self.student_attn.parameters()):
            t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)
        for t, s in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
            t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def initialize_classifier(self):
        for m in list(self.classifier.modules()) + list(self.teacher_classifier.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _hard_mask(self, h: torch.Tensor, teacher_attn: torch.Tensor):
        """
        Mask easy (high-attention) patches; keep hard (low-attention) patches.
        teacher_attn: [B, 1, N]
        """
        B, N, D = h.shape
        k_keep = max(1, int(N * (1.0 - self.mask_ratio)))
        # low-attention = hard instances → keep them
        bottom_idx = teacher_attn.squeeze(1).topk(k_keep, dim=-1, largest=False).indices  # [B, k]
        mask = torch.zeros(B, N, device=h.device)
        mask.scatter_(1, bottom_idx, 1.0)
        return h * mask.unsqueeze(-1)

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        if self.use_layer_norm:
            h = self.feature_norm(h)
        h = self.patch_embed(h)  # [B, N, D]

        if self.training:
            # Teacher pass (no grad)
            with torch.no_grad():
                _, teacher_a = self.teacher_attn(h)  # [B, 1, N]
            h_masked = self._hard_mask(h, teacher_a)
            slide_feat, student_a = self.student_attn(h_masked)
        else:
            slide_feat, student_a = self.student_attn(h)

        log_dict = {
            "attention": student_a if return_attention else None,
            "attention_entropy": self._attn_entropy(F.softmax(student_a, dim=-1)) if return_attention else None,
            "h_embed": h,
        }
        return slide_feat, log_dict

    @staticmethod
    def _attn_entropy(a: torch.Tensor):
        return -(a * torch.log(a + 1e-9)).sum(-1).mean()

    def forward_head(self, h: torch.Tensor):
        return self.classifier(h)

    def forward(
        self,
        h: torch.Tensor,
        loss_fn: nn.Module = None,
        label: torch.LongTensor = None,
        attn_mask=None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
        return_extra: bool = False,
    ):
        slide_feat, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(slide_feat)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        # Teacher auxiliary loss + EMA update (training only)
        if self.training and loss_fn is not None and label is not None:
            h_embed = log_dict["h_embed"]
            with torch.no_grad():
                t_feat, _ = self.teacher_attn(h_embed)
            t_logits = self.teacher_classifier(t_feat)
            cls_loss = cls_loss + 0.5 * loss_fn(t_logits, label)
            self.update_teacher()

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict["attention"] if return_attention else None,
                "attention_entropy": log_dict.get("attention_entropy"),
                "slide_feats": slide_feat if return_slide_feats else None,
            }
        return logits, cls_loss


class MHIMMILTERTConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.999,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.use_layer_norm = use_layer_norm
        self.auto_map = {
            "AutoConfig": "mhim.mhim.MHIMMILTERTConfig",
            "AutoModel": "mhim.mhim.MHIMMILTERTModel",
        }


class MHIMMILTERTModel(PreTrainedModel):
    config_class = MHIMMILTERTConfig

    def __init__(self, config: MHIMMILTERTConfig, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)
        super().__init__(config)

        self.model = MHIMMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            attn_dim=config.attn_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            mask_ratio=config.mask_ratio,
            ema_decay=config.ema_decay,
            use_layer_norm=config.use_layer_norm,
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier
        self.update_teacher = self.model.update_teacher


AutoConfig.register(MHIMMILTERTConfig.model_type, MHIMMILTERTConfig)
AutoModel.register(MHIMMILTERTConfig, MHIMMILTERTModel)
