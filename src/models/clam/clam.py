# src/models/clam/clam.py
"""
CLAM: Clustering-constrained Attention Multiple Instance Learning
Reference: https://www.nature.com/articles/s41551-020-00682-w (Nature BME 2021)
GitHub: https://github.com/mahmoodlab/CLAM  [GPL-3.0]

NOTE: Original code is GPL-3.0. This is an independent re-implementation
      for research use. For commercial use, consult license terms.

Key idea:
- Attention-based pooling with instance-level clustering constraint
- SB (single-branch) or MB (multi-branch) variants
- Instance-level pseudo-labels from attention scores for auxiliary loss

Adapted for Thyroid TERT Mutation (Wild vs Mutant), UNI2-H 1536-dim input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

from ..mil_template import MIL
from ..layers import create_mlp

MODEL_TYPE = "clam_tert"


class CLAMAttention(nn.Module):
    """Gated attention for CLAM (same as ABMIL gated attention)."""

    def __init__(self, embed_dim: int, attn_dim: int, dropout: float = 0.25):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        # h: [B, N, D]
        h_d = self.drop(h)
        A = self.attn_w(self.attn_V(h_d) * self.attn_U(h_d))  # [B, N, 1]
        A = A.transpose(-1, -2)                                  # [B, 1, N]
        A_soft = F.softmax(A, dim=-1)
        feat = torch.bmm(A_soft, h).squeeze(1)                  # [B, D]
        return feat, A


class CLAM(MIL):
    """
    CLAM-SB for Thyroid TERT Mutation Prediction.

    Architecture:
        Input (N, 1536)
        -> LayerNorm (optional)
        -> PatchEmbed MLP (1536 -> embed_dim)
        -> Gated Attention -> slide feature
        -> Classifier (slide-level)
        -> Instance-level pseudo-label clustering loss (top-k / bottom-k)
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        k_sample: int = 8,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.k_sample = k_sample
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

        self.attention = CLAMAttention(embed_dim, attn_dim, dropout)

        # Slide-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # Instance-level classifiers (one per class for SB variant)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, 2) for _ in range(num_classes)]
        )

        self.initialize_weights()

    def initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for clf in self.instance_classifiers:
            nn.init.xavier_uniform_(clf.weight)
            if clf.bias is not None:
                nn.init.constant_(clf.bias, 0)

    def _instance_loss(self, h: torch.Tensor, attn: torch.Tensor, label: torch.Tensor, loss_fn: nn.Module):
        """
        Clustering constraint: top-k patches = positive instances,
        bottom-k patches = negative instances.
        """
        B = h.shape[0]
        total_inst_loss = torch.tensor(0.0, device=h.device)

        for b in range(B):
            lbl = label[b].item()
            inst_clf = self.instance_classifiers[lbl]
            a = attn[b, 0]  # [N]
            N = a.shape[0]
            k = min(self.k_sample, N // 2)

            top_idx = a.topk(k, dim=0).indices
            bot_idx = a.topk(k, dim=0, largest=False).indices

            top_feats = h[b, top_idx]  # [k, D]
            bot_feats = h[b, bot_idx]  # [k, D]

            pos_logits = inst_clf(top_feats)
            neg_logits = inst_clf(bot_feats)

            pos_labels = torch.ones(k, dtype=torch.long, device=h.device)
            neg_labels = torch.zeros(k, dtype=torch.long, device=h.device)

            inst_loss = loss_fn(pos_logits, pos_labels) + loss_fn(neg_logits, neg_labels)
            total_inst_loss = total_inst_loss + inst_loss

        return total_inst_loss / B

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        if self.use_layer_norm:
            h = self.feature_norm(h)
        h = self.patch_embed(h)  # [B, N, D]

        slide_feat, attn = self.attention(h)  # [B, D], [B, 1, N]

        log_dict = {
            "attention": attn if return_attention else None,
            "attention_entropy": self._attn_entropy(F.softmax(attn, dim=-1)) if return_attention else None,
            "h_embed": h,
            "raw_attn": attn,
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

        # Instance-level clustering loss (training only)
        if self.training and loss_fn is not None and label is not None:
            inst_loss = self._instance_loss(log_dict["h_embed"], log_dict["raw_attn"], label, loss_fn)
            cls_loss = cls_loss + 0.3 * inst_loss

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict["attention"] if return_attention else None,
                "attention_entropy": log_dict.get("attention_entropy"),
                "slide_feats": slide_feat if return_slide_feats else None,
            }
        return logits, cls_loss


class CLAMTERTConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        k_sample: int = 8,
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
        self.k_sample = k_sample
        self.use_layer_norm = use_layer_norm
        self.auto_map = {
            "AutoConfig": "clam.clam.CLAMTERTConfig",
            "AutoModel": "clam.clam.CLAMTERTModel",
        }


class CLAMTERTModel(PreTrainedModel):
    config_class = CLAMTERTConfig

    def __init__(self, config: CLAMTERTConfig, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)
        super().__init__(config)

        self.model = CLAM(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            attn_dim=config.attn_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            k_sample=config.k_sample,
            use_layer_norm=config.use_layer_norm,
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(CLAMTERTConfig.model_type, CLAMTERTConfig)
AutoModel.register(CLAMTERTConfig, CLAMTERTModel)
