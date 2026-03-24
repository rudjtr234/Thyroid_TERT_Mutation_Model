# src/models/acmil/acmil.py
"""
ACMIL: Attention-Challenging Multiple Instance Learning
Reference: https://arxiv.org/abs/2311.07125
GitHub: https://github.com/dazhangyu123/ACMIL

Key idea:
- Multi-head attention pooling (n_token heads)
- Auxiliary attention branches to challenge dominant attention
- Reduces attention collapse / over-fitting in small cohorts

Adapted for Thyroid TERT Mutation (Wild vs Mutant), UNI2-H 1536-dim input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

from ..mil_template import MIL
from ..layers import create_mlp

MODEL_TYPE = "acmil_tert"


class AttentionPool(nn.Module):
    """Single attention head pool used inside ACMIL."""

    def __init__(self, embed_dim: int, attn_dim: int, dropout: float = 0.25):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        # h: [B, N, D]
        h = self.drop(h)
        A = self.attn_w(self.attn_V(h) * self.attn_U(h))  # [B, N, 1]
        A = A.transpose(-1, -2)                             # [B, 1, N]
        A_soft = F.softmax(A, dim=-1)
        feat = torch.bmm(A_soft, h).squeeze(1)             # [B, D]
        return feat, A


class ACMIL(MIL):
    """
    ACMIL for Thyroid TERT Mutation Prediction.

    Architecture:
        Input (N, 1536)
        -> LayerNorm (optional)
        -> PatchEmbed MLP (1536 -> embed_dim)
        -> n_token attention heads (primary + auxiliary)
        -> Concat pooled features -> Classifier
        -> Auxiliary classifiers for training (challenging attention)
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        n_token: int = 5,
        n_masked_patch: int = 10,
        mask_drop: float = 0.6,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
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

        # n_token attention heads: 1 primary + (n_token-1) auxiliary
        self.attention_heads = nn.ModuleList(
            [AttentionPool(embed_dim, attn_dim, dropout) for _ in range(n_token)]
        )

        # Primary classifier (uses concatenated features from all heads)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_token, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Auxiliary classifiers (one per auxiliary head, for training only)
        self.aux_classifiers = nn.ModuleList(
            [nn.Linear(embed_dim, num_classes) for _ in range(n_token - 1)]
        )

        self.initialize_weights()

    def initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for clf in self.aux_classifiers:
            nn.init.xavier_uniform_(clf.weight)
            if clf.bias is not None:
                nn.init.constant_(clf.bias, 0)

    def _mask_patches(self, h: torch.Tensor, primary_attn: torch.Tensor):
        """
        Mask top-k patches attended by primary head to challenge auxiliary heads.
        primary_attn: [B, 1, N]
        """
        if not self.training or self.n_masked_patch == 0:
            return h
        B, N, D = h.shape
        k = min(self.n_masked_patch, N)
        # top-k indices from primary attention
        top_idx = primary_attn.squeeze(1).topk(k, dim=-1).indices  # [B, k]
        mask = torch.ones(B, N, device=h.device)
        mask.scatter_(1, top_idx, 0.0)
        # random dropout on remaining
        if self.mask_drop > 0:
            rand_mask = (torch.rand_like(mask) > self.mask_drop).float()
            mask = mask * rand_mask
        return h * mask.unsqueeze(-1)

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        if self.use_layer_norm:
            h = self.feature_norm(h)
        h = self.patch_embed(h)  # [B, N, D]

        # Primary head
        feat0, attn0 = self.attention_heads[0](h)  # feat0: [B, D], attn0: [B, 1, N]

        feats = [feat0]
        aux_feats = []

        # Auxiliary heads on masked input
        h_masked = self._mask_patches(h, attn0.detach())
        for i in range(1, self.n_token):
            feat_i, _ = self.attention_heads[i](h_masked)
            feats.append(feat_i)
            aux_feats.append(feat_i)

        slide_feat = torch.cat(feats, dim=-1)  # [B, D*n_token]

        log_dict = {
            "attention": attn0 if return_attention else None,
            "attention_entropy": self._attn_entropy(F.softmax(attn0, dim=-1)) if return_attention else None,
            "aux_feats": aux_feats,
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

        # Auxiliary loss (during training)
        if self.training and loss_fn is not None and label is not None:
            for i, aux_feat in enumerate(log_dict["aux_feats"]):
                aux_logits = self.aux_classifiers[i](aux_feat)
                cls_loss = cls_loss + 0.5 * loss_fn(aux_logits, label)

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict["attention"] if return_attention else None,
                "attention_entropy": log_dict.get("attention_entropy"),
                "slide_feats": slide_feat if return_slide_feats else None,
            }
        return logits, cls_loss


class ACMILTERTConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        n_token: int = 5,
        n_masked_patch: int = 10,
        mask_drop: float = 0.6,
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
        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.use_layer_norm = use_layer_norm
        self.auto_map = {
            "AutoConfig": "acmil.acmil.ACMILTERTConfig",
            "AutoModel": "acmil.acmil.ACMILTERTModel",
        }


class ACMILTERTModel(PreTrainedModel):
    config_class = ACMILTERTConfig

    def __init__(self, config: ACMILTERTConfig, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)
        super().__init__(config)

        self.model = ACMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            attn_dim=config.attn_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            n_token=config.n_token,
            n_masked_patch=config.n_masked_patch,
            mask_drop=config.mask_drop,
            use_layer_norm=config.use_layer_norm,
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(ACMILTERTConfig.model_type, ACMILTERTConfig)
AutoModel.register(ACMILTERTConfig, ACMILTERTModel)
