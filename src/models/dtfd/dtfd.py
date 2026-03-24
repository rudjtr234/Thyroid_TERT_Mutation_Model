# src/models/dtfd/dtfd.py
"""
DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning
Reference: https://arxiv.org/abs/2203.12081 (CVPR 2022)
GitHub: https://github.com/hrzhang1123/DTFD-MIL

Key idea:
- Tier-1: Split WSI into pseudo-bags, get bag-level features via attention
- Tier-2: Distill slide-level representation from pseudo-bag features
- Strong performance on small cohorts via pseudo-bag augmentation

Adapted for Thyroid TERT Mutation (Wild vs Mutant), UNI2-H 1536-dim input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

from ..mil_template import MIL
from ..layers import create_mlp

MODEL_TYPE = "dtfd_tert"


class TierOneAttention(nn.Module):
    """Tier-1 attention: pools patches within a pseudo-bag."""

    def __init__(self, embed_dim: int, attn_dim: int, dropout: float = 0.25):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        # h: [N, D]  (single pseudo-bag, no batch dim)
        h = self.drop(h)
        A = self.attn_w(self.attn_V(h) * self.attn_U(h))  # [N, 1]
        A = F.softmax(A, dim=0)                             # [N, 1]
        feat = (A * h).sum(0, keepdim=True)                 # [1, D]
        return feat, A.squeeze(-1)                          # [1,D], [N]


class TierTwoAttention(nn.Module):
    """Tier-2 attention: pools pseudo-bag features into slide representation."""

    def __init__(self, embed_dim: int, attn_dim: int, dropout: float = 0.25):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        # h: [B, M, D]  M = n_pseudo_bags
        h = self.drop(h)
        A = self.attn_w(self.attn_V(h) * self.attn_U(h))  # [B, M, 1]
        A = A.transpose(-1, -2)                             # [B, 1, M]
        A_soft = F.softmax(A, dim=-1)
        feat = torch.bmm(A_soft, h).squeeze(1)             # [B, D]
        return feat, A


class DTFDMIL(MIL):
    """
    DTFD-MIL for Thyroid TERT Mutation Prediction.

    Architecture:
        Input (N, 1536)
        -> LayerNorm (optional)
        -> PatchEmbed MLP (1536 -> embed_dim)
        -> Split into n_pseudo_bags pseudo-bags
        -> Tier-1: attention pool each pseudo-bag -> M bag features
        -> Tier-2: attention pool M bag features -> slide feature
        -> Classifier
    """

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        n_pseudo_bags: int = 4,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.n_pseudo_bags = n_pseudo_bags
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

        self.tier1_attn = TierOneAttention(embed_dim, attn_dim, dropout)
        self.tier2_attn = TierTwoAttention(embed_dim, attn_dim, dropout)

        # Tier-1 auxiliary classifier (distillation target)
        self.tier1_classifier = nn.Linear(embed_dim, num_classes)

        # Tier-2 main classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.initialize_weights()

    def initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.tier1_classifier.weight)
        if self.tier1_classifier.bias is not None:
            nn.init.constant_(self.tier1_classifier.bias, 0)

    def _split_pseudo_bags(self, h: torch.Tensor):
        """Split [N, D] into n_pseudo_bags chunks (last bag may be smaller)."""
        N = h.shape[0]
        chunk = max(1, N // self.n_pseudo_bags)
        return h.split(chunk, dim=0)  # tuple of [chunk, D]

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        # h: [B, N, D]  — batch=1 in MIL setting
        B = h.shape[0]

        if self.use_layer_norm:
            h = self.feature_norm(h)
        h = self.patch_embed(h)  # [B, N, embed_dim]

        bag_feats_list = []
        tier1_logits_list = []
        patch_attns = []

        for b in range(B):
            bags = self._split_pseudo_bags(h[b])  # list of [chunk, D]
            b_feats = []
            for bag in bags:
                feat, a = self.tier1_attn(bag)   # feat: [1,D], a: [chunk]
                b_feats.append(feat)
                patch_attns.append(a)
                tier1_logits_list.append(self.tier1_classifier(feat))  # [1, C]
            bag_feats_list.append(torch.cat(b_feats, dim=0))  # [M, D]

        bag_feats = torch.stack(bag_feats_list, dim=0)  # [B, M, D]
        slide_feat, attn2 = self.tier2_attn(bag_feats)  # [B, D], [B,1,M]

        # patch-level attention: tier2 bag score를 각 patch로 전파
        # attn2: [B,1,M], patch_attns: list of [chunk] per bag
        # 결과: [B, 1, N] (N = 전체 patch 수)
        patch_level_attn = None
        if return_attention:
            attn2_soft = F.softmax(attn2, dim=-1)  # [B, 1, M]
            batch_patch_attns = []
            bag_idx = 0
            for b in range(B):
                bags = self._split_pseudo_bags(h[b])
                n_bags = len(bags)
                expanded = []
                for m, bag in enumerate(bags):
                    bag_weight = attn2_soft[b, 0, m]        # scalar
                    patch_w = patch_attns[bag_idx + m]       # [chunk]
                    expanded.append(patch_w * bag_weight)    # [chunk]
                bag_idx += n_bags
                batch_patch_attns.append(torch.cat(expanded, dim=0))  # [N]
            # [B, 1, N]
            patch_level_attn = torch.stack(batch_patch_attns, dim=0).unsqueeze(1)

        log_dict = {
            "attention": patch_level_attn if return_attention else None,
            "attention_entropy": self._attn_entropy(F.softmax(attn2, dim=-1)) if return_attention else None,
            "tier1_logits": tier1_logits_list,
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

        # Tier-1 distillation loss
        if self.training and loss_fn is not None and label is not None:
            for t1_logit in log_dict["tier1_logits"]:
                # label broadcast to each pseudo-bag
                cls_loss = cls_loss + 0.5 * loss_fn(t1_logit, label)

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict["attention"] if return_attention else None,
                "attention_entropy": log_dict.get("attention_entropy"),
                "slide_feats": slide_feat if return_slide_feats else None,
            }
        return logits, cls_loss


class DTFDMILTERTConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        attn_dim: int = 384,
        num_fc_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = 2,
        n_pseudo_bags: int = 4,
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
        self.n_pseudo_bags = n_pseudo_bags
        self.use_layer_norm = use_layer_norm
        self.auto_map = {
            "AutoConfig": "dtfd.dtfd.DTFDMILTERTConfig",
            "AutoModel": "dtfd.dtfd.DTFDMILTERTModel",
        }


class DTFDMILTERTModel(PreTrainedModel):
    config_class = DTFDMILTERTConfig

    def __init__(self, config: DTFDMILTERTConfig, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)
        super().__init__(config)

        self.model = DTFDMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            attn_dim=config.attn_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
            n_pseudo_bags=config.n_pseudo_bags,
            use_layer_norm=config.use_layer_norm,
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(DTFDMILTERTConfig.model_type, DTFDMILTERTConfig)
AutoModel.register(DTFDMILTERTConfig, DTFDMILTERTModel)
