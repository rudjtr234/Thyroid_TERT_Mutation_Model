"""
TransMIL implementation adapted to match the official GitHub structure.

Reference architecture:
- Linear projection
- 2x TransLayer (Nyström attention)
- PPEG between layer1 and layer2
- CLS-token classification
"""

import math

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig

from ..mil_template import MIL

MODEL_TYPE = "transmil_tert"

try:
    from nystrom_attention import NystromAttention
except ImportError:
    NystromAttention = None


class TransLayer(nn.Module):
    """Official-style TransLayer with Nyström attention."""

    def __init__(self, dim: int = 512):
        super().__init__()
        if NystromAttention is None:
            raise ImportError(
                "nystrom_attention is required for TransMIL. "
                "Install with: pip install nystrom-attention"
            )

        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    """Positional Prior Enhancement Generator."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, h_grid: int, w_grid: int) -> torch.Tensor:
        batch_size, _, channels = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(batch_size, channels, h_grid, w_grid)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(MIL):
    """TransMIL for TERT mutation prediction."""

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        num_classes: int = 2,
        **kwargs
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.pos_layer = PPEG(dim=embed_dim)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer1 = TransLayer(dim=embed_dim)
        self.layer2 = TransLayer(dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self._fc2 = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_classifier(self):
        nn.init.xavier_uniform_(self._fc2.weight)
        if self._fc2.bias is not None:
            nn.init.constant_(self._fc2.bias, 0)

    @staticmethod
    def _attention_entropy(attn_probs: torch.Tensor) -> torch.Tensor:
        entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1)
        return entropy.mean()

    def _encode_tokens(self, h: torch.Tensor):
        """Encode patch features into token sequence and keep original patch count."""
        h = self._fc1(h)

        num_tokens = h.shape[1]
        h_grid, w_grid = math.ceil(num_tokens ** 0.5), math.ceil(num_tokens ** 0.5)
        add_length = h_grid * w_grid - num_tokens
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        batch_size = h.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, h_grid, w_grid)
        h = self.layer2(h)
        h = self.norm(h)
        return h, num_tokens

    def forward_features(
        self,
        h: torch.Tensor,
        attn_mask=None,
        return_attention: bool = True,
    ):
        h, num_tokens = self._encode_tokens(h)
        slide_feat = h[:, 0]

        log_dict = {
            "attention": None,
            "attention_entropy": None,
        }

        if return_attention:
            # GitHub 구조는 patch attention을 직접 반환하지 않으므로,
            # 마지막 layer 출력에서 CLS-patch 유사도 기반 score를 구성한다.
            cls_token = h[:, :1, :]                    # [B, 1, C]
            patch_tokens = h[:, 1:1 + num_tokens, :]  # [B, N, C] (unpadded only)
            attn_logits = torch.matmul(patch_tokens, cls_token.transpose(-1, -2)).squeeze(-1)
            attn_logits = attn_logits / math.sqrt(h.shape[-1])
            attn_probs = torch.softmax(attn_logits, dim=-1)

            log_dict["attention"] = attn_logits.unsqueeze(1)  # [B, 1, N]
            log_dict["attention_entropy"] = self._attention_entropy(attn_probs)

        return slide_feat, log_dict

    def get_patch_attribution(
        self,
        h: torch.Tensor,
        target_class: torch.Tensor = None,
        use_pred_class: bool = True,
        positive_only: bool = True,
    ):
        """
        TransMIL-B: class-specific patch attribution via gradient * input.
        Returns patch scores for original input patches.
        """
        # Attribute to original patch features for robust gradient connectivity.
        h_input = h.detach().clone().requires_grad_(True)
        slide_feat, _ = self.forward_features(
            h_input,
            attn_mask=None,
            return_attention=False,
        )
        logits = self._fc2(slide_feat)

        if target_class is None:
            if use_pred_class:
                target_class = torch.argmax(logits, dim=1)
            else:
                target_class = torch.ones(
                    logits.shape[0], dtype=torch.long, device=logits.device
                )
        elif isinstance(target_class, int):
            target_class = torch.full(
                (logits.shape[0],),
                fill_value=target_class,
                dtype=torch.long,
                device=logits.device,
            )
        else:
            target_class = target_class.to(logits.device)

        target_logit = logits.gather(1, target_class.view(-1, 1)).sum()
        grads = torch.autograd.grad(
            target_logit,
            h_input,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        attr_scores = (grads * h_input).sum(dim=-1)  # [B, N]
        if positive_only:
            attr_scores = torch.relu(attr_scores)

        return {
            "logits": logits.detach(),
            "attribution_scores": attr_scores,
            "target_class": target_class,
        }

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return self._fc2(h)

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
        slide_feats, log_dict = self.forward_features(
            h=h,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )
        logits = self.forward_head(slide_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict["attention"] if return_attention else None,
                "attention_entropy": log_dict.get("attention_entropy"),
                "slide_feats": slide_feats if return_slide_feats else None,
            }

        return logits, cls_loss


class TransMILTERTConfig(PretrainedConfig):
    """HuggingFace config for TransMIL TERT model."""

    model_type = MODEL_TYPE

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_layer_norm = use_layer_norm
        self.auto_map = {
            "AutoConfig": "transmil.transmil.TransMILTERTConfig",
            "AutoModel": "transmil.transmil.TransMILTERTModel",
        }


class TransMILTERTModel(PreTrainedModel):
    config_class = TransMILTERTConfig

    def __init__(self, config: TransMILTERTConfig, **kwargs):
        self.config = config
        for key, value in kwargs.items():
            setattr(config, key, value)
        super().__init__(config)

        self.model = TransMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.get_patch_attribution = self.model.get_patch_attribution
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(TransMILTERTConfig.model_type, TransMILTERTConfig)
AutoModel.register(TransMILTERTConfig, TransMILTERTModel)
