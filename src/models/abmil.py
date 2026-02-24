# src/models/abmil.py
"""
ABMIL (Attention-based Multiple Instance Learning) model for Thyroid TERT Mutation Prediction.

Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
- Wild (no mutation) = 0
- Mutant (C228T or C250T) = 1

Optimized for UNI2 1536-dimensional features.
"""

from .mil_template import MIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GlobalAttention, GlobalGatedAttention, create_mlp
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel

MODEL_TYPE = 'abmil_tert'


class ABMIL(MIL):
    """
    ABMIL (Attention-based Multiple Instance Learning) model for Thyroid TERT.

    Task: TERT Promoter Mutation Prediction (Wild vs Mutant)
    - Wild (no mutation) = 0
    - Mutant (C228T or C250T) = 1

    Optimized for UNI2 1536-dimensional features with balanced parameter count.
    """

    def __init__(self,
                 gate: bool = True,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 2,
                 dropout: float = 0.25,
                 in_dim: int = 1536,           # UNI2 dimension
                 num_classes: int = 2,
                 use_layer_norm: bool = True,
                 **kwargs):

        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        # Feature normalization (UNI2 feature stabilization)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.feature_norm = nn.LayerNorm(in_dim)

        # Patch embedding with gradual dimension reduction
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        # Global attention
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        # Classifier with additional regularization
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes)
            )

        self.initialize_weights()

    def initialize_classifier(self):
        """Sequential 내부의 Linear 레이어들을 초기화"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        """
        Compute attention scores with optional input normalization.
        """
        # Input normalization (UNI2 feature stabilization)
        if self.use_layer_norm:
            h = self.feature_norm(h)

        h = self.patch_embed(h)
        A = self.global_attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M

        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        A = F.softmax(A_base, dim=-1)
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C

        log_dict = {
            'attention': A_base if return_attention else None,
            'attention_entropy': self._compute_attention_entropy(A) if return_attention else None
        }
        return h, log_dict

    def _compute_attention_entropy(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute attention entropy for monitoring attention collapse.
        Low entropy (<1.0) indicates potential collapse.
        """
        entropy = -(A * torch.log(A + 1e-9)).sum(-1)
        return entropy.mean()

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Classification head with improved architecture.
        """
        logits = self.classifier(h)
        return logits

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
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.
            return_extra: If True, return dict (for eval/analysis).
                          If False, return (logits, loss) tuple for training (DDP-safe).
        """
        wsi_feats, log_dict = self.forward_features(
            h, attn_mask=attn_mask, return_attention=return_attention
        )
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        if return_extra:
            return {
                "logits": logits,
                "loss": cls_loss,
                "attention": log_dict["attention"] if return_attention else None,
                "attention_entropy": log_dict.get("attention_entropy", None),
                "slide_feats": wsi_feats if return_slide_feats else None,
            }

        # DDP-friendly output: only tensors
        return logits, cls_loss


class ABMILTERTConfig(PretrainedConfig):
    """
    Configuration class for UNI2-optimized ABMIL model for TERT prediction.
    """
    model_type = MODEL_TYPE

    def __init__(self,
                 gate: bool = True,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 2,
                 dropout: float = 0.25,
                 in_dim: int = 1536,
                 num_classes: int = 2,
                 use_layer_norm: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.use_layer_norm = use_layer_norm
        self.auto_map = {
            "AutoConfig": "abmil.ABMILTERTConfig",
            "AutoModel": "abmil.ABMILTERTModel",
        }


class ABMILTERTModel(PreTrainedModel):
    config_class = ABMILTERTConfig

    def __init__(self, config: ABMILTERTConfig, **kwargs):
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)
        self.model = ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            num_classes=config.num_classes,
            use_layer_norm=config.use_layer_norm
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(ABMILTERTConfig.model_type, ABMILTERTConfig)
AutoModel.register(ABMILTERTConfig, ABMILTERTModel)


# Test code
if __name__ == "__main__":
    print("Testing UNI2-optimized ABMIL for TERT binary classification...")
    print("=" * 60)

    # Create model with config
    config = ABMILTERTConfig()
    model = ABMILTERTModel(config)

    print(f"Model Configuration:")
    print(f"  - Input dim: {config.in_dim}")
    print(f"  - Embed dim: {config.embed_dim}")
    print(f"  - Attn dim: {config.attn_dim}")
    print(f"  - FC layers: {config.num_fc_layers}")
    print(f"  - Dropout: {config.dropout}")
    print(f"  - Layer Norm: {config.use_layer_norm}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Test data
    batch_size = 2
    num_patches = 1000
    uni2_dim = 1536

    dummy_features = torch.randn(batch_size, num_patches, uni2_dim)
    dummy_label = torch.tensor([0, 1])  # 0=Wild, 1=Mutant

    print(f"Input shape: {dummy_features.shape}")
    print(f"Input feature norm: {dummy_features.norm(dim=-1).mean():.3f}")
    print()

    # Training mode
    model.train()
    logits, loss = model(
        h=dummy_features,
        loss_fn=nn.CrossEntropyLoss(),
        label=dummy_label,
        return_extra=False
    )
    print(f"Training mode:")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Loss: {loss:.4f}")
    print()

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        outputs = model(
            h=dummy_features,
            loss_fn=nn.CrossEntropyLoss(),
            label=dummy_label,
            return_attention=True,
            return_slide_feats=True,
            return_extra=True
        )

    print(f"Evaluation mode:")
    print(f"  - Logits shape: {outputs['logits'].shape}")
    print(f"  - Loss: {outputs['loss']:.4f}")
    print(f"  - Attention shape: {outputs['attention'].shape}")
    print(f"  - Attention entropy: {outputs['attention_entropy']:.3f}")
    print(f"  - Slide features shape: {outputs['slide_feats'].shape}")
    print()

    # Attention collapse check
    if outputs['attention_entropy'] < 1.0:
        print("WARNING: Low attention entropy detected (<1.0)")
        print("   Consider increasing dropout or adding more regularization")
    else:
        print("Attention entropy is healthy")

    print()
    print("ABMIL TERT model working correctly!")
    print("=" * 60)