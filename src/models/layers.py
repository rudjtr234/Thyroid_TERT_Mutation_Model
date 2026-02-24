# src/models/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def create_mlp(in_dim: int,
               hid_dims: List[int],
               dropout: float,
               out_dim: int,
               end_with_fc: bool = False) -> nn.Module:
    """
    Create a Multi-Layer Perceptron (MLP) with specified dimensions.

    Args:
        in_dim (int): Input dimension
        hid_dims (List[int]): List of hidden layer dimensions
        dropout (float): Dropout probability
        out_dim (int): Output dimension
        end_with_fc (bool): If True, end with only linear layer without activation

    Returns:
        nn.Module: Sequential MLP module
    """
    layers = []
    dims = [in_dim] + hid_dims + [out_dim]

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))

        # Add activation and dropout except for the last layer if end_with_fc is True
        if i < len(dims) - 2 or not end_with_fc:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class GlobalAttention(nn.Module):
    """
    Standard global attention mechanism for MIL.

    This implements attention pooling across instances in a bag.
    """

    def __init__(self, L: int, D: int, dropout: float = 0.0, num_classes: int = 1):
        """
        Initialize GlobalAttention.

        Args:
            L (int): Input feature dimension
            D (int): Hidden attention dimension
            dropout (float): Dropout probability
            num_classes (int): Number of attention classes (usually 1)
        """
        super().__init__()
        self.L = L
        self.D = D
        self.num_classes = num_classes

        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, num_classes)
        )

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention mechanism.

        Args:
            h (torch.Tensor): Input features [batch_size, num_instances, feature_dim]

        Returns:
            torch.Tensor: Attention scores [batch_size, num_instances, num_classes]
        """
        h = self.dropout(h)
        attention_scores = self.attention(h)  # [B, N, num_classes]
        return attention_scores


class GlobalGatedAttention(nn.Module):
    """
    Gated attention mechanism for MIL.

    This implements a more sophisticated attention mechanism with gating,
    which can learn more complex attention patterns.
    """

    def __init__(self, L: int, D: int, dropout: float = 0.0, num_classes: int = 1):
        """
        Initialize GlobalGatedAttention.

        Args:
            L (int): Input feature dimension
            D (int): Hidden attention dimension
            dropout (float): Dropout probability
            num_classes (int): Number of attention classes (usually 1)
        """
        super().__init__()
        self.L = L
        self.D = D
        self.num_classes = num_classes

        # Attention branch V (values)
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh()
        )

        # Attention branch U (gates)
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )

        # Final attention weights
        self.attention_w = nn.Linear(D, num_classes)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of gated attention mechanism.

        Args:
            h (torch.Tensor): Input features [batch_size, num_instances, feature_dim]

        Returns:
            torch.Tensor: Attention scores [batch_size, num_instances, num_classes]
        """
        h = self.dropout(h)

        # Compute attention components
        A_V = self.attention_V(h)  # [B, N, D] - values
        A_U = self.attention_U(h)  # [B, N, D] - gates

        # Element-wise multiplication (gating)
        attention_scores = self.attention_w(A_V * A_U)  # [B, N, num_classes]

        return attention_scores


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based MIL models.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        """
        Initialize positional encoding.

        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x (torch.Tensor): Input tensor [seq_len, batch_size, d_model]

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for transformer-based MIL models.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            mask (torch.Tensor, optional): Attention mask

        Returns:
            torch.Tensor: Attended features
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.w_o(attended)

        return output