# src/models/mil_template.py

import torch
import torch.nn as nn


class MIL(nn.Module):
    """
    Base class for Multiple Instance Learning models.

    This class provides common functionality and interface for all MIL models.
    """

    def __init__(self, in_dim: int, embed_dim: int, num_classes: int):
        """
        Initialize MIL base class.

        Args:
            in_dim (int): Input feature dimension
            embed_dim (int): Embedding dimension
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_classifier(self):
        """Initialize classifier layer weights."""
        if hasattr(self, 'classifier'):
            nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)

    @staticmethod
    def compute_loss(loss_fn: nn.Module, logits: torch.Tensor, label: torch.Tensor):
        """
        Compute loss using the provided loss function.

        Args:
            loss_fn (nn.Module): Loss function to use
            logits (torch.Tensor): Model predictions
            label (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss, or None if loss_fn or label is None
        """
        if loss_fn is None or label is None:
            return None
        return loss_fn(logits, label)

    def forward_features(self, h: torch.Tensor, **kwargs):
        """
        Extract features from input patches.
        Should be implemented by subclasses.

        Args:
            h (torch.Tensor): Input patch features

        Returns:
            torch.Tensor: Extracted features
        """
        raise NotImplementedError("Subclasses must implement forward_features")

    def forward_head(self, h: torch.Tensor):
        """
        Forward pass through classification head.
        Should be implemented by subclasses.

        Args:
            h (torch.Tensor): Features to classify

        Returns:
            torch.Tensor: Classification logits
        """
        raise NotImplementedError("Subclasses must implement forward_head")

    def forward(self, h: torch.Tensor, **kwargs):
        """
        Full forward pass.
        Should be implemented by subclasses.

        Args:
            h (torch.Tensor): Input patch features

        Returns:
            Tuple of results and logs
        """
        raise NotImplementedError("Subclasses must implement forward")
