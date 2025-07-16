"""
Custom loss functions for neural network training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss implementation.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0.0 = no smoothing)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for label smoothing loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: True labels
        
        Returns:
            Smoothed loss
        """
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Create smooth target distribution
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.mean(torch.sum(-smooth_targets * log_probs, dim=1))
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for focal loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: True labels
        
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixupLoss(nn.Module):
    """
    Mixup loss implementation.
    """
    
    def __init__(self, criterion):
        """
        Initialize mixup loss.
        
        Args:
            criterion: Base loss function
        """
        super(MixupLoss, self).__init__()
        self.criterion = criterion
    
    def forward(self, predictions: torch.Tensor, targets_a: torch.Tensor, 
                targets_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Forward pass for mixup loss.
        
        Args:
            predictions: Model predictions
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixing parameter
        
        Returns:
            Mixed loss
        """
        return lam * self.criterion(predictions, targets_a) + \
               (1 - lam) * self.criterion(predictions, targets_b)
