"""
Loss functions for Hadam optimizer experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from datetime import datetime


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
            lam: Mixup parameter
        
        Returns:
            Mixup loss
        """
        return lam * self.criterion(predictions, targets_a) + \
               (1 - lam) * self.criterion(predictions, targets_b)


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'crossentropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    
    elif loss_name == 'labelsmoothing' or loss_name == 'label_smoothing':
        return LabelSmoothingLoss(**kwargs)
    
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    
    elif loss_name == 'mse':
        return nn.MSELoss(**kwargs)
    
    elif loss_name == 'mae' or loss_name == 'l1':
        return nn.L1Loss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Data augmentation with mixup
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Apply mixup augmentation to data.
    
    Args:
        x: Input data
        y: Target labels
        alpha: Mixup parameter
    
    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Apply CutMix augmentation to data.
    
    Args:
        x: Input data
        y: Target labels
        alpha: CutMix parameter
    
    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(1. - lam)
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()
    
    # Uniform sampling
    cx = torch.randint(W, (1,))
    cy = torch.randint(H, (1,))
    
    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


class LossTracker:
    """
    Track and log loss values during training.
    """
    
    def __init__(self, optimizer_name: str, dataset_name: str):
        """
        Initialize loss tracker.
        
        Args:
            optimizer_name: Name of optimizer
            dataset_name: Name of dataset
        """
        self.optimizer_name = optimizer_name
        self.dataset_name = dataset_name
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Track batch-level losses for detailed analysis
        self.batch_losses = []
        self.batch_epochs = []
        self.batch_indices = []
        
    def log_batch_loss(self, epoch: int, batch_idx: int, loss: float):
        """Log loss for a specific batch."""
        self.batch_losses.append(loss)
        self.batch_epochs.append(epoch)
        self.batch_indices.append(batch_idx)
    
    def save_detailed_loss_log(self, output_dir: str = "results"):
        """Save detailed batch-level loss log."""
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = f"{self.optimizer_name}-{self.dataset_name}-{self.date}"
        detailed_loss_file = os.path.join(output_dir, f"{base_filename}-detailed_loss.txt")
        
        with open(detailed_loss_file, 'w') as f:
            f.write("# Detailed Batch Loss Log\n")
            f.write("# Epoch\tBatch_Index\tLoss\n")
            for epoch, batch_idx, loss in zip(self.batch_epochs, self.batch_indices, self.batch_losses):
                f.write(f"{epoch}\t{batch_idx}\t{loss:.6f}\n")
        
        return detailed_loss_file
