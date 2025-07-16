"""
Data augmentation utilities for training.
"""

import torch


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
    _, _, img_h, img_w = x.shape
    
    # Generate random bounding box
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = (img_w * cut_rat).int()
    cut_h = (img_h * cut_rat).int()
    
    cx = torch.randint(0, img_w, (1,))
    cy = torch.randint(0, img_h, (1,))
    
    bbx1 = torch.clamp(cx - cut_w // 2, 0, img_w)
    bby1 = torch.clamp(cy - cut_h // 2, 0, img_h)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, img_w)
    bby2 = torch.clamp(cy + cut_h // 2, 0, img_h)
    
    # Apply cutmix
    index = torch.randperm(batch_size)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exact area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_w * img_h))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam
