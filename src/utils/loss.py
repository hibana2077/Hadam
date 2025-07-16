"""
Main loss module for Hadam optimizer experiments.
Re-exports all loss functions for backward compatibility.
"""

# Import all loss functions
from .loss_functions import (
    LabelSmoothingLoss,
    FocalLoss,
    MixupLoss
)

# Import loss tracking
from .loss_tracker import LossTracker

# Import data augmentation
from .augmentation import mixup_data, cutmix_data

# Import utilities
import torch.nn as nn


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


# Re-export for backward compatibility
__all__ = [
    'LabelSmoothingLoss',
    'FocalLoss', 
    'MixupLoss',
    'LossTracker',
    'get_loss_function',
    'mixup_data',
    'cutmix_data'
]
