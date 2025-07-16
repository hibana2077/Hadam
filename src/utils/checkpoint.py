"""
Checkpoint utilities for saving and loading model states.
"""

import torch
import torch.nn as nn
from typing import Tuple


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    filepath: str,
    device: torch.device
) -> Tuple[int, float, float]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load to
    
    Returns:
        Tuple of (epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    
    return epoch, loss, accuracy
