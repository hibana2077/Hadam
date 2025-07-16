"""
Core training functions for model training.
"""

import torch
import torch.nn as nn
from typing import Tuple
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device: torch.device,
    enable_progress_bar: bool = True,
    curve_tracker=None,
    loss_tracker=None,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        enable_progress_bar: Whether to show progress bar
        curve_tracker: CurveTracker instance for logging curves
        loss_tracker: LossTracker instance for detailed loss logging
        epoch: Current epoch number
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    if enable_progress_bar:
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log batch loss if tracker is provided
        if loss_tracker is not None:
            loss_tracker.log_batch_loss(epoch, batch_idx, loss.item())
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        if enable_progress_bar:
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Log epoch metrics if tracker is provided
    if curve_tracker is not None:
        lr = get_learning_rate(optimizer)
        curve_tracker.add_train_metrics(epoch, avg_loss, accuracy, lr)
    
    return avg_loss, accuracy


def train_with_scheduler(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    device: torch.device,
    enable_progress_bar: bool = True,
    curve_tracker=None,
    loss_tracker=None,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Train the model for one epoch with learning rate scheduler.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        device: Device to use
        enable_progress_bar: Whether to show progress bar
        curve_tracker: CurveTracker instance for logging curves
        loss_tracker: LossTracker instance for detailed loss logging
        epoch: Current epoch number
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    avg_loss, accuracy = train_epoch(
        model, train_loader, optimizer, criterion, device, 
        enable_progress_bar, curve_tracker, loss_tracker, epoch
    )
    
    # Step scheduler
    scheduler.step()
    
    return avg_loss, accuracy


def get_learning_rate(optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
