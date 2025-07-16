"""
Training utilities for Hadam optimizer experiments.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from tqdm import tqdm
import os
from datetime import datetime


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


def complete_training_run(
    model: nn.Module,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device: torch.device,
    epochs: int,
    optimizer_name: str,
    dataset_name: str,
    scheduler=None,
    enable_progress_bar: bool = True,
    save_curves: bool = True,
    output_dir: str = "results"
) -> Tuple[dict, str]:
    """
    Complete training run with curve tracking and automatic saving.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epochs: Number of epochs to train
        optimizer_name: Name of optimizer for file naming
        dataset_name: Name of dataset for file naming
        scheduler: Learning rate scheduler (optional)
        enable_progress_bar: Whether to show progress bars
        save_curves: Whether to save curves to files
        output_dir: Directory to save results
    
    Returns:
        Tuple of (best_metrics_dict, zip_file_path)
    """
    # Import here to avoid circular imports
    from .eval import CurveTracker, evaluate_model
    from .loss import LossTracker
    
    # Initialize trackers
    curve_tracker = CurveTracker(optimizer_name, dataset_name)
    loss_tracker = LossTracker(optimizer_name, dataset_name)
    
    print(f"Starting training: {optimizer_name} on {dataset_name}")
    print(f"Training for {epochs} epochs...")
    
    best_test_acc = 0.0
    best_test_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        # Training
        if scheduler is not None:
            train_loss, train_acc = train_with_scheduler(
                model, train_loader, optimizer, criterion, scheduler,
                device, enable_progress_bar, curve_tracker, loss_tracker, epoch
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                enable_progress_bar, curve_tracker, loss_tracker, epoch
            )
        
        # Evaluation
        test_loss, test_acc = evaluate_model(
            model, test_loader, criterion, device, enable_progress_bar
        )
        
        # Add evaluation metrics to tracker
        curve_tracker.add_eval_metrics(test_loss, test_acc)
        
        # Track best metrics
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        # Print epoch summary
        lr = get_learning_rate(optimizer)
        print(f"Epoch {epoch:3d} | LR: {lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Get best metrics
    best_metrics = curve_tracker.get_best_metrics()
    best_metrics['final_test_acc'] = test_acc
    best_metrics['final_test_loss'] = test_loss
    best_metrics['best_test_acc_overall'] = best_test_acc
    best_metrics['best_test_loss_overall'] = best_test_loss
    
    print(f"\n=== Training Complete ===")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Save curves and create zip
    zip_file_path = None
    if save_curves:
        try:
            zip_file_path = curve_tracker.save_curves_to_zip(output_dir)
            loss_tracker.save_detailed_loss_log(output_dir)
            print(f"Results saved to: {zip_file_path}")
        except Exception as e:
            print(f"Warning: Could not save curves: {e}")
    
    return best_metrics, zip_file_path
