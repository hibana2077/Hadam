"""
Complete training workflow utilities.
"""

from typing import Tuple
import torch
import torch.nn as nn
from .training import train_with_scheduler, train_epoch, get_learning_rate


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
    from .evaluation import evaluate_model
    from .curve_tracker import CurveTracker
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
