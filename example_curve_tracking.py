#!/usr/bin/env python3
"""
Example script demonstrating the new curve tracking functionality.
This script shows how to use the enhanced training utilities to collect
loss curves and accuracy curves, and save them as zip and txt files.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.eval import CurveTracker
from utils.loss import LossTracker
from utils.train import complete_training_run


def create_dummy_model():
    """Create a simple dummy model for demonstration."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def create_dummy_data():
    """Create dummy data loaders for demonstration."""
    # Create fake MNIST-like data
    train_data = torch.randn(1000, 1, 28, 28)
    train_labels = torch.randint(0, 10, (1000,))
    test_data = torch.randn(200, 1, 28, 28)
    test_labels = torch.randint(0, 10, (200,))
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def example_manual_tracking():
    """Example of manual curve tracking."""
    print("=== Manual Curve Tracking Example ===")
    
    # Initialize trackers
    curve_tracker = CurveTracker("sgd", "mnist")
    loss_tracker = LossTracker("sgd", "mnist")
    
    # Simulate training data
    for epoch in range(1, 6):
        # Simulate decreasing loss and increasing accuracy
        train_loss = 2.0 - (epoch * 0.3) + torch.rand(1).item() * 0.1
        train_acc = 20.0 + (epoch * 15.0) + torch.rand(1).item() * 5.0
        eval_loss = train_loss + 0.1 + torch.rand(1).item() * 0.05
        eval_acc = train_acc - 2.0 + torch.rand(1).item() * 2.0
        lr = 0.01 * (0.9 ** (epoch - 1))
        
        # Add metrics to trackers
        curve_tracker.add_train_metrics(epoch, train_loss, train_acc, lr)
        curve_tracker.add_eval_metrics(eval_loss, eval_acc)
        
        # Simulate batch losses
        for batch_idx in range(10):
            batch_loss = train_loss + torch.rand(1).item() * 0.2
            loss_tracker.log_batch_loss(epoch, batch_idx, batch_loss)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")
    
    # Save curves
    print("\nSaving curves...")
    zip_file = curve_tracker.save_curves_to_zip("example_results")
    loss_tracker.save_detailed_loss_log("example_results")
    
    # Get best metrics
    best_metrics = curve_tracker.get_best_metrics()
    print("\nBest Metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {zip_file}")


def example_complete_training():
    """Example using the complete training run function."""
    print("\n=== Complete Training Run Example ===")
    
    # Setup
    device = torch.device('cpu')  # Use CPU for this example
    model = create_dummy_model()
    train_loader, test_loader = create_dummy_data()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training with complete_training_run...")
    
    # Use complete training run
    best_metrics, zip_file_path = complete_training_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=3,  # Small number for demo
        optimizer_name="sgd",
        dataset_name="dummy_mnist",
        scheduler=None,
        enable_progress_bar=True,
        save_curves=True,
        output_dir="example_results"
    )
    
    print("\nBest Metrics from Complete Training:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")
    
    if zip_file_path:
        print(f"\nResults saved to: {zip_file_path}")


def main():
    """Main function."""
    print("Curve Tracking Examples for Hadam Optimizer")
    print("=" * 50)
    
    # Create results directory
    Path("example_results").mkdir(exist_ok=True)
    
    # Run examples
    example_manual_tracking()
    example_complete_training()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check the 'example_results' folder for generated files:")
    print("- Individual .txt files for each curve")
    print("- .zip files containing all curves")
    print("- Summary files with experiment details")
    
    # List generated files
    results_dir = Path("example_results")
    if results_dir.exists():
        print(f"\nGenerated files:")
        for file in sorted(results_dir.glob("*")):
            print(f"  {file.name}")


if __name__ == "__main__":
    main()
