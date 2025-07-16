"""
Curve tracking utilities for training experiments.
"""

import os
import zipfile
from datetime import datetime
from typing import List, Dict


class CurveTracker:
    """
    Track training and evaluation curves.
    """
    
    def __init__(self, optimizer_name: str, dataset_name: str):
        """
        Initialize curve tracker.
        
        Args:
            optimizer_name: Name of optimizer (e.g., 'hadam', 'sgd', 'adam')
            dataset_name: Name of dataset (e.g., 'cifar10', 'cifar100')
        """
        self.optimizer_name = optimizer_name
        self.dataset_name = dataset_name
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize tracking lists
        self.train_losses = []
        self.train_accuracies = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.epochs = []
        self.learning_rates = []
        
    def add_train_metrics(self, epoch: int, loss: float, accuracy: float, lr: float = None):
        """Add training metrics for an epoch."""
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def add_eval_metrics(self, loss: float, accuracy: float):
        """Add evaluation metrics for an epoch."""
        self.eval_losses.append(loss)
        self.eval_accuracies.append(accuracy)
    
    def save_curves_to_txt(self, output_dir: str = "results"):
        """
        Save curves to individual txt files.
        
        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = f"{self.optimizer_name}-{self.dataset_name}-{self.date}"
        
        # Save train loss curve
        train_loss_file = os.path.join(output_dir, f"{base_filename}-train_loss.txt")
        with open(train_loss_file, 'w') as f:
            f.write("# Train Loss Curve\n")
            f.write("# Epoch\tLoss\n")
            for epoch, loss in zip(self.epochs, self.train_losses):
                f.write(f"{epoch}\t{loss:.6f}\n")
        
        # Save train accuracy curve
        train_acc_file = os.path.join(output_dir, f"{base_filename}-train_acc.txt")
        with open(train_acc_file, 'w') as f:
            f.write("# Train Accuracy Curve\n")
            f.write("# Epoch\tAccuracy\n")
            for epoch, acc in zip(self.epochs, self.train_accuracies):
                f.write(f"{epoch}\t{acc:.4f}\n")
        
        # Save eval loss curve
        eval_loss_file = os.path.join(output_dir, f"{base_filename}-eval_loss.txt")
        with open(eval_loss_file, 'w') as f:
            f.write("# Eval Loss Curve\n")
            f.write("# Epoch\tLoss\n")
            for epoch, loss in zip(self.epochs, self.eval_losses):
                f.write(f"{epoch}\t{loss:.6f}\n")
        
        # Save eval accuracy curve
        eval_acc_file = os.path.join(output_dir, f"{base_filename}-eval_acc.txt")
        with open(eval_acc_file, 'w') as f:
            f.write("# Eval Accuracy Curve\n")
            f.write("# Epoch\tAccuracy\n")
            for epoch, acc in zip(self.epochs, self.eval_accuracies):
                f.write(f"{epoch}\t{acc:.4f}\n")
        
        # Save learning rate curve if available
        if self.learning_rates:
            lr_file = os.path.join(output_dir, f"{base_filename}-learning_rate.txt")
            with open(lr_file, 'w') as f:
                f.write("# Learning Rate Curve\n")
                f.write("# Epoch\tLearning_Rate\n")
                for epoch, lr in zip(self.epochs, self.learning_rates):
                    f.write(f"{epoch}\t{lr:.8f}\n")
        
        # Save summary
        summary_file = os.path.join(output_dir, f"{base_filename}-summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"# Experiment Summary\n")
            f.write(f"Optimizer: {self.optimizer_name}\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Date: {self.date}\n")
            f.write(f"Total Epochs: {len(self.epochs)}\n")
            
            if self.train_losses:
                f.write(f"Best Train Loss: {min(self.train_losses):.6f}\n")
                f.write(f"Best Train Accuracy: {max(self.train_accuracies):.4f}%\n")
            
            if self.eval_losses:
                f.write(f"Best Eval Loss: {min(self.eval_losses):.6f}\n")
                f.write(f"Best Eval Accuracy: {max(self.eval_accuracies):.4f}%\n")
        
        return output_dir, base_filename
    
    def save_curves_to_zip(self, output_dir: str = "results"):
        """
        Save all curve files to a zip archive.
        
        Args:
            output_dir: Directory to save files
        """
        # First save to txt files
        txt_dir, base_filename = self.save_curves_to_txt(output_dir)
        
        # Create zip file
        zip_filename = os.path.join(output_dir, f"{base_filename}.zip")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all txt files to zip
            for file in os.listdir(txt_dir):
                if file.startswith(base_filename) and file.endswith('.txt'):
                    file_path = os.path.join(txt_dir, file)
                    zipf.write(file_path, file)
        
        print(f"Curves saved to: {zip_filename}")
        return zip_filename
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from training."""
        best_metrics = {}
        
        if self.train_losses:
            best_metrics['best_train_loss'] = min(self.train_losses)
            best_metrics['best_train_acc'] = max(self.train_accuracies)
            best_metrics['best_train_loss_epoch'] = self.epochs[self.train_losses.index(min(self.train_losses))]
            best_metrics['best_train_acc_epoch'] = self.epochs[self.train_accuracies.index(max(self.train_accuracies))]
        
        if self.eval_losses:
            best_metrics['best_eval_loss'] = min(self.eval_losses)
            best_metrics['best_eval_acc'] = max(self.eval_accuracies)
            best_metrics['best_eval_loss_epoch'] = self.epochs[self.eval_losses.index(min(self.eval_losses))]
            best_metrics['best_eval_acc_epoch'] = self.epochs[self.eval_accuracies.index(max(self.eval_accuracies))]
        
        return best_metrics
