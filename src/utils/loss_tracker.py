"""
Loss tracking utilities for detailed training monitoring.
"""

import os
import json
from datetime import datetime
from typing import List


class LossTracker:
    """
    Track detailed loss information during training.
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
        
        # Track batch-level losses
        self.batch_losses = {}  # epoch -> list of batch losses
        self.epoch_stats = {}   # epoch -> {'mean': float, 'std': float, 'min': float, 'max': float}
    
    def log_batch_loss(self, epoch: int, batch_idx: int, loss: float):
        """Log a single batch loss."""
        if epoch not in self.batch_losses:
            self.batch_losses[epoch] = []
        
        self.batch_losses[epoch].append({
            'batch_idx': batch_idx,
            'loss': loss
        })
    
    def finalize_epoch(self, epoch: int):
        """Calculate epoch statistics."""
        if epoch in self.batch_losses:
            losses = [b['loss'] for b in self.batch_losses[epoch]]
            
            import statistics
            self.epoch_stats[epoch] = {
                'mean': statistics.mean(losses),
                'std': statistics.stdev(losses) if len(losses) > 1 else 0.0,
                'min': min(losses),
                'max': max(losses),
                'count': len(losses)
            }
    
    def save_detailed_loss_log(self, output_dir: str = "results"):
        """
        Save detailed loss information to JSON file.
        
        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        base_filename = f"{self.optimizer_name}-{self.dataset_name}-{self.date}"
        loss_file = os.path.join(output_dir, f"{base_filename}-detailed_loss.json")
        
        # Finalize all epochs
        for epoch in self.batch_losses:
            if epoch not in self.epoch_stats:
                self.finalize_epoch(epoch)
        
        loss_data = {
            'optimizer': self.optimizer_name,
            'dataset': self.dataset_name,
            'date': self.date,
            'batch_losses': self.batch_losses,
            'epoch_stats': self.epoch_stats
        }
        
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        print(f"Detailed loss log saved to: {loss_file}")
        return loss_file
