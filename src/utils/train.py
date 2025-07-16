"""
Main training module for Hadam optimizer experiments.
Re-exports all training functions for backward compatibility.
"""

# Import all training functions
from .training import (
    train_epoch,
    train_with_scheduler,
    get_learning_rate
)

# Import checkpoint utilities
from .checkpoint import (
    save_checkpoint,
    load_checkpoint
)

# Import complete training workflow
from .training_workflow import complete_training_run

# Re-export for backward compatibility
__all__ = [
    'train_epoch',
    'train_with_scheduler',
    'get_learning_rate',
    'save_checkpoint',
    'load_checkpoint',
    'complete_training_run'
]
