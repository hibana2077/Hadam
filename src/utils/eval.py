"""
Main evaluation module for Hadam optimizer experiments.
Re-exports all evaluation functions for backward compatibility.
"""

# Import all evaluation functions
from .evaluation import (
    evaluate_model,
    evaluate_per_class,
    calculate_confusion_matrix,
    calculate_top_k_accuracy,
    get_model_predictions
)

# Import curve tracking
from .curve_tracker import CurveTracker

# Re-export for backward compatibility
__all__ = [
    'evaluate_model',
    'evaluate_per_class', 
    'calculate_confusion_matrix',
    'calculate_top_k_accuracy',
    'get_model_predictions',
    'CurveTracker'
]
