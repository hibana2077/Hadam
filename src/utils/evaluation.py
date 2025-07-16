"""
Core evaluation functions for model testing.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from tqdm import tqdm
import numpy as np


def evaluate_model(
    model: nn.Module,
    test_loader,
    criterion,
    device: torch.device,
    enable_progress_bar: bool = True
) -> Tuple[float, float]:
    """
    Evaluate the model on test data.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
        enable_progress_bar: Whether to show progress bar
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    if enable_progress_bar:
        pbar = tqdm(test_loader, desc="Evaluating")
    else:
        pbar = test_loader
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
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
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate_per_class(
    model: nn.Module,
    test_loader,
    device: torch.device,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate model performance per class.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        class_names: List of class names
    
    Returns:
        Dictionary with per-class accuracies
    """
    model.eval()
    
    # Initialize counters
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Count correct predictions per class
            for i in range(len(target)):
                label = target[i].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                
                class_total[label] += 1
                if pred[i] == target[i]:
                    class_correct[label] += 1
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for class_id in class_correct:
        accuracy = 100.0 * class_correct[class_id] / class_total[class_id]
        
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class_{class_id}"
        
        class_accuracies[class_name] = accuracy
    
    return class_accuracies


def calculate_confusion_matrix(
    model: nn.Module,
    test_loader,
    device: torch.device,
    num_classes: int
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        num_classes: Number of classes
    
    Returns:
        Confusion matrix as numpy array
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1)
            
            for i in range(len(target)):
                true_label = target[i].item()
                pred_label = pred[i].item()
                confusion_matrix[true_label][pred_label] += 1
    
    return confusion_matrix


def calculate_top_k_accuracy(
    model: nn.Module,
    test_loader,
    device: torch.device,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        k: k for top-k accuracy
    
    Returns:
        Top-k accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            _, pred = output.topk(k, 1, True, True)
            pred = pred.t()
            correct_k = pred.eq(target.view(1, -1).expand_as(pred))
            
            correct += correct_k[:k].reshape(-1).float().sum(0).item()
            total += target.size(0)
    
    return 100.0 * correct / total


def get_model_predictions(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> Tuple[List[int], List[int], List[torch.Tensor]]:
    """
    Get all model predictions and true labels.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu())
    
    return all_predictions, all_true_labels, all_probabilities
