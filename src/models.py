"""
Model creation utilities.
"""

import torch.nn as nn
import torchvision.models as models


def create_model(model_name: str, num_classes: int, input_shape: tuple) -> nn.Module:
    """Create a CNN model based on configuration."""
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        
        # Adjust first layer for different input shapes
        if input_shape[0] != 3:  # Not RGB
            model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Adjust for smaller images (like CIFAR, Fashion-MNIST)
        if input_shape[1] == 28 or input_shape[1] == 32:
            model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
