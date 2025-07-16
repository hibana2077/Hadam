"""
Dataset factory for creating datasets based on configuration.
"""

from typing import Optional, Dict, Any
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .cifar_dataset import CIFAR10Dataset, CIFAR100Dataset
from .svhn_dataset import SVHNDataset
from .fashion_mnist_dataset import FashionMNISTDataset


def get_default_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """Get default transforms for a dataset."""
    
    if dataset_name in ['cifar10', 'cifar100']:
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    elif dataset_name == 'svhn':
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    elif dataset_name == 'fashion_mnist':
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
    
    else:
        # Default transforms
        return transforms.Compose([
            transforms.ToTensor(),
        ])


def get_dataset(
    name: str,
    root: str = "./data",
    train: bool = True,
    transform: Optional[transforms.Compose] = None,
    target_transform: Optional[transforms.Compose] = None,
    download: bool = True,
    **kwargs
) -> Dataset:
    """
    Create a dataset instance.
    
    Args:
        name: Dataset name ('cifar10', 'cifar100', 'svhn', 'fashion_mnist')
        root: Root directory for dataset storage
        train: If True, load training data, else test data
        transform: Transform to apply to images (if None, use default)
        target_transform: Transform to apply to targets
        download: If True, download dataset if not found
        **kwargs: Additional arguments for dataset
    
    Returns:
        Dataset instance
    """
    name = name.lower()
    
    # Use default transform if none provided
    if transform is None:
        transform = get_default_transforms(name, train)
    
    dataset_classes = {
        'cifar10': CIFAR10Dataset,
        'cifar100': CIFAR100Dataset,
        'svhn': SVHNDataset,
        'fashion_mnist': FashionMNISTDataset,
    }
    
    if name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_classes.keys())}")
    
    dataset_class = dataset_classes[name]
    
    return dataset_class(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        **kwargs
    )


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get information about a dataset."""
    name = name.lower()
    
    info = {
        'cifar10': {
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        },
        'cifar100': {
            'num_classes': 100,
            'input_shape': (3, 32, 32),
            'classes': 100  # Too many to list here
        },
        'svhn': {
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        },
        'fashion_mnist': {
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        }
    }
    
    if name not in info:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(info.keys())}")
    
    return info[name]
