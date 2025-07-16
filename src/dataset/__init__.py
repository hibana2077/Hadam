"""
Dataset module for Hadam optimizer experiments.
Supports CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST in Parquet format.
"""

from .base_dataset import BaseDataset
from .cifar_dataset import CIFAR10Dataset, CIFAR100Dataset
from .svhn_dataset import SVHNDataset
from .fashion_mnist_dataset import FashionMNISTDataset
from .dataset_factory import get_dataset

__all__ = [
    'BaseDataset',
    'CIFAR10Dataset', 
    'CIFAR100Dataset',
    'SVHNDataset',
    'FashionMNISTDataset',
    'get_dataset'
]
