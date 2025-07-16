"""
Dataset module for Hadam optimizer experiments.
Supports CIFAR-10, CIFAR-100, SVHN, and Fashion-MNIST in Parquet format.
"""

from .base_dataset import BaseDataset
from .hf_parquet_dataset import HFParquetDataset
from .cifar_dataset import CIFAR10Dataset, CIFAR100Dataset, CIFAR10DatasetLegacy, CIFAR100DatasetLegacy
from .svhn_dataset import SVHNDataset, SVHNDatasetLegacy
from .fashion_mnist_dataset import FashionMNISTDataset, FashionMNISTDatasetLegacy
from .dataset_factory import get_dataset, get_dataset_info

__all__ = [
    'BaseDataset',
    'HFParquetDataset',
    'CIFAR10Dataset', 
    'CIFAR100Dataset',
    'CIFAR10DatasetLegacy',
    'CIFAR100DatasetLegacy',
    'SVHNDataset',
    'SVHNDatasetLegacy',
    'FashionMNISTDataset',
    'FashionMNISTDatasetLegacy',
    'get_dataset',
    'get_dataset_info'
]
