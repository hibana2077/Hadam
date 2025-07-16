"""
CIFAR-10 and CIFAR-100 datasets with Hugging Face Parquet format support.
"""

import numpy as np
from typing import Tuple
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .base_dataset import BaseDataset
from .hf_parquet_dataset import HFParquetDataset


class CIFAR10Dataset(HFParquetDataset):
    """CIFAR-10 dataset with Hugging Face Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "cifar10"
    
    @property
    def parquet_url(self) -> str:
        return "https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/cifar10.parquet?download=true"


class CIFAR10DatasetLegacy(BaseDataset):
    """CIFAR-10 dataset with legacy Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "cifar10"
    
    def _download_original(self) -> Tuple[np.ndarray, np.ndarray]:
        """Download original CIFAR-10 dataset."""
        # Use torchvision to download
        dataset = datasets.CIFAR10(
            root=self.root / "temp",
            train=self.train,
            download=True,
            transform=None
        )
        
        # Extract data and targets
        data = np.array([np.array(img) for img, _ in dataset])
        targets = np.array([target for _, target in dataset])
        
        # Clean up temporary files
        import shutil
        temp_dir = self.root / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return data, targets
    
    def get_classes(self) -> list:
        """Return CIFAR-10 class names."""
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]


class CIFAR100Dataset(HFParquetDataset):
    """CIFAR-100 dataset with Hugging Face Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "cifar100"
    
    @property
    def parquet_url(self) -> str:
        return "https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/cifar100.parquet?download=true"


class CIFAR100DatasetLegacy(BaseDataset):
    """CIFAR-100 dataset with legacy Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "cifar100"
    
    def _download_original(self) -> Tuple[np.ndarray, np.ndarray]:
        """Download original CIFAR-100 dataset."""
        # Use torchvision to download
        dataset = datasets.CIFAR100(
            root=self.root / "temp",
            train=self.train,
            download=True,
            transform=None
        )
        
        # Extract data and targets
        data = np.array([np.array(img) for img, _ in dataset])
        targets = np.array([target for _, target in dataset])
        
        # Clean up temporary files
        import shutil
        temp_dir = self.root / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return data, targets
    
    def get_classes(self) -> list:
        """Return CIFAR-100 class names."""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
