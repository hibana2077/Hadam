"""
Fashion-MNIST dataset with Hugging Face Parquet format support.
"""

import numpy as np
from typing import Tuple
import torchvision.datasets as datasets
from .base_dataset import BaseDataset
from .hf_parquet_dataset import HFParquetDataset


class FashionMNISTDataset(HFParquetDataset):
    """Fashion-MNIST dataset with Hugging Face Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "fashion_mnist"
    
    @property
    def parquet_url(self) -> str:
        return "https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/fashion_mnist.parquet?download=true"


class FashionMNISTDatasetLegacy(BaseDataset):
    """Fashion-MNIST dataset with legacy Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "fashion_mnist"
    
    def _download_original(self) -> Tuple[np.ndarray, np.ndarray]:
        """Download original Fashion-MNIST dataset."""
        # Use torchvision to download
        dataset = datasets.FashionMNIST(
            root=self.root / "temp",
            train=self.train,
            download=True,
            transform=None
        )
        
        # Extract data and targets
        data = np.array([np.array(img) for img, _ in dataset])
        # Add channel dimension for grayscale images
        data = np.expand_dims(data, axis=-1)
        targets = np.array([target for _, target in dataset])
        
        # Clean up temporary files
        import shutil
        temp_dir = self.root / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return data, targets
    
    def get_classes(self) -> list:
        """Return Fashion-MNIST class names."""
        return [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
