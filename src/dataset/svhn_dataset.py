"""
SVHN dataset with Hugging Face Parquet format support.
"""

import numpy as np
from typing import Tuple
import torchvision.datasets as datasets
from .base_dataset import BaseDataset
from .hf_parquet_dataset import HFParquetDataset


class SVHNDataset(HFParquetDataset):
    """SVHN dataset with Hugging Face Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "svhn"
    
    @property
    def parquet_url(self) -> str:
        return "https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/svhn.parquet?download=true"


class SVHNDatasetLegacy(BaseDataset):
    """SVHN dataset with legacy Parquet format support."""
    
    @property
    def dataset_name(self) -> str:
        return "svhn"
    
    def _download_original(self) -> Tuple[np.ndarray, np.ndarray]:
        """Download original SVHN dataset."""
        # Use torchvision to download
        split = 'train' if self.train else 'test'
        dataset = datasets.SVHN(
            root=self.root / "temp",
            split=split,
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
        """Return SVHN class names (digits 0-9)."""
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
