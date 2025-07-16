"""
Base dataset class for converting datasets to Parquet format.
"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BaseDataset(Dataset, ABC):
    """Base class for all datasets with Parquet format support."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: bool = True,
        use_parquet: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory for dataset
            train: If True, load training data, else test data
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            download: If True, download dataset if not found
            use_parquet: If True, use/create Parquet format
        """
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.use_parquet = use_parquet
        
        # Create data directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Define parquet paths
        split = "train" if train else "test"
        self.parquet_path = self.root / f"{self.dataset_name}_{split}.parquet"
        
        # Load or create dataset
        if self.use_parquet and self.parquet_path.exists():
            self._load_from_parquet()
        else:
            if download:
                self._download_and_convert()
            else:
                raise FileNotFoundError(f"Dataset not found at {self.parquet_path}")
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        pass
    
    @abstractmethod
    def _download_original(self) -> Tuple[np.ndarray, np.ndarray]:
        """Download original dataset and return data, targets."""
        pass
    
    def _download_and_convert(self):
        """Download original dataset and convert to Parquet."""
        print(f"Downloading and converting {self.dataset_name} to Parquet format...")
        
        # Download original data
        data, targets = self._download_original()
        
        # Convert to DataFrame
        if data.ndim == 4:  # Images: (N, H, W, C) or (N, C, H, W)
            # Flatten images for storage
            if data.shape[1] == 3 or data.shape[1] == 1:  # Channel first
                data = data.transpose(0, 2, 3, 1)  # Convert to channel last
            
            n_samples, h, w, c = data.shape
            flattened_data = data.reshape(n_samples, -1)
            
            # Create column names
            columns = [f"pixel_{i}" for i in range(flattened_data.shape[1])]
            df = pd.DataFrame(flattened_data, columns=columns)
            
            # Add metadata
            df['height'] = h
            df['width'] = w
            df['channels'] = c
        else:
            # For other data formats
            columns = [f"feature_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=columns)
        
        # Add targets
        df['target'] = targets
        
        # Save to Parquet
        df.to_parquet(self.parquet_path, compression='snappy')
        print(f"Dataset saved to {self.parquet_path}")
        
        # Load the data
        self._load_from_parquet()
    
    def _load_from_parquet(self):
        """Load data from Parquet file."""
        df = pd.read_parquet(self.parquet_path)
        
        # Extract targets
        self.targets = df['target'].values
        
        # Extract image data
        if 'height' in df.columns:
            # Image data
            h, w, c = df['height'].iloc[0], df['width'].iloc[0], df['channels'].iloc[0]
            pixel_cols = [col for col in df.columns if col.startswith('pixel_')]
            pixel_data = df[pixel_cols].values
            self.data = pixel_data.reshape(-1, h, w, c)
        else:
            # Feature data
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            self.data = df[feature_cols].values
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        data, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image if it's image data
        if data.ndim == 3:  # Image
            # Convert to PIL Image for transforms
            from PIL import Image
            if data.dtype != np.uint8:
                data = (data * 255).astype(np.uint8)
            
            if data.shape[2] == 1:  # Grayscale
                data = data.squeeze(2)
                img = Image.fromarray(data, mode='L')
            else:  # RGB
                img = Image.fromarray(data, mode='RGB')
            
            if self.transform is not None:
                img = self.transform(img)
        else:
            # Feature data
            img = torch.from_numpy(data).float()
            if self.transform is not None:
                img = self.transform(img)
        
        target = torch.tensor(target, dtype=torch.long)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_classes(self) -> list:
        """Return list of class names."""
        return list(range(len(np.unique(self.targets))))
