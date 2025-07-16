"""
Hugging Face Parquet dataset base class for direct parquet loading.
"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import requests
from io import BytesIO
from PIL import Image


class HFParquetDataset(Dataset, ABC):
    """Base class for datasets loading from Hugging Face parquet files."""
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory for dataset
            train: If True, load training data, else test data
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            download: If True, download dataset if not found
        """
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # Create data directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Define parquet path
        self.parquet_path = self.root / f"{self.dataset_name}.parquet"
        
        # Load or download dataset
        if self.parquet_path.exists():
            self._load_from_parquet()
        else:
            if download:
                try:
                    self._download_parquet()
                    self._load_from_parquet()
                except Exception as e:
                    # Clean up partially downloaded file if it exists
                    if self.parquet_path.exists():
                        self.parquet_path.unlink()
                    raise RuntimeError(f"Failed to download dataset: {e}")
            else:
                raise FileNotFoundError(f"Dataset not found at {self.parquet_path}. Set download=True to download automatically.")
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        pass
    
    @property
    @abstractmethod
    def parquet_url(self) -> str:
        """Return the URL to download the parquet file."""
        pass
    
    def _download_parquet(self):
        """Download parquet file from Hugging Face."""
        print(f"Downloading {self.dataset_name} from Hugging Face...")
        print(f"URL: {self.parquet_url}")
        print(f"Target: {self.parquet_path}")
        
        try:
            response = requests.get(self.parquet_url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(self.parquet_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Show progress for large files
                        if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded_size / total_size) * 100
                            print(f"Progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f} MB)")
            
            print(f"Dataset downloaded successfully to {self.parquet_path}")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error while downloading dataset: {e}")
        except IOError as e:
            raise RuntimeError(f"File I/O error while downloading dataset: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while downloading dataset: {e}")
    
    def _load_from_parquet(self):
        """Load data from Hugging Face parquet file."""
        df = pd.read_parquet(self.parquet_path)
        
        # Filter by split
        split = "train" if self.train else "test"
        df = df[df['split'] == split].reset_index(drop=True)
        
        # Extract data
        self.data = []
        self.targets = []
        self.class_names = []
        
        for idx, row in df.iterrows():
            # Load image from bytes
            image_bytes = row['image']
            image = Image.open(BytesIO(image_bytes))
            # Convert to numpy array
            img_array = np.array(image)
            
            self.data.append(img_array)
            self.targets.append(row['label'])
            self.class_names.append(row['class_name'])
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        
        print(f"Loaded {len(self.data)} {split} samples")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        data, target = self.data[idx], self.targets[idx]
        
        # Convert numpy array to PIL Image
        if data.dtype != np.uint8:
            data = (data * 255).astype(np.uint8)
        
        if data.ndim == 2:  # Grayscale
            img = Image.fromarray(data, mode='L')
        elif data.ndim == 3 and data.shape[2] == 1:  # Grayscale with channel
            img = Image.fromarray(data.squeeze(2), mode='L')
        elif data.ndim == 3 and data.shape[2] == 3:  # RGB
            img = Image.fromarray(data, mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {data.shape}")
        
        if self.transform is not None:
            img = self.transform(img)
        
        target = torch.tensor(target, dtype=torch.long)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_classes(self) -> list:
        """Return list of unique class names."""
        return sorted(list(set(self.class_names)))
    
    def get_class_name(self, idx: int) -> str:
        """Get class name for a given index."""
        return self.class_names[idx]
