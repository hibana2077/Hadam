"""
Dataset utilities for data loading and management.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from .dataset_factory import get_dataset, get_dataset_info


def create_data_loaders(
    dataset_name: str,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_parquet: bool = True,
    custom_transform: Optional[transforms.Compose] = None,
    download: bool = True  # Enable auto-download by default
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_root: Root directory for dataset storage
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: If True, pin memory for faster GPU transfer
        use_parquet: If True, use Parquet format
        custom_transform: Custom transform to use instead of default
        download: If True, download dataset if not found (True by default)
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = get_dataset(
        name=dataset_name,
        root=data_root,
        train=True,
        transform=custom_transform,
        download=download,
        use_parquet=use_parquet
    )
    
    test_dataset = get_dataset(
        name=dataset_name,
        root=data_root,
        train=False,
        transform=custom_transform,
        download=download,
        use_parquet=use_parquet
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader


def get_dataset_statistics(dataset: Dataset) -> dict:
    """
    Calculate basic statistics for a dataset.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    # Sample some data to calculate statistics
    sample_size = min(1000, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]
    
    data_samples = []
    targets = []
    
    for idx in indices:
        data, target = dataset[idx]
        if isinstance(data, torch.Tensor):
            data_samples.append(data)
        targets.append(target)
    
    if data_samples:
        data_tensor = torch.stack(data_samples)
        mean = data_tensor.mean(dim=(0, 2, 3)) if data_tensor.dim() == 4 else data_tensor.mean(dim=0)
        std = data_tensor.std(dim=(0, 2, 3)) if data_tensor.dim() == 4 else data_tensor.std(dim=0)
    else:
        mean, std = None, None
    
    targets_tensor = torch.stack(targets) if isinstance(targets[0], torch.Tensor) else torch.tensor(targets)
    unique_targets = torch.unique(targets_tensor)
    
    stats = {
        'num_samples': len(dataset),
        'num_classes': len(unique_targets),
        'class_distribution': {int(cls): int((targets_tensor == cls).sum()) for cls in unique_targets},
        'data_mean': mean.tolist() if mean is not None else None,
        'data_std': std.tolist() if std is not None else None,
    }
    
    return stats


def convert_dataset_to_parquet(
    dataset_name: str,
    data_root: str = "./data",
    force_reconvert: bool = False
):
    """
    Convert a dataset to Parquet format.
    
    Args:
        dataset_name: Name of the dataset to convert
        data_root: Root directory for dataset storage
        force_reconvert: If True, reconvert even if Parquet files exist
    """
    print(f"Converting {dataset_name} to Parquet format...")
    
    # Check if already converted
    from pathlib import Path
    root_path = Path(data_root)
    train_parquet = root_path / f"{dataset_name}_train.parquet"
    test_parquet = root_path / f"{dataset_name}_test.parquet"
    
    if not force_reconvert and train_parquet.exists() and test_parquet.exists():
        print(f"{dataset_name} already converted to Parquet format.")
        return
    
    # Force conversion by creating datasets
    print("Converting training set...")
    train_dataset = get_dataset(
        name=dataset_name,
        root=data_root,
        train=True,
        download=True,
        use_parquet=True
    )
    
    print("Converting test set...")
    test_dataset = get_dataset(
        name=dataset_name,
        root=data_root,
        train=False,
        download=True,
        use_parquet=True
    )
    
    # Print statistics
    train_stats = get_dataset_statistics(train_dataset)
    test_stats = get_dataset_statistics(test_dataset)
    
    print(f"\n{dataset_name} conversion completed!")
    print(f"Training set: {train_stats['num_samples']} samples")
    print(f"Test set: {test_stats['num_samples']} samples")
    print(f"Number of classes: {train_stats['num_classes']}")
    print(f"Files saved at: {data_root}")


if __name__ == "__main__":
    # Example usage
    datasets_to_convert = ['cifar10', 'cifar100', 'svhn', 'fashion_mnist']
    
    for dataset_name in datasets_to_convert:
        try:
            convert_dataset_to_parquet(dataset_name)
        except Exception as e:
            print(f"Error converting {dataset_name}: {e}")
