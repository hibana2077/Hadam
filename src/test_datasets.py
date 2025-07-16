"""
Example script to test dataset functionality and convert datasets to Parquet format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.utils import convert_dataset_to_parquet, create_data_loaders, get_dataset_statistics
from dataset.dataset_factory import get_dataset_info
import yaml


def load_config(config_path: str = "cfg.yml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def test_dataset(dataset_name: str, data_root: str = "./data"):
    """Test a specific dataset."""
    print(f"\n{'='*50}")
    print(f"Testing {dataset_name.upper()} Dataset")
    print(f"{'='*50}")
    
    try:
        # Get dataset info
        info = get_dataset_info(dataset_name)
        print(f"Dataset Info:")
        print(f"  - Number of classes: {info['num_classes']}")
        print(f"  - Input shape: {info['input_shape']}")
        print(f"  - Classes: {info['classes'][:5]}{'...' if len(str(info['classes'])) > 100 else ''}")
        
        # Convert to Parquet
        print(f"\nConverting {dataset_name} to Parquet format...")
        convert_dataset_to_parquet(dataset_name, data_root)
        
        # Create data loaders
        print(f"\nCreating data loaders...")
        train_loader, test_loader = create_data_loaders(
            dataset_name=dataset_name,
            data_root=data_root,
            batch_size=32,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
            use_parquet=True
        )
        
        # Test loading a batch
        print(f"\nTesting data loading...")
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        print(f"Train batch shape: {train_batch[0].shape}, {train_batch[1].shape}")
        print(f"Test batch shape: {test_batch[0].shape}, {test_batch[1].shape}")
        
        # Get statistics
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        
        print(f"\nDataset Statistics:")
        train_stats = get_dataset_statistics(train_dataset)
        test_stats = get_dataset_statistics(test_dataset)
        
        print(f"Training set: {train_stats['num_samples']} samples")
        print(f"Test set: {test_stats['num_samples']} samples")
        print(f"Classes: {train_stats['num_classes']}")
        
        if train_stats['data_mean'] is not None:
            print(f"Data mean: {[f'{x:.4f}' for x in train_stats['data_mean']]}")
            print(f"Data std: {[f'{x:.4f}' for x in train_stats['data_std']]}")
        
        print(f"✅ {dataset_name} dataset test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to test all datasets."""
    print("Dataset Testing and Conversion Script")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        print(f"Loaded configuration from cfg.yml")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        config = {'dataset': 'cifar10'}
    
    # Test datasets
    datasets_to_test = ['cifar10', 'cifar100', 'svhn', 'fashion_mnist']
    current_dataset = config.get('dataset', 'cifar10')
    
    print(f"Current dataset in config: {current_dataset}")
    
    # Test current dataset first
    if current_dataset in datasets_to_test:
        test_dataset(current_dataset)
        datasets_to_test.remove(current_dataset)
    
    # Ask user if they want to test other datasets
    print(f"\nDo you want to test other datasets? {datasets_to_test}")
    response = input("Enter 'y' to test all, 'n' to skip, or dataset names separated by comma: ").strip().lower()
    
    if response == 'y':
        for dataset_name in datasets_to_test:
            test_dataset(dataset_name)
    elif response != 'n':
        # Parse comma-separated dataset names
        selected_datasets = [name.strip() for name in response.split(',') if name.strip() in datasets_to_test]
        for dataset_name in selected_datasets:
            test_dataset(dataset_name)
    
    print(f"\n{'='*50}")
    print("Dataset testing completed!")
    print("All datasets are now available in Parquet format for HPC usage.")
    print("Benefits:")
    print("- Reduced file count (single file per train/test split)")
    print("- Faster loading times")
    print("- Compressed storage")
    print("- HPC-friendly format")


if __name__ == "__main__":
    main()
