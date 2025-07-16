#!/usr/bin/env python3
"""
Example script demonstrating the usage of Hadam dataset system.
This script shows both download and training modes.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def update_config(config_updates: dict, config_file: str = "src/cfg.yml"):
    """Update configuration file with new values."""
    config_path = Path(config_file)
    
    # Load current config
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update with new values
    config.update(config_updates)
    
    # Save updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Updated configuration:")
    for key, value in config_updates.items():
        print(f"  {key}: {value}")

def run_download_mode(dataset: str = "cifar10"):
    """Run in download mode to prepare datasets."""
    print(f"\n{'='*50}")
    print(f"DOWNLOAD MODE: Preparing {dataset}")
    print(f"{'='*50}")
    
    # Update config for download mode
    update_config({
        'dataset': dataset,
        'only_download_dataset': True,
        'train': False,
        'eval': False
    })
    
    # Run main.py
    os.system(f"cd src && python main.py")
    
    # Check if files were created
    data_dir = Path("src/data")
    train_file = data_dir / f"{dataset}_train.parquet"
    test_file = data_dir / f"{dataset}_test.parquet"
    
    if train_file.exists() and test_file.exists():
        print(f"✅ {dataset} dataset successfully prepared!")
        print(f"   Train file: {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"   Test file: {test_file} ({test_file.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"❌ Failed to prepare {dataset} dataset")

def run_training_mode(dataset: str = "cifar10", epochs: int = 2):
    """Run in training mode."""
    print(f"\n{'='*50}")
    print(f"TRAINING MODE: {dataset}")
    print(f"{'='*50}")
    
    # Update config for training mode
    update_config({
        'dataset': dataset,
        'only_download_dataset': False,
        'train': True,
        'eval': True,
        'epochs': epochs,
        'enable_progress_bar': True
    })
    
    # Run main.py
    os.system(f"cd src && python main.py")

def main():
    """Main function for demonstration."""
    print("Hadam Dataset System Demonstration")
    print("=" * 50)
    
    # Example datasets to test
    datasets_to_test = ["cifar10"]  # Start with one dataset
    
    print("\nThis script will demonstrate:")
    print("1. Download mode: Prepare datasets for HPC")
    print("2. Training mode: Train with prepared datasets")
    print("\nFor HPC usage:")
    print("- Run step 1 on a machine with internet")
    print("- Copy data/ folder to HPC")
    print("- Run step 2 on HPC without internet")
    
    # Ask user what to do
    choice = input("\nChoose mode:\n1. Download only\n2. Training only\n3. Both\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print(f"\n{'='*60}")
        print("PHASE 1: DATASET PREPARATION (requires internet)")
        print(f"{'='*60}")
        
        for dataset in datasets_to_test:
            run_download_mode(dataset)
        
        print(f"\n{'='*60}")
        print("DOWNLOAD PHASE COMPLETED")
        print(f"{'='*60}")
        print("Next steps for HPC usage:")
        print("1. Copy the 'src/data/' folder to your HPC workspace")
        print("2. Set 'only_download_dataset: false' in cfg.yml")
        print("3. Run training on HPC")
    
    if choice in ['2', '3']:
        # Check if datasets exist
        missing_datasets = []
        for dataset in datasets_to_test:
            data_dir = Path("src/data")
            train_file = data_dir / f"{dataset}_train.parquet"
            test_file = data_dir / f"{dataset}_test.parquet"
            
            if not (train_file.exists() and test_file.exists()):
                missing_datasets.append(dataset)
        
        if missing_datasets and choice == '2':
            print(f"\n❌ Missing dataset files for: {missing_datasets}")
            print("Please run download mode first (choice 1 or 3)")
            return
        
        print(f"\n{'='*60}")
        print("PHASE 2: TRAINING (HPC-compatible, no internet needed)")
        print(f"{'='*60}")
        
        for dataset in datasets_to_test:
            if dataset not in missing_datasets:
                run_training_mode(dataset, epochs=2)  # Short demo
        
        print(f"\n{'='*60}")
        print("TRAINING PHASE COMPLETED")
        print(f"{'='*60}")
        print("The system is working correctly!")
        print("For full training, increase epochs in cfg.yml")

if __name__ == "__main__":
    main()
