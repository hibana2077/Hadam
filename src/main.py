"""
Main training script for Hadam optimizer experiments.
"""

import yaml
import torch
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dataset_downloader import download_datasets_only
from training_workflow import train_model


def load_config(config_path: str = "cfg.yml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    # Load configuration
    config = load_config()
    
    # Check mode
    mode = config.get('mode', 'train')
    
    if mode == 'download':
        # Only download datasets
        download_datasets_only(config)
        
    elif mode == 'train':
        # Training mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Train model
        train_model(config, device)

    elif mode == 'full':
        # Full run: download datasets + train model
        print("Full Run Mode: Download + Train")
        print("=" * 50)
        
        # Step 1: Download and convert datasets
        print("Step 1: Downloading and converting datasets...")
        download_datasets_only(config)
        
        print("\n" + "=" * 50)
        print("Step 2: Starting training...")
        
        # Step 2: Training mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Train model
        train_model(config, device)
        
        print("\n" + "=" * 50)
        print("FULL RUN COMPLETED!")
        print("✅ Dataset downloaded and model training finished.")
        print("=" * 50)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'download' or 'train'")


if __name__ == "__main__":
    main()