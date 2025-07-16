"""
Main training script for Hadam optimizer experiments.
"""

import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dataset.utils import create_data_loaders, get_dataset_statistics
from dataset.dataset_factory import get_dataset_info


def load_config(config_path: str = "cfg.yml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_name: str, num_classes: int, input_shape: tuple) -> nn.Module:
    """Create a CNN model based on configuration."""
    if model_name == "resnet18":
        import torchvision.models as models
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        
        # Adjust first layer for different input shapes
        if input_shape[0] != 3:  # Not RGB
            model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Adjust for smaller images (like CIFAR, Fashion-MNIST)
        if input_shape[1] == 28 or input_shape[1] == 32:
            model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_optimizer(model: nn.Module, config: dict):
    """Create optimizer based on configuration."""
    opt_name = config['optimizer'].lower()
    lr = config['lr']
    
    if opt_name == 'hadam':
        try:
            from opt.hadam import HAdam
            optimizer = HAdam(
                model.parameters(),
                lr=lr,
                order=config['hadam']['order']
            )
        except ImportError:
            print("Warning: Hadam optimizer not found, using SGD instead")
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=config['sgd']['momentum'],
                weight_decay=config['sgd']['weight_decay']
            )
    
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['sgd']['momentum'],
            weight_decay=config['sgd']['weight_decay']
        )
    
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config['adam']['betas'],
            eps=config['adam']['eps'],
            weight_decay=config['adam']['weight_decay']
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer


def download_datasets_only(config: dict):
    """Download and convert datasets to Parquet format only."""
    print("Dataset Download and Conversion Mode")
    print("=" * 50)
    
    dataset_name = config['dataset']
    print(f"Downloading and converting dataset: {dataset_name}")
    
    # Import the conversion function
    from dataset.utils import convert_dataset_to_parquet
    
    # Convert dataset to Parquet format
    convert_dataset_to_parquet(
        dataset_name=dataset_name,
        data_root="./data",
        force_reconvert=False
    )
    
    print(f"\n✅ Dataset {dataset_name} successfully downloaded and converted to Parquet format!")
    print("Files are ready for HPC usage without internet connection.")
    print("You can now copy the ./data folder to your HPC environment.")


def train_model(config: dict, device: torch.device):
    """Training function."""
    print("Training Mode")
    print("=" * 20)
    
    # Get dataset information
    dataset_name = config['dataset']
    dataset_info = get_dataset_info(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Input shape: {dataset_info['input_shape']}")
    
    # Create data loaders (no download, use existing Parquet files)
    print("\nLoading data from Parquet files...")
    train_loader, test_loader = create_data_loaders(
        dataset_name=dataset_name,
        data_root="./data",
        batch_size=config['batch_size'],
        num_workers=4,
        use_parquet=True
    )
    
    # Print dataset statistics
    train_stats = get_dataset_statistics(train_loader.dataset)
    test_stats = get_dataset_statistics(test_loader.dataset)
    print(f"Training samples: {train_stats['num_samples']}")
    print(f"Test samples: {test_stats['num_samples']}")
    
    # Create model
    print(f"\nCreating model: {config['cnn_model']}")
    model = create_model(
        config['cnn_model'],
        dataset_info['num_classes'],
        dataset_info['input_shape']
    ).to(device)
    
    # Create optimizer
    print(f"Creating optimizer: {config['optimizer']}")
    optimizer = create_optimizer(model, config)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
      # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    
    # Import training utilities
    try:
        from utils.train import complete_training_run
        
        # Use the new complete training function with curve tracking
        best_metrics, zip_file_path = complete_training_run(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=config['epochs'],
            optimizer_name=config['optimizer'],
            dataset_name=dataset_name,
            scheduler=None,  # Add scheduler here if needed
            enable_progress_bar=config.get('enable_progress_bar', True),
            save_curves=True,
            output_dir="results"
        )
        
        print(f"\n=== Final Results ===")
        print(f"Best test accuracy: {best_metrics.get('best_eval_acc', 0):.2f}%")
        print(f"Final test accuracy: {best_metrics.get('final_test_acc', 0):.2f}%")
        if zip_file_path:
            print(f"Results saved to: {zip_file_path}")
            
    except ImportError:
        print("Warning: Advanced training utilities not found, using basic training loop")
        
        # Fallback to basic training loop with manual curve tracking
        from utils.eval import CurveTracker, evaluate_model
        from utils.loss import LossTracker
        from utils.train import train_epoch, get_learning_rate
        
        # Initialize trackers
        curve_tracker = CurveTracker(config['optimizer'], dataset_name)
        loss_tracker = LossTracker(config['optimizer'], dataset_name)
        
        best_accuracy = 0.0
        
        for epoch in range(1, config['epochs'] + 1):
            # Training
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, 
                enable_progress_bar=config.get('enable_progress_bar', True),
                curve_tracker=curve_tracker,
                loss_tracker=loss_tracker,
                epoch=epoch
            )
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%')
            
            # Evaluation
            if config.get('eval', True):
                test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
                curve_tracker.add_eval_metrics(test_loss, test_acc)
                
                print(f'         Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%')
                
                # Save best model
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    torch.save(model.state_dict(), f'best_model_{dataset_name}_{config["optimizer"]}.pth')
                    print(f'         New best model saved! Accuracy: {best_accuracy:.2f}%')
            
            print("-" * 60)
        
        # Save curves after training
        try:
            zip_file_path = curve_tracker.save_curves_to_zip("results")
            loss_tracker.save_detailed_loss_log("results")
            print(f"Training curves saved to: {zip_file_path}")
        except Exception as e:
            print(f"Warning: Could not save curves: {e}")
        
        print(f"\nTraining completed!")
        print(f"Best test accuracy: {best_accuracy:.2f}%")


def main():
    """Main function with mode control."""
    print("Hadam Optimizer Experiment System")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded:")
    for key, value in config.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Check mode
    only_download = config.get('only_download_dataset', False)
    train_mode = config.get('train', True)
    eval_mode = config.get('eval', True)
    
    print(f"\nMode Configuration:")
    print(f"  Only download dataset: {only_download}")
    print(f"  Training enabled: {train_mode}")
    print(f"  Evaluation enabled: {eval_mode}")
    
    if only_download:
        # Dataset download mode for pre-HPC setup
        download_datasets_only(config)
        return
    
    if not train_mode:
        print("\nTraining is disabled in configuration.")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Check if dataset files exist
    from pathlib import Path
    dataset_name = config['dataset']
    data_root = Path("./data")
    train_parquet = data_root / f"{dataset_name}_train.parquet"
    test_parquet = data_root / f"{dataset_name}_test.parquet"
    
    if not (train_parquet.exists() and test_parquet.exists()):
        print(f"\n❌ Error: Dataset files not found!")
        print(f"Expected files:")
        print(f"  {train_parquet}")
        print(f"  {test_parquet}")
        print(f"\nPlease run with 'only_download_dataset: true' first to download datasets.")
        return
    
    print(f"\n✅ Dataset files found, starting training...")
    
    # Start training
    train_model(config, device)


if __name__ == "__main__":
    main()