"""
Training workflow functions.
"""

import torch
from dataset.utils import create_data_loaders, get_dataset_statistics
from dataset.dataset_factory import get_dataset_info
from models import create_model
from optimizers import create_optimizer
from utils.loss import get_loss_function
from utils.train import complete_training_run


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
    loss_config = config.get('loss', {})
    loss_function = get_loss_function(
        loss_config.get('name', 'crossentropy'),
        **loss_config.get('params', {})
    )
    
    # Create scheduler if specified
    scheduler = None
    if 'scheduler' in config and config['scheduler']['enabled']:
        scheduler_type = config['scheduler']['type']
        scheduler_params = config['scheduler']['params']
        
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_type == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    
    # Training
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_metrics, zip_file_path = complete_training_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=loss_function,
        device=device,
        epochs=config['epochs'],
        optimizer_name=config['optimizer'],
        dataset_name=dataset_name,
        scheduler=scheduler,
        enable_progress_bar=True,
        save_curves=True,
        output_dir="results"
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Best Test Accuracy: {best_metrics.get('best_test_acc_overall', 0):.2f}%")
    if zip_file_path:
        print(f"Results saved to: {zip_file_path}")
    print("="*50)
    
    return best_metrics, zip_file_path
