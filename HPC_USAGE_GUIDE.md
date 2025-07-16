# HPC Usage Guide for Hadam Dataset System

## Overview

This guide explains how to use the Hadam dataset system in HPC environments where internet access is restricted.

## Two-Phase Workflow

### Phase 1: Dataset Preparation (With Internet)

Before submitting jobs to HPC, prepare datasets on a machine with internet access.

1. **Configure for Download Mode**
   
   Edit `cfg.yml`:
   ```yaml
   dataset: "cifar10"  # or cifar100, svhn, fashion_mnist
   only_download_dataset: true
   train: false  # Optional, will be ignored in download mode
   eval: false   # Optional, will be ignored in download mode
   ```

2. **Run Download Script**
   ```bash
   cd src
   python main.py
   ```
   
   This will:
   - Download the original dataset
   - Convert it to Parquet format
   - Save files in `./data/` directory
   - Display confirmation message

3. **Verify Download**
   
   Check that these files exist:
   ```
   data/
   ├── cifar10_train.parquet
   └── cifar10_test.parquet
   ```

4. **Transfer to HPC**
   
   Copy the entire `data/` folder to your HPC workspace:
   ```bash
   scp -r ./data/ username@hpc-cluster:/path/to/your/workspace/
   ```

### Phase 2: Training on HPC (No Internet)

1. **Configure for Training Mode**
   
   Edit `cfg.yml`:
   ```yaml
   dataset: "cifar10"
   cnn_model: "resnet18"
   batch_size: 128
   epochs: 100
   lr: 0.1
   optimizer: "hadam"
   
   # Mode controls
   only_download_dataset: false
   train: true
   eval: true
   
   enable_progress_bar: false  # Recommended for HPC
   ```

2. **HPC Job Script Example**
   
   Create `job_hadam.sh`:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=hadam_cifar10
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=8
   #SBATCH --gres=gpu:1
   #SBATCH --time=04:00:00
   #SBATCH --partition=gpu
   
   # Load modules
   module load python/3.8
   module load cuda/11.0
   
   # Activate environment
   source ~/venv/bin/activate
   
   # Navigate to workspace
   cd $SLURM_SUBMIT_DIR/src
   
   # Run training
   python main.py
   ```

3. **Submit Job**
   ```bash
   sbatch job_hadam.sh
   ```

## Configuration Options

### Dataset Selection
```yaml
dataset: "cifar10"     # Options: cifar10, cifar100, svhn, fashion_mnist
```

### Mode Controls
```yaml
only_download_dataset: false  # true = download only, false = training mode
train: true                   # Enable/disable training
eval: true                    # Enable/disable evaluation during training
```

### Training Parameters
```yaml
cnn_model: "resnet18"        # Model architecture
batch_size: 128              # Batch size for training
epochs: 100                  # Number of training epochs
lr: 0.1                      # Learning rate
optimizer: "hadam"           # Optimizer: hadam, sgd, adam
enable_progress_bar: false   # Disable for HPC environments
```

### Optimizer-Specific Settings
```yaml
# Hadam optimizer
hadam:
  order: 2

# SGD optimizer  
sgd:
  momentum: 0.9
  weight_decay: 5e-4

# Adam optimizer
adam:
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0
```

## File Structure on HPC

Your HPC workspace should look like:
```
your_workspace/
├── src/
│   ├── cfg.yml
│   ├── main.py
│   ├── dataset/
│   ├── opt/
│   └── utils/
├── data/
│   ├── cifar10_train.parquet
│   ├── cifar10_test.parquet
│   └── ... (other datasets)
├── requirements.txt
└── job_hadam.sh
```

## Error Handling

### Common Issues

1. **Dataset files not found**
   ```
   Error: Dataset files not found!
   Expected files:
     ./data/cifar10_train.parquet
     ./data/cifar10_test.parquet
   ```
   
   **Solution**: Run Phase 1 (dataset preparation) first.

2. **Import errors**
   ```
   ModuleNotFoundError: No module named 'pandas'
   ```
   
   **Solution**: Install requirements on HPC:
   ```bash
   pip install -r requirements.txt
   ```

3. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   
   **Solution**: Reduce batch size in `cfg.yml`:
   ```yaml
   batch_size: 64  # or smaller
   ```

## Performance Tips

1. **Optimal Batch Size**: Start with 128, reduce if memory issues occur
2. **Progress Bar**: Disable for HPC (`enable_progress_bar: false`)
3. **Number of Workers**: Use 4-8 for data loading
4. **Mixed Precision**: Consider adding mixed precision training for larger models

## Multiple Datasets

To prepare multiple datasets:

1. **Download all datasets**:
   ```yaml
   # Run for each dataset
   dataset: "cifar10"
   only_download_dataset: true
   ```
   
   Then change to `cifar100`, `svhn`, `fashion_mnist` and repeat.

2. **Switch datasets for different experiments**:
   Just change the `dataset` field in `cfg.yml` and rerun.

## Monitoring

The system will output:
- Training progress per epoch
- Training and validation accuracy
- Best model checkpoints
- Final results summary

Example output:
```
Epoch 1: Train Loss: 2.1234, Train Acc: 23.45%
         Test Loss: 1.9876, Test Acc: 28.90%
         New best model saved! Accuracy: 28.90%
```

## Best Practices

1. **Always test locally first** with a few epochs
2. **Use version control** for your configuration files
3. **Save multiple checkpoints** for long training runs
4. **Monitor resource usage** to optimize job parameters
5. **Use array jobs** for hyperparameter sweeps
