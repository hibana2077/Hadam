# Hadam Optimizer with Parquet Dataset Support

This project implements the Hadam optimizer with support for multiple datasets in Parquet format, optimized for HPC environments.

## Features

- **Multiple Datasets**: CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST
- **Parquet Format**: Reduced file count for HPC-friendly storage
- **Modular Design**: Easy to extend with new datasets
- **Configurable**: YAML-based configuration system

## Dataset System

### Supported Datasets

1. **CIFAR-10**: 10 classes, 32x32 RGB images
2. **CIFAR-100**: 100 classes, 32x32 RGB images  
3. **SVHN**: 10 digit classes, 32x32 RGB street view house numbers
4. **Fashion-MNIST**: 10 fashion item classes, 28x28 grayscale images

### Parquet Format Benefits

- **Reduced File Count**: Single file per train/test split instead of thousands
- **Faster Loading**: Optimized columnar format
- **Compression**: Built-in compression reduces storage requirements
- **HPC Friendly**: Fewer files = less filesystem stress

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test dataset functionality:
```bash
cd src
python test_datasets.py
```

3. Run main training:
```bash
cd src
python main.py
```

## Configuration

Edit `src/cfg.yml` to configure your experiment:

```yaml
dataset: "cifar10"  # cifar10, cifar100, svhn, fashion_mnist
cnn_model: "resnet18"
batch_size: 128
epochs: 100
lr: 0.1
optimizer: "hadam"  # hadam, sgd, adam
```

## Usage Examples

### Basic Usage

```python
from dataset import get_dataset, create_data_loaders

# Create data loaders
train_loader, test_loader = create_data_loaders(
    dataset_name='cifar10',
    batch_size=128,
    use_parquet=True
)

# Use in training loop
for data, target in train_loader:
    # Your training code here
    pass
```

### Custom Transforms

```python
import torchvision.transforms as transforms

custom_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader, test_loader = create_data_loaders(
    dataset_name='svhn',
    custom_transform=custom_transform
)
```

### Convert Dataset to Parquet

```python
from dataset.utils import convert_dataset_to_parquet

# Convert all datasets
datasets = ['cifar10', 'cifar100', 'svhn', 'fashion_mnist']
for dataset_name in datasets:
    convert_dataset_to_parquet(dataset_name, data_root='./data')
```

## File Structure

```
src/
├── cfg.yml                 # Configuration file
├── main.py                 # Main training script
├── test_datasets.py        # Dataset testing script
├── dataset/
│   ├── __init__.py
│   ├── base_dataset.py     # Base dataset class
│   ├── cifar_dataset.py    # CIFAR-10/100 implementations
│   ├── svhn_dataset.py     # SVHN implementation
│   ├── fashion_mnist_dataset.py  # Fashion-MNIST implementation
│   ├── dataset_factory.py  # Dataset factory and info
│   └── utils.py            # Utilities and data loaders
├── opt/
│   └── hadam.py           # Hadam optimizer (to be implemented)
└── utils/
    ├── train.py           # Training utilities
    ├── eval.py            # Evaluation utilities
    └── loss.py            # Loss functions
```

## HPC Usage

The Parquet format is particularly beneficial for HPC environments:

1. **Reduced Inode Usage**: Each dataset uses only 2 files (train/test) instead of thousands
2. **Faster I/O**: Columnar format with compression
3. **Memory Efficient**: Lazy loading and efficient data types
4. **Parallel Friendly**: Can be easily sharded for distributed training

### Sample HPC Job Script

```bash
#!/bin/bash
#SBATCH --job-name=hadam_experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load python/3.8
module load cuda/11.0

# Install dependencies
pip install -r requirements.txt

# Convert datasets (run once)
python test_datasets.py

# Run training
python main.py
```

## Extending the System

### Adding New Datasets

1. Create a new dataset class inheriting from `BaseDataset`
2. Implement the required methods:
   - `dataset_name` property
   - `_download_original()` method
   - `get_classes()` method

```python
class MyDataset(BaseDataset):
    @property
    def dataset_name(self) -> str:
        return "my_dataset"
    
    def _download_original(self) -> Tuple[np.ndarray, np.ndarray]:
        # Download and return data, targets
        pass
    
    def get_classes(self) -> list:
        return ["class1", "class2", ...]
```

3. Add to `dataset_factory.py`
4. Update configuration options

## Performance Notes

- First run will download and convert datasets to Parquet (one-time cost)
- Subsequent runs use cached Parquet files for fast loading
- Memory usage is optimized through lazy loading
- Compression reduces disk space requirements by 2-3x

## License

See LICENSE file for details.
