# Hadam Optimizer

A research project implementing the Hadam optimizer with support for multiple datasets optimized for High-Performance Computing (HPC) environments.

## Key Features

- **HPC-Optimized**: Datasets converted to Parquet format to reduce file count
- **Two-Phase Workflow**: Separate download and training phases for internet-restricted HPC
- **Multiple Datasets**: CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST
- **Configurable**: YAML-based configuration system
- **Flexible Training**: Support for multiple optimizers and training modes

## Quick Start

### For Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run demonstration**:
```bash
python demo.py
```

### For HPC Usage

See [HPC_USAGE_GUIDE.md](HPC_USAGE_GUIDE.md) for detailed instructions.

**Quick HPC workflow:**

1. **On machine with internet** (download datasets):
```yaml
# Edit src/cfg.yml
only_download_dataset: true
dataset: "cifar10"
```
```bash
cd src && python main.py
```

2. **Transfer to HPC**:
```bash
scp -r data/ username@hpc:/path/to/workspace/
```

3. **On HPC** (training):
```yaml
# Edit cfg.yml
only_download_dataset: false
train: true
eval: true
```
```bash
cd src && python main.py
```

## Configuration

### Basic Configuration (`src/cfg.yml`)

```yaml
# Dataset and model
dataset: "cifar10"  # cifar10, cifar100, svhn, fashion_mnist
cnn_model: "resnet18"

# Training parameters
batch_size: 128
epochs: 100
lr: 0.1
optimizer: "hadam"  # hadam, sgd, adam

# Mode controls (NEW)
only_download_dataset: false  # true = download only, false = training
train: true                   # Enable training
eval: true                    # Enable evaluation

# Optimizer-specific settings
hadam:
  order: 2
sgd:
  momentum: 0.9
  weight_decay: 5e-4
adam:
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0
```

## Dataset System

### Supported Datasets

| Dataset | Classes | Resolution | Format | HPC Files |
|---------|---------|------------|--------|-----------|
| CIFAR-10 | 10 | 32×32 RGB | 2 Parquet files | ✅ |
| CIFAR-100 | 100 | 32×32 RGB | 2 Parquet files | ✅ |
| SVHN | 10 | 32×32 RGB | 2 Parquet files | ✅ |
| Fashion-MNIST | 10 | 28×28 Grayscale | 2 Parquet files | ✅ |

### Benefits of Parquet Format

- **Reduced File Count**: 2 files per dataset vs. thousands of individual images
- **Faster I/O**: Columnar format with compression
- **HPC-Friendly**: Fewer inodes, better filesystem performance
- **Memory Efficient**: Lazy loading and optimized data types

## Project Structure

```
Hadam/
├── src/
│   ├── cfg.yml              # Configuration file
│   ├── main.py              # Main training script
│   ├── dataset/             # Dataset system
│   │   ├── __init__.py
│   │   ├── base_dataset.py     # Base dataset class
│   │   ├── cifar_dataset.py    # CIFAR-10/100
│   │   ├── svhn_dataset.py     # SVHN
│   │   ├── fashion_mnist_dataset.py  # Fashion-MNIST
│   │   ├── dataset_factory.py  # Dataset factory
│   │   └── utils.py            # Data loading utilities
│   ├── opt/
│   │   └── hadam.py            # Hadam optimizer (implement)
│   └── utils/
│       ├── train.py            # Training utilities
│       ├── eval.py             # Evaluation utilities
│       └── loss.py             # Loss functions
├── data/                    # Generated dataset files
├── requirements.txt         # Dependencies
├── demo.py                  # Demonstration script
├── HPC_USAGE_GUIDE.md      # Detailed HPC guide
└── README.md               # This file
```

## Usage Examples

### Download Datasets for HPC

```python
from dataset.utils import convert_dataset_to_parquet

# Convert all datasets
datasets = ['cifar10', 'cifar100', 'svhn', 'fashion_mnist']
for dataset_name in datasets:
    convert_dataset_to_parquet(dataset_name)
```

### Training with Custom Configuration

```python
# Edit cfg.yml, then:
from main import main
main()
```

### Advanced Data Loading

```python
from dataset.utils import create_data_loaders
import torchvision.transforms as transforms

# Custom transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create data loaders
train_loader, test_loader = create_data_loaders(
    dataset_name='cifar10',
    batch_size=128,
    custom_transform=transform
)
```

## Research Applications

This system is designed for:

- **Optimizer Comparison**: Easy switching between SGD, Adam, and Hadam
- **Dataset Benchmarking**: Consistent evaluation across multiple datasets
- **HPC Experiments**: Large-scale experiments on computing clusters
- **Ablation Studies**: Systematic parameter exploration

## Performance Notes

### Local Development
- First run downloads and converts datasets (one-time cost)
- Subsequent runs use cached Parquet files
- Memory usage optimized for laptops/workstations

### HPC Environment
- No internet required after dataset preparation
- Optimized for distributed filesystems
- Reduced filesystem pressure with fewer files
- Faster loading times with compressed format

## Extending the System

### Adding New Datasets

1. Create dataset class in `src/dataset/`:
```python
class NewDataset(BaseDataset):
    @property
    def dataset_name(self) -> str:
        return "new_dataset"
    
    def _download_original(self):
        # Implement download logic
        pass
```

2. Register in `dataset_factory.py`
3. Add to configuration options

### Custom Optimizers

Implement in `src/opt/` and add to `create_optimizer()` in `main.py`.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the project structure
4. Test with both local and HPC workflows
5. Submit a pull request

## Citation

If you use this dataset system or the Hadam optimizer in your research, please cite:

```bibtex
@misc{hadam2024,
  title={Hadam Optimizer with HPC-Optimized Dataset System},
  author={[Your Name]},
  year={2024},
  url={https://github.com/hibana2077/Hadam}
}
```