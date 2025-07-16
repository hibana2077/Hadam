# Hugging Face Parquet Datasets Integration

本項目已經整合了來自 Hugging Face 的預處理 Parquet 格式數據集，提供更快的加載速度和更好的性能。

## 支援的數據集

- **CIFAR-10**: 10類自然圖像分類
- **CIFAR-100**: 100類自然圖像分類  
- **Fashion-MNIST**: 10類時裝圖像分類
- **SVHN**: 街景房屋號碼識別

## 數據集來源

所有數據集來自 Hugging Face Hub：

- [hibana2077/CV-dataset-all-in-parquet](https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet)

## 使用方法

### 基本使用

```python
from src.dataset import get_dataset

# 載入 CIFAR-10 訓練集
train_dataset = get_dataset(
    name='cifar10',
    root='./data',
    train=True,
    download=True
)

# 載入測試集
test_dataset = get_dataset(
    name='cifar10',
    root='./data',
    train=False,
    download=True
)

print(f"訓練樣本數量: {len(train_dataset)}")
print(f"測試樣本數量: {len(test_dataset)}")
```

### 使用 DataLoader

```python
from torch.utils.data import DataLoader

# 創建數據載入器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# 獲取一個批次
images, labels = next(iter(train_loader))
print(f"圖像形狀: {images.shape}")
print(f"標籤形狀: {labels.shape}")
```

### 自定義變換

```python
import torchvision.transforms as transforms

# 定義自定義變換
custom_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用自定義變換
dataset = get_dataset(
    name='cifar10',
    root='./data',
    train=True,
    transform=custom_transform,
    download=True
)
```

## 數據集資訊

### CIFAR-10

- **樣本數量**: 60,000 (50,000 訓練 + 10,000 測試)
- **圖像大小**: 32×32 彩色
- **類別數量**: 10
- **類別**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### CIFAR-100

- **樣本數量**: 60,000 (50,000 訓練 + 10,000 測試)
- **圖像大小**: 32×32 彩色
- **類別數量**: 100

### Fashion-MNIST

- **樣本數量**: 70,000 (60,000 訓練 + 10,000 測試)
- **圖像大小**: 28×28 灰階
- **類別數量**: 10
- **類別**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

### SVHN

- **樣本數量**: 99,289 (73,257 訓練 + 26,032 測試)
- **圖像大小**: 32×32 彩色
- **類別數量**: 10 (數字 0-9)

## 示範腳本

運行示範腳本來測試數據集：

```bash
# 運行基本示範
python demo_datasets.py

# 運行完整測試（包含視覺化）
python test_datasets.py
```

## 優勢

1. **快速加載**: Parquet 格式提供比原始格式更快的 I/O 性能
2. **自動下載**: 首次使用時自動從 Hugging Face 下載
3. **統一介面**: 所有數據集使用相同的 API
4. **記憶體效率**: 優化的數據結構減少記憶體使用
5. **兼容性**: 保持與 PyTorch DataLoader 的完全兼容

## 注意事項

- 首次下載時需要網路連接
- 數據集會被快取在指定的 `root` 目錄中
- 支援所有標準的 torchvision transforms

## 故障排除

### 下載失敗

如果下載失敗，請檢查網路連接並重試。數據集會自動重新下載。

### 記憶體不足

對於大數據集，建議使用較小的 batch_size 或增加系統記憶體。

### 自定義數據集

如需添加新的數據集，請參考 `hf_parquet_dataset.py` 中的 `HFParquetDataset` 基類。
