# Curve Tracking and Output System

這個系統提供了完整的訓練曲線追蹤和輸出功能，能夠收集 loss curve 和 accuracy curve，並自動輸出成 zip 和個別 txt 檔案。

## 新增功能

### 1. CurveTracker (utils/eval.py)
- 追蹤訓練和驗證的 loss 和 accuracy 曲線
- 支援學習率曲線追蹤
- 自動生成檔案名格式：`{optimizer}-{dataset}-{date}`

### 2. LossTracker (utils/loss.py)
- 追蹤詳細的 batch-level loss 資訊
- 提供更細緻的 loss 分析

### 3. Enhanced Training Functions (utils/train.py)
- `complete_training_run()`: 完整的訓練流程，包含自動曲線追蹤
- 自動保存曲線到 txt 和 zip 檔案

## 檔案輸出格式

### 個別 txt 檔案
- `{opt}-{dataset}-{date}-train_loss.txt`: 訓練 loss 曲線
- `{opt}-{dataset}-{date}-train_acc.txt`: 訓練 accuracy 曲線
- `{opt}-{dataset}-{date}-eval_loss.txt`: 驗證 loss 曲線
- `{opt}-{dataset}-{date}-eval_acc.txt`: 驗證 accuracy 曲線
- `{opt}-{dataset}-{date}-learning_rate.txt`: 學習率曲線
- `{opt}-{dataset}-{date}-summary.txt`: 實驗摘要
- `{opt}-{dataset}-{date}-detailed_loss.txt`: 詳細 batch loss

### Zip 檔案
- `{opt}-{dataset}-{date}.zip`: 包含所有上述 txt 檔案

### 檔案格式範例
```
# Train Loss Curve
# Epoch	Loss
1	2.301234
2	1.543210
3	1.123456
...
```

## 使用方法

### 方法 1: 使用 complete_training_run (推薦)

```python
from utils.train import complete_training_run

best_metrics, zip_file_path = complete_training_run(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=100,
    optimizer_name="hadam",  # 或 "sgd", "adam"
    dataset_name="cifar10",  # 或 "cifar100", "svhn", "fashion_mnist"
    scheduler=None,
    enable_progress_bar=True,
    save_curves=True,
    output_dir="results"
)
```

### 方法 2: 手動追蹤

```python
from utils.eval import CurveTracker
from utils.loss import LossTracker

# 初始化追蹤器
curve_tracker = CurveTracker("hadam", "cifar10")
loss_tracker = LossTracker("hadam", "cifar10")

# 在訓練迴圈中
for epoch in range(epochs):
    # ... 訓練代碼 ...
    
    # 添加訓練指標
    curve_tracker.add_train_metrics(epoch, train_loss, train_acc, lr)
    
    # 添加驗證指標
    curve_tracker.add_eval_metrics(eval_loss, eval_acc)
    
    # 記錄 batch loss (在訓練迴圈內)
    for batch_idx, (data, target) in enumerate(train_loader):
        # ... 前向傳播和反向傳播 ...
        loss_tracker.log_batch_loss(epoch, batch_idx, loss.item())

# 保存曲線
zip_file = curve_tracker.save_curves_to_zip("results")
loss_tracker.save_detailed_loss_log("results")
```

## 主程式使用

修改 `cfg.yml` 設定：

```yaml
dataset: "cifar10"  # cifar10, cifar100, svhn, fashion_mnist
optimizer: "hadam"  # hadam, sgd, adam
batch_size: 128
epochs: 100
lr: 0.1
train: true
eval: true
only_download_dataset: false  # 設為 false 開始訓練
```

執行訓練：

```bash
cd src
python main.py
```

## 輸出結果

執行完成後，會在 `results/` 目錄下生成：

1. **Zip 檔案**: `hadam-cifar10-20240716_143022.zip`
   - 包含所有曲線檔案

2. **個別 txt 檔案**:
   - `hadam-cifar10-20240716_143022-train_loss.txt`
   - `hadam-cifar10-20240716_143022-train_acc.txt`
   - `hadam-cifar10-20240716_143022-eval_loss.txt`
   - `hadam-cifar10-20240716_143022-eval_acc.txt`
   - `hadam-cifar10-20240716_143022-learning_rate.txt`
   - `hadam-cifar10-20240716_143022-summary.txt`
   - `hadam-cifar10-20240716_143022-detailed_loss.txt`

## 範例腳本

執行範例腳本來了解功能：

```bash
python example_curve_tracking.py
```

這會生成示例曲線檔案，展示系統的功能。

## 最佳指標

系統會自動計算並報告：
- 最佳訓練 loss 和 accuracy
- 最佳驗證 loss 和 accuracy
- 對應的 epoch 數
- 最終結果

## 注意事項

1. 檔案名包含時間戳，避免覆蓋
2. 所有曲線資料都會保存為易於分析的文字格式
3. Zip 檔案方便傳輸和備份
4. 系統相容不同的優化器和資料集
5. 支援自定義輸出目錄
