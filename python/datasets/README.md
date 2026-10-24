# 数据集接口 (Datasets)

- `sar_autofocus_dataset.py`：标准的 PyTorch `Dataset` 实现。
  - **接口说明**：读取 `manifest.csv`，将复数图像拆分为实部（Channel 0）和虚部（Channel 1），返回形状为 `(2, H, W)` 的 Tensor。
  - **防泄漏机制**：支持传入 `split_json`，严格按照场景（Scene）级别进行过滤，确保训练集与验证集完全隔离。
