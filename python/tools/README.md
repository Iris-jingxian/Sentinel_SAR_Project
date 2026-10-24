# 数据处理工具 (Tools)

- `decode_s1_raw.py` / `decode_s1_navigation.py`：用于将 Sentinel-1 RAW 数据流式解码为 MAT 格式，并解析辅助轨道数据。
- `make_defocus_dataset.py`：**核心数据生成工具**。读取清晰的 SAR 图像，根据多项式模型注入频域相位误差，自动生成 `input_defocus.mat`（网络输入）和 `label_alpha.mat`（网络标签），并生成 `manifest.csv` 清单。
