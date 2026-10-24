# 提交数据清单

本文件夹包含了训练集（ROI_1）、验证集（ROI_2）和测试集（ROI_3）的完整样本清单。

每个 `manifest.csv` 记录了 100 个散焦样本的：
- `input_path`：散焦复数图路径
- `label_path`：真实多项式相位系数（\(\alpha\)）路径
- `defocus_strength`：注入的散焦强度（weak / medium / strong）
- `phase_rms_rad`：相位误差 RMS 值

*注意：表格中的 `input_path` 是相对路径，在实际加载时请确保本地已生成了对应的 `.mat` 文件。*
