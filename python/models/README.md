# 网络模型定义 (Models)

- `model_rafnet.py`：定义了 RAFnet（Range-aware Autofocus Network）。网络包含一个共享特征提取骨干，以及三个渐进式聚焦单元（分别估计 2/3阶、4/5阶、6/7阶多项式相位误差）。输入为 2 通道复数张量 `(B, 2, H, W)`，输出为 6 维相位系数 `(B, 6)`。
