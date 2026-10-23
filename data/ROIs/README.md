# 真实 SLC 数据 ROI 测试集

本文件夹包含了从 Sentinel-1 官方 SLC 数据中裁剪出的 3 个感兴趣区域（ROI），专供自聚焦与旁瓣抑制网络进行测试与验证。

## 📦 数据文件清单

所有数据均保存为 `.mat` 格式，内部变量名为 `roi_complex`，数据类型为 **complex single（复数单精度）**，**保留了完整的幅度与相位信息**。

*   **`ROI_1_complex.mat`**：强反射目标区（包含密集亮斑，适合用于测试旁瓣抑制效果及主瓣能量保持）。
*   **`ROI_2_complex.mat`**：均匀弱目标区（如海面，适合用于测试算法是否会将弱信号当作噪声抹除，或检查虚假峰）。
*   **`ROI_3_complex.mat`**：复杂边缘交界区（包含明显的纹理与边缘，适合用于测试自聚焦网络对边缘锐度的恢复能力）。

## 📐 数据规格
- **尺寸**：全分辨率 \(512 \times 512\)（**未进行降采样**，每个像素点都是原始采样点）。
- **保存方式**：直接保存了复数矩阵，未做 `abs()` 运算。

## 💻 如何使用

在 MATLAB 或 Python 中直接加载即可：

**MATLAB 读取示例：**
```matlab
% 读取强目标区数据
load('ROI_1_complex.mat'); 

% 查看幅度图 (dB)
figure; 
imagesc(20*log10(abs(roi_complex))); 
colormap gray; axis xy; colorbar;
title('ROI 1 幅度图');
