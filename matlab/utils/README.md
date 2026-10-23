# MATLAB 辅助函数库 (Utils)

本文件夹包含了 Sentinel-1 L0 数据读取与 RD (距离多普勒) 成像算法运行所必需的底层支撑函数。为了方便主程序调用，统一归类于此。

## 📂 文件说明

*   **`doppler_geo_parameter.m`**：解析 XML 文件，计算多普勒中心频率、调频率、等效速度及几何参数，为后续成像提供基础参数。
*   **`IQ_correction.m`**：对原始接收到的有效样本进行 IQ 正交校正，消除直流偏置和幅相不平衡。
*   **`echo_SWST.m`**：接收窗口对齐，处理不同脉冲的采样窗口起始时间（SWST）差异。
*   **`Shift_Delay_correction.m`**：补偿系统时钟漂移与内部时延，并负责寻找并解析对应的 SLC annotation XML 文件。
*   **`interpolate_s1_orbit.m`**：利用三次 Hermite 插值算法，对卫星 GPS 轨道状态矢量（ECEF 坐标系）进行高精度插值。

## 💡 使用说明

在运行主目录下的 `Read_Data_RD.m` 或 `read_s1_raw.m` 之前，请确保在 MATLAB 中将此文件夹添加到搜索路径：

```matlab
addpath('utils');
