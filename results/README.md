# 项目可视化结果展示

本文件夹用于存放各阶段代码跑出的真实结果图（仅存放预览图，不存放原始数据）。

## 📊 图片清单

*   **`RD_SM.jpg`**：基于真实 Sentinel-1 L0 RAW 数据跑通的 RD 成像结果图。
*   **`SLC_amplitude.jpg`**：直接读取官方 SLC 复数数据生成的幅度图（抽样率 1/4）。
*   **`sim_point_RD.jpg`**：RD 算法点目标仿真的二维冲激响应（十字星图），用于验证算法正确性。
*   **`ROI_1_preview.png`**：强反射区 ROI 切片预览图。
*   **`ROI_2_preview.png`**：均匀弱区 ROI 切片预览图。
*   **`ROI_3_preview.png`**：复杂边缘区 ROI 切片预览图。

> 💡 **注**：对应的原始复数 `.mat` 文件请前往 `data/ROIs/` 目录查看。
