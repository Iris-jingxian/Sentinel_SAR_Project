# 🛰️ Sentinel_SAR_Project

> 哨兵SAR图像处理项目 —— 人工智能课程本科生大作业

---

## 👥 团队成员
- 朱静娴
- 廖智慧
- 林嘉欣
- 郭涵卓

## 📂 项目文档导航
- 👉 [B组项目分工表](docs/01_team/team_roles.md)
- 📋 [哨兵SAR图像处理软件需求清单（V0.1）](docs/02_requirements/requirements.md)
- *(后续文献笔记、周报等文档将陆续在此补充)*

---

### 🎯 周二：成像与数据

L0 解码与 RD 真实场景成像

- **底层解码与成像代码**：
  - `python/decode_s1_raw.py`、`python/decode_s1_navigation.py`
  - `matlab/Read_Data_RD.m`、`matlab/read_s1_raw.m`
- 🔗 [真实场景 RD 成像结果 (`RD_SM.jpg`)](results/RD_SM.jpg)
  - *说明：成功解码 Sentinel-1 L0 RAW 原始回波数据，并跑通距离多普勒（RD）算法生成的聚焦图像。*
- 🔗 [点目标仿真结果 (`sim_point_RD.jpg`)](results/sim_point_RD.jpg)
  - *说明：理想单点目标经过 RD 算法聚焦后的二维冲激响应图（十字星），验证了算法在距离向与方位向的聚焦正确性。*

官方 SLC 数据读取与参数提取

- **SLC 读取脚本**：`matlab/read_slc.m`
- **参数表记录**：`data/slc_params.mat`、`data/设置SAR系统参数和成像参数.txt`
- 🔗 [SLC 原始幅度图 (`SLC_amplitude.jpg`)](results/SLC_amplitude.jpg)
  - *说明：通过底层读取 Tiff 偏移量解析官方 SLC 数据，1/4 抽样后展示的全幅度图。清晰呈现了雷达相干斑噪声特征与地表强散射点，验证了复数读取的准确性。*

真实数据测试集 (ROI) 切片与交付

- **切片数据文件**：`data/ROIs/ROI_1_complex.mat`、`ROI_2_complex.mat`、`ROI_3_complex.mat`
- 🔗 [ROI 1: 强目标区 (`ROI_1_preview.png`)](results/ROI_1_preview.png)
  - *说明：全分辨率 512x512 复数切片，包含密集亮斑，适合用于测试旁瓣抑制效果及主瓣能量保持。*
- 🔗 [ROI 2: 均匀弱区 (`ROI_2_preview.png`)](results/ROI_2_preview.png)
  - *说明：均匀斑点背景区，适合用于测试网络是否会将弱信号当作噪声抹除，检查虚假峰。*
- 🔗 [ROI 3: 复杂边缘区 (`ROI_3_preview.png`)](results/ROI_3_preview.png)
  - *说明：包含明显纹理与海陆边缘交界，适合用于测试自聚焦网络对边缘锐度的恢复能力。*

---

### 🎯 周三：成像与数据验证

*   🛰️ [真实 SLC 原始回波幅度图 (`SLC_amplitude.jpg`)](results/SLC_amplitude.jpg)
    *   *说明：基于底层二进制读取解析真实 Sentinel-1 SLC 数据，1/4 抽样后展示的全幅幅度图。清晰呈现了雷达相干斑噪声特征与地表强散射点，验证了数据读取与解析的正确性。*
*   🖥️ [RD 成像与定量分析脚本 (`MultiTarget_Simulation_and_Profile.m`)](matlab/MultiTarget_Simulation_and_Profile.m)
    *   *说明：包含多点目标回波生成、SRC与RCMC高精度RD成像、亚像素插值的主瓣宽度定量计算。*
*   🖼️ [多点目标仿真基线图 (`sim_point_multi.jpg`)](results/sim_point_multi.jpg)
    *   *说明：5个分布在不同距离/方位位置的点目标，经RD算法处理后完美聚焦为清晰的十字星，无相互干扰。*
*   📈 [-3dB 主瓣宽度剖面图 (`profile_3dB.jpg`)](results/profile_3dB.jpg)
    *   *说明：沿中心目标切取的距离向与方位向剖面图，精准标定 -3dB 阈值线与实测边界。*
### 🎯 周三：基线生成与最小熵自聚焦

* 🖼️ [相位误差注入生成散焦图 (`defocus_sample.jpg`)](results/defocus_sample.jpg)
  * *说明：基于 `matlab_generate_defocus.m`，对恢复的真实 SAR 干净 ROI 沿方位向频域注入二次与三次相位误差（Q=10 rad），生成已知真值的散焦图像，用于测试自聚焦算法下限。*
* 🖼️ [最小熵自聚焦恢复结果对比图 (`matlab_autofocus_compare.jpg`)](results/matlab_autofocus_compare.jpg)
  * *说明：基于 `matlab_min_entropy_autofocus.m`，以“逐距离门能量加权熵”为目标函数，采用“一维粗网格搜索 + 二维 Nelder-Mead 精修”策略估计相位系数，完成频域相位补偿。结果显示散焦星点重新聚焦，且估计的 \(\alpha_2, \alpha_3\) 与注入真值高度吻合。*
