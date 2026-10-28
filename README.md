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

### 🎯 周三：成像与数据验证

*   🛰️ [真实 SLC 原始回波幅度图 (`SLC_amplitude.jpg`)](results/SLC_amplitude.jpg)
    *   *说明：基于底层二进制读取解析真实 Sentinel-1 SLC 数据，1/4 抽样后展示的全幅幅度图。清晰呈现了雷达相干斑噪声特征与地表强散射点，验证了数据读取与解析的正确性。*
*   🖥️ [RD 成像与定量分析脚本 (`MultiTarget_Simulation_and_Profile.m`)](matlab/MultiTarget_Simulation_and_Profile.m)
    *   *说明：包含多点目标回波生成、SRC与RCMC高精度RD成像、亚像素插值的主瓣宽度定量计算。*
*   🖼️ [多点目标仿真基线图 (`sim_point_multi.jpg`)](results/sim_point_multi.jpg)
    *   *说明：5个分布在不同距离/方位位置的点目标，经RD算法处理后完美聚焦为清晰的十字星，无相互干扰。*
*   📈 [-3dB 主瓣宽度剖面图 (`profile_3dB.jpg`)](results/profile_3dB.jpg)
    *   *说明：沿中心目标切取的距离向与方位向剖面图，精准标定 -3dB 阈值线与实测边界。*
