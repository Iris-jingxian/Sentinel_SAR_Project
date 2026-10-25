# B组运行环境配置与算力自查指南

为了保证后续 RD 成像、自聚焦和旁瓣抑制网络的顺利运行，我们组必须统一环境标准。本项目最核心的依赖是**MATLAB 与 Python 的联动**。

请各位组员务必按照以下步骤完成部署：

## 一、 基础软件与版本匹配 (极其重要)

由于 MATLAB 对 Python 版本有严格的兼容性要求，请根据你电脑上的 MATLAB 版本，选择对应的 Python 版本进行安装：

- **MATLAB 版本要求**：R2024a 或更高版本。
  - 必备工具箱：**Signal Processing Toolbox** (可在 MATLAB 命令行输入 `ver signal` 验证是否安装)。
- **Python 版本对应**：
  - 如果你的 MATLAB 是 **R2024a** 👉 **必须安装 Python 3.11**（最高仅支持到 3.11，切勿安装 3.12）。
  - 如果你的 MATLAB 是 **R2024b 或更新** 👉 可以安装 Python 3.11 或 3.12。

## 二、 安装 Python 项目依赖包

确认 Python 安装完毕后，打开终端（CMD 或 PowerShell），进入项目文件夹，运行以下命令安装解码依赖库：

```bash
pip install -r requirements.txt
