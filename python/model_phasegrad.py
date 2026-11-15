"""model_phasegrad.py — 相位梯度网络：在场景不变「斜率」特征上回归相位误差系数。

输入 (B, 1)：相位梯度斜率 c1（见 features.phase_gradient_feature），场景不变。
输出 (B, 6)：α₂..α₇；只用第 0 个（α₂）做二次项补偿（n_orders=1，与经典法/RAFnet 同口径）。

关键（2026-11-14 实测）：
  斜率 → α₂ 是线性映射 α₂ = c1·n_az/(8π²)，斜率 w≈6.48 场景不变；
  但截距 b 是「场景相关」的（train≈0.43 / val≈1.2 / test≈0），是场景 clean 相位
  沿 range 平均未完全抵消而泄漏的常量，跨场景不迁移、会把 test 从 0.86 拖到 0.60。
  物理上相位误差模型 φ(u)=Σαₖ(2πu)ᵏ 无常数项/线性项 ⇒ dφ/du 过原点 ⇒ b≡0。
  因此去掉 bias（bias=False），只学场景不变的斜率。救援核心是「特征+物理先验」，非网络深度。
"""
import torch
from torch import nn


class PhaseGradNet(nn.Module):
    def __init__(self, n_in: int = 1, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, out_dim, bias=False),  # b≡0：相位误差模型无常数项，截距是场景噪声
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
