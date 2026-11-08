"""SCNN 幅度中心值回归，网络尺寸依据 CN110333489A 正文。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SameConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        self.left = (kernel - 1) // 2
        self.right = kernel - 1 - self.left
        self.conv = nn.Conv1d(cin, cout, kernel)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, self.right)))


class SCNNB11(nn.Module):
    """输入 [B,1,50]，返回 [B,1]；外部数值单位沿用 amax=1000。"""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            SameConv1d(1, 10, 30), nn.ReLU(),
            SameConv1d(10, 10, 20), nn.ReLU(),
            SameConv1d(10, 10, 15), nn.ReLU(),
            SameConv1d(10, 3, 5), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(150, 100), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(100, 1))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (1, 50):
            raise ValueError("SCNN 输入须为 [B,1,50]")
        # 内部换算成单位尺度，保持数据接口的1000尺度，改善数值条件。
        # 线性输出允许监督标签超过输入峰值；推理显示时另行截去负幅度。
        return 1000.0 * self.head(self.features(x / 1000.0))


def load_scnn(path, device="cpu") -> SCNNB11:
    model = SCNNB11().to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()
