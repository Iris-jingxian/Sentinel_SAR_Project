import torch
import torch.nn as nn

class RAFnet(nn.Module):
    def __init__(self, in_channels=2, out_dim=6, dropout=0.5, az_bins=32):
        super().__init__()
        # 方位向大核（输入 H x W，方位向是 W 轴 = tensor 最后一维，kernel size 为 (1, K)）。
        # 不能用 InstanceNorm2d：它按 (H, W) 对每个样本/通道零均值化，紧接着 GAP 再平均，
        # 结果等价于只剩 bias（常数），输入被抹掉、模型“躺平”。这里用 BatchNorm2d 保留样本间差异。
        #
        # fix3 中等改动：
        #   1) 卷积 5 层 → 3 层（感受野 115 → 51），减少对单场景纹理的记忆（防过拟合）；
        #   2) LeakyReLU(0.01) → LeakyReLU(0.2)，保留负半轴 20%（符号/幅值不再被压到 1e-12）。
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 31), padding=(0, 15)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 头：只沿 range(H) 池化、保留方位向(W) 空间结构。二次相位 φ=α₂·(2πu)² 的符号决定
        # 散焦方向，编码在方位向左右不对称里。fix2 已确认：删掉 head 里的 BN + 最后 Linear 前
        # 用 Tanh，让符号可学（sign_hit 55% → 85%+）。fix3 保持该成功结构，仅把 LeakyReLU(0.01)
        # 提为 LeakyReLU(0.2)。
        self.head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, az_bins)),
            nn.Flatten(),
            nn.Linear(32 * az_bins, 128),
            nn.Tanh(),                      # 保护符号
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        feat = self.features(x)
        # 沿 range 方向(H, dim=2) 平均，保留方位向(W, dim=3)
        feat = feat.mean(dim=2, keepdim=True)
        out = self.head(feat)
        return out
