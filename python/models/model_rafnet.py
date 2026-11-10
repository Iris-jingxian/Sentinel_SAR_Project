import torch
import torch.nn as nn

class RAFnet(nn.Module):
    def __init__(self, in_channels=2, out_dim=6, dropout=0.5, az_bins=32):
        super().__init__()
        # 方位向大核（输入 H x W，方位向是 W 轴 = tensor 最后一维，kernel size 为 (1, K)）
        # 注意：不能用 InstanceNorm2d——它会按 (H, W) 对每个样本/通道做零均值化，
        # 紧接着 GAP 再对 (H, W) 求平均，结果等价于只剩 InstanceNorm 的 bias（常数），
        # 输入信息被完全抹掉，模型对所有输入输出几乎相同（即“预测躺平/欠拟合”）。
        # 这里改用 BatchNorm2d（按 batch+spatial 归一化，保留样本间差异）。
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(1, 63), padding=(0, 31)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(16, 32, kernel_size=(1, 31), padding=(0, 15)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # 头：只沿 range(H) 池化、保留方位向(W) 的空间结构。
        # 二次相位 φ=α₂·(2πu)² 的符号决定散焦方向（向左糊 vs 向右糊），这个「方向」信息
        # 编码在方位向的左右不对称里。旧版 AdaptiveAvgPool2d((1,1)) 沿方位向整体平均，
        # 会把左右不对称抹掉，导致符号不可学（实测 medium 只学到 |α|、符号近随机）。
        # 这里改为：沿 range 平均（散焦沿方位向是全局的、与 range 无关，平均可降噪），
        # 沿方位向粗量化到 az_bins 格，再进 FC——head 能读到「左重 vs 右重」的不对称。
        self.head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),          # 降通道
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.AdaptiveAvgPool2d((1, az_bins)),          # 沿方位向粗量化，保留左右结构
            nn.Flatten(),
            nn.Linear(32 * az_bins, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        feat = self.features(x)
        # 沿 range 方向(H, dim=2) 平均，保留方位向(W, dim=3)
        feat = feat.mean(dim=2, keepdim=True)
        out = self.head(feat)
        return out
