import torch
import torch.nn as nn

class RAFnet(nn.Module):
    def __init__(self, in_channels=2, out_dim_per_stage=2):
        super().__init__()
        # 特征提取主干 (共享)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 256x256
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 128x128
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 64x64
        )
        
        # 三级渐进式聚焦单元
        # 第一级：估计 2阶、3阶 误差
        self.stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim_per_stage)
        )
        
        # 第二级：结合特征，估计 4阶、5阶 误差
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim_per_stage)
        )
        
        # 第三级：结合特征，估计 6阶、7阶 误差
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim_per_stage)
        )

    def forward(self, x):
        # 提取特征
        feat = self.backbone(x)
        
        # 逐级估计
        alpha1 = self.stage1(feat) # (B, 2) -> 对应 alpha2, alpha3
        alpha2 = self.stage2(feat) # (B, 2) -> 对应 alpha4, alpha5
        alpha3 = self.stage3(feat) # (B, 2) -> 对应 alpha6, alpha7
        
        # 拼接到一起，变成 (B, 6)，这样能兼容你们之前的 Dataset 和 Loss
        final_alpha = torch.cat([alpha1, alpha2, alpha3], dim=1)
        
        return final_alpha
