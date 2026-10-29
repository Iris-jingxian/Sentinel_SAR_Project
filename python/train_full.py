import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent / "python"))
from datasets.sar_autofocus_dataset import SARAutofocusDataset
from model_rafnet import RAFnet
from phase_error import phase_vector_torch

def apply_phase_compensation(x_defocus, alpha):
    """物理补偿 (可微)：I_clean = IFFT_az( FFT_az(I_defocus) · exp(-jφ) )。

    φ 模型统一来自 python/phase_error.py 的 phase_vector_torch，保证与数据生成一致。
    """
    x_c = torch.complex(x_defocus[:, 0], x_defocus[:, 1])
    n_az = x_c.shape[2]
    phi = phase_vector_torch(n_az, alpha)          # (B, n_az)
    X = torch.fft.fft(x_c, dim=2)
    X_comp = X * torch.exp(-1j * phi.unsqueeze(1))
    y_c = torch.fft.ifft(X_comp, dim=2)
    return torch.stack([y_c.real, y_c.imag], dim=1)

def entropy_loss(x_img):
    """计算可微的能量熵 (式 2-3)"""
    x_c = torch.complex(x_img[:, 0], x_img[:, 1])
    power = torch.abs(x_c) ** 2                    # 用功率 |g|^2
    # 展平
    power_flat = power.view(power.size(0), -1)
    Z = torch.sum(power_flat, dim=1, keepdim=True) + 1e-12
    prob = power_flat / Z
    entropy = -torch.sum(prob * torch.log(prob + 1e-12), dim=1)
    return torch.mean(entropy)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 统一数据集，按场景划分 train/val/test（split 列已写入 manifest）
    manifest = Path("data/sim/defocus/unified/manifest.csv")
    split_json = Path("data/splits/autofocus_split_v1.json")
    ds_train = SARAutofocusDataset(manifest, split="train", split_json=split_json)
    
    train_loader = DataLoader(ds_train, batch_size=8, shuffle=True)

    model = RAFnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60
    best_loss = float('inf')

    print("开始纯无监督训练 (熵损失)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_def, x_clean, alpha in train_loader: 
            # 注意：此训练中完全不使用 alpha 和 x_clean，只用 x_def 做输入
            x_def = x_def.to(device)
            
            optimizer.zero_grad()
            pred_alpha = model(x_def)
            pred_clean = apply_phase_compensation(x_def, pred_alpha)
            
            # 纯粹的熵损失
            loss = entropy_loss(pred_clean)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_def.size(0)
            
        train_loss /= len(ds_train)
        print(f"Epoch [{epoch+1:02d}/{epochs}] Entropy Loss: {train_loss:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "model_best.pth")
            print("  -> 模型已保存")

    print("训练全部完成！")

if __name__ == "__main__":
    main()