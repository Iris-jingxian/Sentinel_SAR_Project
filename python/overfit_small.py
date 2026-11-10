import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).parent / "python"))
from datasets.sar_autofocus_dataset import SARAutofocusDataset
from model_rafnet import RAFnet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    manifest = Path("data/sim/defocus/unified/manifest.csv")
    split_json = Path("data/splits/autofocus_split_v1.json")
    ds = SARAutofocusDataset(manifest, split="train", split_json=split_json)

    # 取出 8 个样本
    n_samples = 8
    subset = Subset(ds, list(range(n_samples)))
    loader = DataLoader(subset, batch_size=4, shuffle=True)

    # 标签标准化：逐阶计算 mean / std，让 6 个量纲差异巨大的阶在 Loss 中权重一致。
    # 用完整 subset 确定性计算，避免随 shuffle 波动。
    with torch.no_grad():
        alpha_all = torch.stack([subset[i][2] for i in range(len(subset))])
        alpha_mean = alpha_all.mean(dim=0)
        alpha_std = alpha_all.std(dim=0) + 1e-8  # 防止除 0
        print(f"各阶 alpha 的 mean: {alpha_mean.numpy()}")
        print(f"各阶 alpha 的 std : {alpha_std.numpy()}")

    # dropout=0：过拟合测试要关闭 Dropout，否则模型难以在小样本上记忆。
    model = RAFnet(out_dim=6, dropout=0.0).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n开始在小样本 (N={n_samples}) 上过拟合...")
    for epoch in range(1, 401):
        model.train()
        total_loss = 0.0
        for x, _, alpha in loader:
            x = x.to(device)
            # 标签标准化（零均值、单位方差）
            alpha_scaled = ((alpha - alpha_mean) / alpha_std).to(device)

            optimizer.zero_grad()
            pred_scaled = model(x)
            loss = criterion(pred_scaled, alpha_scaled)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        total_loss /= len(subset)
        if epoch % 40 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/400]  Scaled MSE Loss: {total_loss:.3e}")

    # ====== 保存过拟合后的权重 ======
    torch.save(model.state_dict(), "overfit_model.pth")
    print("已保存过拟合权重 -> overfit_model.pth")

    # ====== 验证预测结果 ======
    print("\n训练完成！进行最终对比验证：")
    model.eval()
    with torch.no_grad():
        full_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        x, _, alpha = next(iter(full_loader))
        x = x.to(device)
        mean_dev = alpha_mean.to(device)
        std_dev = alpha_std.to(device)

        pred_scaled = model(x)
        # 反标准化：还原回真实量级
        pred = (pred_scaled * std_dev + mean_dev).cpu().numpy()
        true = alpha.numpy()

        # 原始量纲下的拟合误差（各阶 MSE / 平均相对误差）
        mse_per_order = np.mean((pred - true) ** 2, axis=0)
        rel_per_order = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-8), axis=0)
        print(f"\n原始量纲 MSE(各阶):        {np.array2string(mse_per_order, precision=6, suppress_small=True)}")
        print(f"原始量纲平均相对误差(各阶): {np.array2string(rel_per_order, precision=4, suppress_small=True)}")

        # 相对误差中位数（对少量离群点更稳健）
        rel_med = np.median(np.abs(pred - true) / (np.abs(true) + 1e-8), axis=0)
        print(f"原始量纲相对误差中位数(各阶): {np.array2string(rel_med, precision=4, suppress_small=True)}")

        for i in range(min(2, len(subset))):
            print(f"\n--- 样本 {i+1} ---")
            print(f"真实 Alpha: {np.array2string(true[i], precision=5, suppress_small=True)}")
            print(f"预测 Alpha: {np.array2string(pred[i], precision=5, suppress_small=True)}")


if __name__ == "__main__":
    main()
