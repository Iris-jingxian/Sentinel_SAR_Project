import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(str(Path(__file__).parent / "python"))
from datasets.sar_autofocus_dataset import SARAutofocusDataset
from model_rafnet import RAFnet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 读取划分文件
    split_json = Path("data/splits/autofocus_split_v1.json")
    
    # 严格使用 split_json 过滤数据
    print("正在加载数据 (严格按 split.json 过滤)...")
    ds_train1 = SARAutofocusDataset(Path("data/sim/defocus/real_roi1/manifest.csv"), "train", split_json=split_json)
    #ds_train2 = SARAutofocusDataset(Path("data/sim/defocus/real_roi2/manifest.csv"), "train", split_json=split_json) # 此时 roi2 是 val，这里过滤后会为空
    ds_train = ConcatDataset([ds_train1]) # 实际上只用 roi1 训练
    
    ds_val = SARAutofocusDataset(Path("data/sim/defocus/real_roi2/manifest.csv"), "val", split_json=split_json)
    
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=16, shuffle=False)
    
    print(f"Train Size: {len(ds_train)} | Val Size: {len(ds_val)}")

    model = RAFnet().to(device)
    criterion = nn.MSELoss()
    # 降低学习率，防止塌缩
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 60  # 增加轮数
    best_val_loss = float('inf')

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, alpha in train_loader:
            x, alpha = x.to(device), alpha.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), alpha)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(ds_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, alpha in val_loader:
                x, alpha = x.to(device), alpha.to(device)
                loss = criterion(model(x), alpha)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(ds_val)

        print(f"Epoch [{epoch+1:02d}/{epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model_best.pth")
            print("  -> 模型已保存 (Best Val Loss)")

    print("训练全部完成！最佳验证集 Loss:", best_val_loss)

if __name__ == "__main__":
    main()
