import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent / "python"))
from model_rafnet import RAFnet
from datasets.sar_autofocus_dataset import SARAutofocusDataset

def grad_check():
    device = torch.device("cpu") 
    model = RAFnet(out_dim=6).to(device)
    model.eval()
    
    # 【改动 1】：把所有参数和模型整体转换为 float64 双精度
    model = model.double() 

    manifest = Path("data/sim/defocus/unified/manifest.csv")
    split_json = Path("data/splits/autofocus_split_v1.json")
    ds = SARAutofocusDataset(manifest, split="train", split_json=split_json)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    x, _, target_alpha = next(iter(loader))
    
    # 【改动 2】：输入数据也必须是 float64，标签也要转双精度
    x = x.to(device).double() 
    target_alpha = target_alpha.to(device).double()
    
    print(f"输入数据 shape: {x.shape}, 标签 alpha shape: {target_alpha.shape}")

    criterion = nn.MSELoss()

    model.zero_grad()
    pred = model(x)
    loss = criterion(pred, target_alpha)
    loss.backward()

    params = [p for p in model.parameters() if p.requires_grad]
    analytic_grads = torch.cat([p.grad.flatten() for p in params]).clone()

    # float64 下用中心差分，eps=1e-6 已足够（截断误差 O(eps^2) 可忽略）
    eps = 1e-6
    
    n_checks = 5 
    np.random.seed(42)
    idxs = np.random.choice(len(analytic_grads), size=n_checks, replace=False)

    print(f"\n开始检查 {n_checks} 个随机参数的梯度...")
    print(f"{'参数索引':>8} | {'数值梯度(参考)':>14} | {'解析梯度(反向传播)':>16} | {'相对误差':>10}")
    print("-" * 60)

    max_rel_err = 0.0
    for i in idxs:
        offset = 0
        for p in params:
            n = p.numel()
            if offset <= i < offset + n:
                local_idx = i - offset
                orig_val = p.data.flatten()[local_idx].item()

                # 计算 loss(w + eps)
                p.data.flatten()[local_idx] = orig_val + eps
                with torch.no_grad():
                    loss_plus = criterion(model(x), target_alpha).item()

                # 计算 loss(w - eps)
                p.data.flatten()[local_idx] = orig_val - eps
                with torch.no_grad():
                    loss_minus = criterion(model(x), target_alpha).item()

                p.data.flatten()[local_idx] = orig_val

                num_grad = (loss_plus - loss_minus) / (2 * eps)
                ana_grad = analytic_grads[i].item()

                rel_err = abs(num_grad - ana_grad) / (max(abs(num_grad), abs(ana_grad)) + 1e-12)
                max_rel_err = max(max_rel_err, rel_err)

                print(f"{i:>8} | {num_grad:>14.6e} | {ana_grad:>16.6e} | {rel_err:>10.2e}")
                break
            offset += n

    print("-" * 60)
    thresh = 1e-2  # 数值梯度与解析梯度的相对误差阈值
    if max_rel_err < thresh:
        print(f"✅ PASS! 最大相对误差: {max_rel_err:.2e} (小于 {thresh:.0e})")
        print("证明 PyTorch 的反向传播（backward）计算完全正确！")
    else:
        print(f"❌ FAIL! 最大相对误差: {max_rel_err:.2e} (大于 {thresh:.0e})")
        print("请检查网络中是否包含不可导操作（如 round, argmax 等）。")

if __name__ == "__main__":
    grad_check()
