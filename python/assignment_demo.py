#!/usr/bin/env python3
"""
assignment_demo.py — SAR 自聚焦作业一次性验收脚本

按顺序跑通 6 项任务，每项打印清晰的数值结论：
  1. 搭建所选网络（RAFnet）
  2. 前向传播
  3. 反向传播
  4. 权重保存 / 加载
  5. 小样本过拟合
  6. 梯度检查（float64 中心差分）

用法（必须在项目根目录运行）：
  py assignment_demo.py              # 快速模式（默认）：小分辨率，约 2 分钟
  py assignment_demo.py --full       # 全分辨率 512x512（过拟合 400 epoch，约 40 分钟）
  py assignment_demo.py --epochs 120 --samples 16   # 自定义过拟合规模

环境注意：必须用 `py`（= Python 3.13.7 + torch 2.14.1+cpu）。
裸 `python` 是 Anaconda 3.11，没装 torch，会报 ModuleNotFoundError。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent / "python"))
from model_rafnet import RAFnet
from datasets.sar_autofocus_dataset import SARAutofocusDataset


RESULTS = []  # (序号, 名称, 通过/失败, 说明)


def record(step: int, name: str, ok: bool, note: str):
    mark = "✅ PASS" if ok else "❌ FAIL"
    RESULTS.append((step, name, ok, note))
    print(f"[{step}/6] {name:24s} {mark}  {note}")


# ---------------------------------------------------------------------------
# 1. 搭建网络
# ---------------------------------------------------------------------------
def build_model(device):
    model = RAFnet(out_dim=6, dropout=0.0).to(device)
    total = sum(p.numel() for p in model.parameters())
    feat = sum(p.numel() for p in model.features.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    record(1, "搭建所选网络 RAFnet", total > 0,
           f"参数量 {total:,} (features {feat:,} + head {head:,})")
    return model


# ---------------------------------------------------------------------------
# 2. 前向传播
# ---------------------------------------------------------------------------
def forward_pass(model, device):
    x = torch.randn(2, 2, 64, 64, device=device)      # (B=2, 2 通道实/虚部, H, W)
    y = model(x)
    ok = tuple(y.shape) == (2, 6)
    record(2, "前向传播", ok, f"输入 (2,2,64,64) -> 输出 {tuple(y.shape)}（6 阶相位系数）")
    return x, y


# ---------------------------------------------------------------------------
# 3. 反向传播
# ---------------------------------------------------------------------------
def backward_pass(model, x, y):
    model.zero_grad()
    loss = (y ** 2).mean()                              # 用一个简单可导损失验证反向
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    n_nonzero = sum(1 for g in grads if g is not None and g.abs().sum() > 0)
    ok = n_nonzero == len(grads) and len(grads) > 0
    record(3, "反向传播", ok,
           f"loss={loss.item():.4f}，{n_nonzero}/{len(grads)} 个参数张量有非零梯度")


# ---------------------------------------------------------------------------
# 4. 权重保存 / 加载
# ---------------------------------------------------------------------------
def weight_save_load(model, x, y):
    path = Path("assignment_demo_weights.pth")
    torch.save(model.state_dict(), path)
    m2 = RAFnet(out_dim=6, dropout=0.0)
    m2.load_state_dict(torch.load(path, weights_only=True))
    ok = torch.allclose(y, m2(x), atol=1e-6)
    record(4, "权重保存 / 加载", ok,
           f"保存到 {path}，重载后前向输出一致 = {ok}")


# ---------------------------------------------------------------------------
# 5. 小样本过拟合
# ---------------------------------------------------------------------------
def overfit_small(device, epochs, n_samples, range_stride):
    manifest = Path("data/sim/defocus/unified/manifest.csv")
    split_json = Path("data/splits/autofocus_split_v1.json")
    ds = SARAutofocusDataset(manifest, split="train", split_json=split_json,
                             range_stride=range_stride)
    subset = Subset(ds, list(range(min(n_samples, len(ds)))))
    loader = DataLoader(subset, batch_size=4, shuffle=True)

    # 逐阶标准化（6 个阶量级差 5 个数量级，必须标准化才能一起学）
    with torch.no_grad():
        alpha_all = torch.stack([subset[i][2] for i in range(len(subset))])
        alpha_mean = alpha_all.mean(0)
        alpha_std = alpha_all.std(0) + 1e-8

    model = RAFnet(out_dim=6, dropout=0.0).to(device)   # dropout=0：过拟合要关闭
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    first_loss = None
    final_loss = None
    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, _, a in loader:
            x = x.to(device)
            a_scaled = ((a - alpha_mean) / alpha_std).to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, a_scaled)
            loss.backward()
            optimizer.step()
            tot += loss.item() * x.size(0)
        tot /= len(subset)
        final_loss = tot
        if first_loss is None:
            first_loss = tot
        if epoch == 1 or epoch % max(1, epochs // 4) == 0 or epoch == epochs:
            print(f"      epoch {epoch:4d}/{epochs}  scaled MSE = {tot:.3e}")

    torch.save(model.state_dict(), "assignment_demo_overfit.pth")
    ok = final_loss < 0.2 and final_loss < 0.5 * first_loss
    record(5, "小样本过拟合", ok,
           f"{len(subset)} 样本 x {epochs} epoch，scaled MSE {first_loss:.3f} -> {final_loss:.3e}"
           f"（已存 assignment_demo_overfit.pth）")


# ---------------------------------------------------------------------------
# 6. 梯度检查（float64 中心差分）
# ---------------------------------------------------------------------------
def gradient_check(device):
    model = RAFnet(out_dim=6, dropout=0.0).to(device)
    model.eval().double()                               # float64 + eval 冻结 BN 统计量
    x = torch.randn(2, 2, 64, 64, device=device, dtype=torch.float64)
    target = torch.randn(2, 6, device=device, dtype=torch.float64)
    criterion = nn.MSELoss()

    model.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()

    params = [p for p in model.parameters() if p.requires_grad]
    analytic = torch.cat([p.grad.flatten() for p in params])

    eps = 1e-6
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(analytic), size=5, replace=False)

    max_rel_err = 0.0
    for i in idxs:
        off = 0
        for p in params:
            n = p.numel()
            if off <= i < off + n:
                li = i - off
                flat = p.data.flatten()
                v0 = flat[li].item()
                flat[li] = v0 + eps
                lp = criterion(model(x), target).item()
                flat[li] = v0 - eps
                lm = criterion(model(x), target).item()
                flat[li] = v0
                num = (lp - lm) / (2 * eps)
                ana = analytic[i].item()
                rel = abs(num - ana) / (max(abs(num), abs(ana)) + 1e-12)
                max_rel_err = max(max_rel_err, rel)
                break
            off += n

    ok = max_rel_err < 1e-2
    record(6, "梯度检查", ok,
           f"float64 中心差分，5 个随机参数，最大相对误差 {max_rel_err:.2e}（阈值 1e-2）")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="全分辨率 512x512（过拟合 400 epoch）")
    ap.add_argument("--epochs", type=int, default=None, help="过拟合 epoch 数")
    ap.add_argument("--samples", type=int, default=8, help="过拟合样本数")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    full = args.full
    epochs = args.epochs if args.epochs else (400 if full else 80)
    range_stride = 1 if full else 4          # 快速模式沿 range 下采样 4x，约省 4 倍时间

    print("=" * 70)
    print("SAR 自聚焦作业验收  |  device:", device,
          "|  mode:", "FULL 512x512" if full else "QUICK (range_stride=4)")
    print("=" * 70)

    model = build_model(device)
    x, y = forward_pass(model, device)
    backward_pass(model, x, y)
    weight_save_load(model, x, y)
    overfit_small(device, epochs, args.samples, range_stride)
    gradient_check(device)

    print("\n" + "=" * 70)
    print("验收汇总")
    print("=" * 70)
    all_ok = True
    for step, name, ok, note in RESULTS:
        print(f"  [{step}/6] {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print("=" * 70)
    print("结论:", "✅ 6 项全部通过" if all_ok else "❌ 存在未通过项，见上方 FAIL")
    print("=" * 70)


if __name__ == "__main__":
    main()
