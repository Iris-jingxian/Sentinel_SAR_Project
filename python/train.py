#!/usr/bin/env python3
"""
train.py — 监督训练（回归相位误差系数）

数据: data/sim/defocus/unified/manifest.csv（场景级 train/val/test 已写入 split 列）
输入: 散焦复图像 I_defocus 的 [实部, 虚部]，shape (2, H, W)
标签: 相位误差系数 α（6 维，对应 2..7 阶）

损失: 逐阶标准化的 MSE，且**只监督二次项（k=2，默认 n_orders=1）**。

为什么这么设计（实测结论）：
  1. 各阶 α 量级差 5 个数量级，必须逐阶标准化，否则高阶项梯度 ~1e-5、几乎无梯度；
     反过来直接用原始 MSE 会发散（实测 val_corr 0.39 < 基线 0.62）。
  2. 二次项是散焦的唯一主导来源：只用二次项补偿，重聚焦后与干净图的相关已达
     0.981（全 6 阶才是 1.000）。k=3..7 合计只贡献 ~2% 的差距，且带内相位 RMS
     依次 ~0.35 / 0.16 / 0.07 / 0.03 / 0.01 rad，几乎不可从图像反演。
  3. 所以只监督二次项（可学且重要），其余阶让模型预测 0（真值本就接近 0），
     避免不可学的高阶项稀释损失 / 与二次项争抢共享特征。

评估指标: 用「重聚焦后与干净图的相关 val_corr」而不是 alpha loss——这才是
下游 evaluate_model / compare_baselines 真正关心的（熵↓/对比度↑/与干净图相关↑）。
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 用中文字体渲染图表标题（Windows 自带微软雅黑/黑体；DejaVu Sans 无 CJK 字形会出豆腐块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).parent / "python"))
from datasets.sar_autofocus_dataset import SARAutofocusDataset, _load_alpha
from model_rafnet import RAFnet
from phase_error import phase_vector_torch


def apply_phase_compensation(x_defocus, alpha, n_orders=None):
    """可微补偿：I_clean = IFFT_az( FFT_az(I_defocus) · exp(-jφ) )，用于验证指标。

    n_orders: 只使用前 n_orders 个系数（其余置 0）。训练只监督前 n_orders 阶，
    未监督的高阶预测停留在随机初始化、量级远大于真值（实测 k=4..7 分别达真值的
    ~20/45/400/∞ 倍），会把补偿彻底毁掉，所以补偿时必须把未监督的高阶置 0。
    """
    if n_orders is not None:
        alpha = torch.cat([alpha[:, :n_orders],
                           torch.zeros_like(alpha[:, n_orders:])], dim=1)
    x_c = torch.complex(x_defocus[:, 0], x_defocus[:, 1])
    n_az = x_c.shape[2]
    phi = phase_vector_torch(n_az, alpha)
    X = torch.fft.fft(x_c, dim=2)
    X_comp = X * torch.exp(-1j * phi.unsqueeze(1))
    y_c = torch.fft.ifft(X_comp, dim=2)
    return torch.stack([y_c.real, y_c.imag], dim=1)


def apply_phase_compensation_spectral(x_spectral, alpha, n_orders=None):
    """补偿（输入已是方位频谱 [real(FFT), imag(FFT)]）：乘 exp(-jφ) 再 IFFT 回空间域。

    与 apply_phase_compensation 对应，但跳过 FFT（输入已是频域）。用于 --spectral 训练时的
    val 指标——模型吃频谱输入，但「重聚焦后与干净图相关」必须在空间域比，故这里 IFFT 回空间。
    """
    if n_orders is not None:
        alpha = torch.cat([alpha[:, :n_orders],
                           torch.zeros_like(alpha[:, n_orders:])], dim=1)
    X = torch.complex(x_spectral[:, 0], x_spectral[:, 1])
    n_az = X.shape[2]
    phi = phase_vector_torch(n_az, alpha)
    X_comp = X * torch.exp(-1j * phi.unsqueeze(1))
    y_c = torch.fft.ifft(X_comp, dim=2)
    return torch.stack([y_c.real, y_c.imag], dim=1)


def image_corr(img_a, img_b):
    """图像幅值相关系数（对齐 evaluate_model 的 corr_with_clean 口径）。"""
    a = torch.abs(torch.complex(img_a[:, 0], img_a[:, 1])).reshape(img_a.shape[0], -1)
    b = torch.abs(torch.complex(img_b[:, 0], img_b[:, 1])).reshape(img_b.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    return (num / den).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/sim/defocus/unified/manifest.csv"))
    p.add_argument("--split-json", type=Path, default=Path("data/splits/autofocus_split_v1.json"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="学习率（默认 3e-4；续训时显式传入会覆盖 checkpoint 里保存的旧 lr）")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--range-stride", type=int, default=1,
                   help="range 向下采样倍数（1=不降；4 降显存 4x，可开大 batch）") # <--- 新增这行
    p.add_argument("--n-orders", type=int, default=1,
                   help="只监督前 N 个阶（默认 1 = 二次项 k=2），其余阶不参与损失")
    p.add_argument("--aux-alpha-weight", type=float, default=0.0,
                   help="辅助 alpha 监督权重（原始量级 MSE，仅前 --aux-orders 阶；默认 0=关闭，"
                        ">0 时把预测幅值拉回真值量级、纠正 a2std 塌缩）")
    p.add_argument("--aux-orders", type=int, default=1,
                   help="辅助 alpha 监督覆盖的阶数（默认 1 = 仅二次项 k=2）")
    p.add_argument("--limit", type=int, default=0,
                   help="仅用前 N 个训练/验证样本（0=全部），便于快速验证")
    p.add_argument("--step-size", type=int, default=30, help="学习率每 N 轮 ×0.5")
    p.add_argument("--out", type=Path, default=Path("model_best.pth"))
    p.add_argument("--spectral", action="store_true",
                   help="模型输入改为方位频域（forward 内先 FFT 再提取特征），暴露二次相位曲率、让符号可学")
    p.add_argument("--resume", type=Path, default=None,
                   help="从已有权重/checkpoint 继续训练（state_dict 或含 optimizer/epoch 的完整 checkpoint）")
    return p.parse_args()

def _corr_1d(a, b) -> float:
    """一维 Pearson 相关；std≈0（塌缩）时返回 NaN（本身就是诊断信号）。"""
    a = a.astype(float)
    b = b.astype(float)
    a = a - a.mean()
    b = b - b.mean()
    den = a.std() * b.std()
    return float((a * b).mean() / den) if den > 1e-12 else float("nan")


def evaluate_val(model, val_loader, device, n_orders=None, spectral=False) -> dict:
    """验证：重聚焦相关（主指标）+ α₂ 诊断（预测 std / 符号命中 / signed·|α| 相关）。

    α₂ 诊断比 val_corr 更早暴露失败：spatial 输入下模型会塌缩到「预测常数」——std→0、
    sign_hit≈50%（随机）、signed_corr≈0。塌缩时 val_corr 可能被选择偏差抬成假高（实测 0.73），
    但 std/sign 一眼现形，省掉全量白跑。
    """
    model.eval()
    total_corr = 0.0
    pred_a2, true_a2 = [], []
    with torch.no_grad():
        for x_def, x_clean, alpha in val_loader:
            x_def, x_clean = x_def.to(device), x_clean.to(device)
            pred = model(x_def)
            pred_a2.append(pred[:, 0].detach().cpu())   # α₂ = 第 0 个系数（k=2）
            true_a2.append(alpha[:, 0].cpu())
            if spectral:
                refocused = apply_phase_compensation_spectral(x_def, pred, n_orders=n_orders)
            else:
                refocused = apply_phase_compensation(x_def, pred, n_orders=n_orders)
            total_corr += image_corr(refocused, x_clean).item() * x_def.size(0)
    p = torch.cat(pred_a2).float().numpy()
    t = torch.cat(true_a2).float().numpy()
    return {
        "corr": total_corr / len(val_loader.dataset),
        "alpha2_std": float(p.std()),
        "sign_hit": float((np.sign(p) == np.sign(t)).mean()),
        "signed_corr": _corr_1d(p, t),
        "abs_corr": _corr_1d(np.abs(p), np.abs(t)),
    }


def plot_curves(rows: list, out_path: Path) -> None:
    """保存训练/验证曲线：train_loss 与 val_corr 各一张子图（单一序列，无需图例）。"""
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_corr = [r["val_corr"] for r in rows]

    ink, muted, grid_c, base_c = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.plot(epochs, train_loss, color="#2a78d6", linewidth=2, marker="o", markersize=4)
    ax.set_title("训练损失 (scaled MSE)", color=ink)
    ax.set_xlabel("epoch", color=muted)
    ax.set_ylabel("loss", color=muted)
    ax.grid(True, color=grid_c, linewidth=0.8)
    ax.tick_params(colors=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(base_c)
    ax.spines["bottom"].set_color(base_c)

    ax = axes[1]
    ax.plot(epochs, val_corr, color="#eb6834", linewidth=2, marker="o", markersize=4)
    ax.axhline(0.62, color=base_c, linestyle="--", linewidth=1)
    ax.text(epochs[0], 0.625, "基线 ~0.62", color=muted, fontsize=9)
    ax.set_title("验证重聚焦相关 val_corr", color=ink)
    ax.set_xlabel("epoch", color=muted)
    ax.set_ylabel("corr", color=muted)
    ax.grid(True, color=grid_c, linewidth=0.8)
    ax.tick_params(colors=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(base_c)
    ax.spines["bottom"].set_color(base_c)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"训练/验证曲线已保存到 {out_path}")


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)  # 自动创建 --out 父目录（如 runs/sup_spectral_fix3/）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    input_domain = "spectral" if args.spectral else "spatial"
    ds_train = SARAutofocusDataset(args.manifest, split="train", split_json=args.split_json,
                                   input_domain=input_domain, range_stride=args.range_stride)
    ds_val = SARAutofocusDataset(args.manifest, split="val", split_json=args.split_json,
                                 input_domain=input_domain, range_stride=args.range_stride)
    if args.limit > 0:
        ds_train = Subset(ds_train, list(range(min(args.limit, len(ds_train)))))
        ds_val = Subset(ds_val, list(range(min(args.limit, len(ds_val)))))
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"train={len(ds_train)}  val={len(ds_val)}")

    # 逐阶 mean/std（基于训练集），用于标准化损失
    root = ds_train.dataset.root if isinstance(ds_train, Subset) else ds_train.root
    rows = ds_train.dataset.rows if isinstance(ds_train, Subset) else ds_train.rows
    alphas = []
    for r in rows:
        lp = Path(r["label_path"])
        if not lp.is_absolute():
            lp = root / lp
        alphas.append(torch.from_numpy(_load_alpha(lp)))
    alpha_mean = torch.stack(alphas).float().mean(0).to(device)
    alpha_std = (torch.stack(alphas).float().std(0) + 1e-8).to(device)
    print(f"标签各阶 std: {alpha_std.cpu().numpy()}")
    print(f"只监督前 {args.n_orders} 个阶（k=2..{args.n_orders + 1}）")
    if args.aux_alpha_weight > 0:
        print(f"辅助 alpha 监督：weight={args.aux_alpha_weight}，前 {args.aux_orders} 阶（原始量级 MSE）")

    lr = args.lr if args.lr is not None else 3e-4
    model = RAFnet(out_dim=6, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    # 恢复已有训练（继续训练）
    start_epoch = 0
    best_val_corr = -float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if args.lr is not None:
                    for g in optimizer.param_groups:
                        g["lr"] = args.lr
                    print(f"已用 --lr={args.lr} 覆盖续训学习率（丢弃 checkpoint 里的旧 lr）")
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_corr = float(ckpt.get("best_val_corr", -float("inf")))
            print(f"恢复完整 checkpoint {args.resume}: epoch={start_epoch}, best_val_corr={best_val_corr:.4f}")
        else:
            model.load_state_dict(ckpt)
            print(f"加载权重 {args.resume}（无 optimizer/epoch，优化器与 LR 从默认开始）")
        m0 = evaluate_val(model, val_loader, device, n_orders=args.n_orders, spectral=args.spectral)
        base_corr = m0["corr"]
        print(f"恢复后 val_corr 基线 = {base_corr:.4f}  a2std={m0['alpha2_std']:.3f}  sign={m0['sign_hit']*100:.0f}%")
        best_val_corr = max(best_val_corr, base_corr)

    # 训练/验证曲线与首轮权重保存
    fieldnames = ["epoch", "lr", "train_loss", "val_corr", "alpha2_std", "sign_hit", "signed_corr", "abs_corr"]
    metrics_rows = []
    metrics_path = args.out.with_name(args.out.stem + "_metrics.csv")
    curve_path = args.out.with_name(args.out.stem + "_curve.png")
    epoch1_path = args.out.with_name(args.out.stem + "_epoch1.pth")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x_def, _x_clean, alpha in train_loader:
            x_def, alpha = x_def.to(device), alpha.to(device)
            optimizer.zero_grad()
            pred = model(x_def)
            # 只监督前 n_orders 阶，逐阶标准化
            n = args.n_orders
            loss = criterion((pred[:, :n] - alpha_mean[:n]) / alpha_std[:n],
                             (alpha[:, :n] - alpha_mean[:n]) / alpha_std[:n])
            # 辅助 alpha 监督：原始量级 MSE，把预测幅值拉到真值量级（纠正 a2std 塌缩）
            if args.aux_alpha_weight > 0:
                n_aux = min(args.aux_orders, alpha.shape[1])
                aux = criterion(pred[:, :n_aux], alpha[:, :n_aux])
                loss = loss + args.aux_alpha_weight * aux
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_def.size(0)
        train_loss /= len(ds_train)
        scheduler.step()

        # 验证：重聚焦相关（主指标）+ α₂ 诊断（std/sign/signed·|α| 相关）
        m = evaluate_val(model, val_loader, device, n_orders=args.n_orders, spectral=args.spectral)
        val_corr = m["corr"]
        warn = ("  WARN_COLLAPSE" if m["alpha2_std"] < 0.5 * alpha_std[0].item() else "")

        metrics_rows.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "val_corr": val_corr,
            "alpha2_std": m["alpha2_std"], "sign_hit": m["sign_hit"],
            "signed_corr": m["signed_corr"], "abs_corr": m["abs_corr"],
        })
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(metrics_rows[-1])
        if epoch == start_epoch + 1:
            torch.save(model.state_dict(), epoch1_path)
            print(f"  -> 保存首轮权重到 {epoch1_path}")

        print(f"Epoch [{epoch:02d}/{start_epoch + args.epochs}] lr={optimizer.param_groups[0]['lr']:.1e} "
              f"train_loss={train_loss:.4f}  val_corr={val_corr:.4f}  "
              f"a2std={m['alpha2_std']:.3f} sign={m['sign_hit']*100:.0f}% "
              f"sr={m['signed_corr']:+.2f} |a|r={m['abs_corr']:+.2f}{warn}")
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), args.out)
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_corr": best_val_corr,
            }, args.out.with_suffix(".ckpt.pth"))
            print(f"  -> 保存最佳权重到 {args.out} (val_corr={val_corr:.4f})")

        # 每轮保存最新 checkpoint（含 optimizer/scheduler），即使本轮非最优也能精确续训，
        # 避免进程被系统内存保护 kill 时丢失多轮进度（--resume 可从此处精确恢复）。
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_corr": best_val_corr,
        }, args.out.with_suffix(".latest.ckpt.pth"))

    plot_curves(metrics_rows, curve_path)
    print(f"训练完成，最佳 val_corr={best_val_corr:.4f}  (基线 ~0.62 / 理想 1.0)")


if __name__ == "__main__":
    main()
