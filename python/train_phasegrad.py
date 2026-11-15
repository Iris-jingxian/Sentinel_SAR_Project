#!/usr/bin/env python3
"""train_phasegrad.py — 在场景不变「相位梯度」特征上训练 PhaseGradNet 救 DL 模型。

针对 RAFnet 塌缩的两层根因做修复：
  1. 输入换成谱相关特征 G = mean_r(X·conj(roll(X,1)))（抵消场景 clean 相位），非原始 [实,虚] 谱；
  2. Tanh（奇函数）保符号、无 GlobalAvgPool / 无 ReLU 截负。

其余口径与正式训练一致：
  - 场景级划分 train=ROI_2 / val=ROI_3 / test=ROI_1（无泄漏）；
  - 损失只监督 α₂（高阶系数幅值小、噪声大，全监督会发散）；
  - 保存首轮权重、训练/验证曲线 PNG、metrics CSV。

产出：
  model_phasegrad_best.pth / model_phasegrad_epoch1.pth
  model_phasegrad_curve.png
  model_phasegrad_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.io import loadmat

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "python"))
from features import phase_gradient_feature, phase_gradient_slope  # noqa: E402
from model_phasegrad import PhaseGradNet  # noqa: E402
from phase_error import apply_phase_compensation  # noqa: E402
from datasets.sar_autofocus_dataset import SARAutofocusDataset, _load_alpha  # noqa: E402
from train import plot_curves  # noqa: E402  复用同一套曲线样式


def load_clean(r: dict) -> np.ndarray:
    cm = loadmat(r["clean_path"])
    return cm["roi_complex"] if "roi_complex" in cm else next(
        v for k, v in cm.items() if not k.startswith("__") and isinstance(v, np.ndarray) and np.iscomplexobj(v))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.abs(a).ravel(), np.abs(b).ravel())[0, 1])


def load_split(manifest: Path, split_json: Path, split: str):
    ds = SARAutofocusDataset(manifest, split=split, split_json=split_json, normalize=False)
    feats, alpha, raw, clean = [], [], [], []
    n_reject = 0
    for r in ds.rows:
        img = loadmat(r["input_path"])["input_defocus"]
        # 数据清洗：|raw slope|>1 是相位解卷失败（dphi 绕了 2π 的伪斜率，实测 ±10；
        # 真值 α₂∈[-2.2,2.2] ⇒ |slope|≤0.34，物理上不可能 >1）。剔除，避免 MSE 被 (∓66)² 支配。
        if abs(phase_gradient_slope(img, axis=1)) > 1.0:
            n_reject += 1
            continue
        feats.append(phase_gradient_feature(img, axis=1))
        alpha.append(_load_alpha(Path(r["label_path"])))
        raw.append(img)
        clean.append(load_clean(r))
    print(f"  [{split}] 剔除解卷失败样本 {n_reject} 个")
    return (np.stack(feats).astype(np.float32),
            np.stack(alpha).astype(np.float32),
            np.stack(raw), np.stack(clean))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/sim/defocus/unified/manifest.csv"))
    p.add_argument("--split-json", type=Path, default=Path("data/splits/autofocus_split_v1.json"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val-corr-n", type=int, default=32, help="val_corr 只用前 N 个样本（省内存，sign_hit 仍用全量）")
    p.add_argument("--log-every", type=int, default=5, help="每多少个 epoch 打印一次")
    p.add_argument("--prefix", type=str, default="model_phasegrad")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print("加载 train/val（场景级划分）并预计算相位梯度特征 ...")
    train_feats, train_alpha, _, _ = load_split(args.manifest, args.split_json, "train")
    val_feats, val_alpha, val_raw, val_clean = load_split(args.manifest, args.split_json, "val")
    n_in = train_feats.shape[1]
    print(f"  train={train_feats.shape[0]}  val={val_feats.shape[0]}  特征维={n_in}  "
          f"α₂(std) train={train_alpha[:,0].std():.3f} val={val_alpha[:,0].std():.3f}")

    val_raw_sub = val_raw[: args.val_corr_n]
    val_clean_sub = val_clean[: args.val_corr_n]

    model = PhaseGradNet(n_in=n_in, out_dim=6).to(device)
    # 物理先验初始化：α₂ = slope·n_az/(8π²)。相位误差模型 φ(u)=Σαₖ(2πu)ᵏ 无常数项 ⇒ b≡0，
    # 斜率 w=n_az/(8π²) 场景不变。从随机初始自由训练会学到场景相关截距（train≈0.43/val≈1.2）
    # 而把 w 拉偏到 ~5.3、跨场景劣化；从解析解初始化则稳在最优、复现经典相位梯度法。
    n_az = val_raw[0].shape[1] if len(val_raw) else 512
    with torch.no_grad():
        model.net[1].weight.zero_()
        model.net[1].weight[0, 0] = n_az / (8.0 * np.pi ** 2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_ft = torch.from_numpy(train_feats)
    train_at = torch.from_numpy(train_alpha)
    val_ft = torch.from_numpy(val_feats)
    val_at = torch.from_numpy(val_alpha)

    N = train_feats.shape[0]
    rows_out = []
    best_corr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb = train_ft[idx].to(device)
            yb = train_at[idx].to(device)
            pred = model(xb)
            loss = loss_fn(pred[:, 0], yb[:, 0])  # 只监督 α₂
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / N

        # 验证
        model.eval()
        with torch.no_grad():
            pred_val = model(val_ft.to(device)).cpu()
        p2 = pred_val[:, 0].numpy()
        t2 = val_at[:, 0].numpy()
        sign_hit = float((np.sign(p2) == np.sign(t2)).mean())
        alpha2_std = float(p2.std())

        corrs = []
        with torch.no_grad():
            for k in range(len(val_raw_sub)):
                a = np.zeros(6); a[0] = float(p2[k])
                comp = apply_phase_compensation(val_raw_sub[k], a, axis=1, n_orders=1)
                corrs.append(corr(comp, val_clean_sub[k]))
        val_corr = float(np.mean(corrs))

        rows_out.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                         "val_corr": round(val_corr, 6),
                         "val_alpha2_std": round(alpha2_std, 6),
                         "val_sign_hit": round(sign_hit, 6)})

        if epoch == 1:
            torch.save(model.state_dict(), f"{args.prefix}_epoch1.pth")

        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), f"{args.prefix}_best.pth")

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={train_loss:.5f}  "
                  f"val_corr={val_corr:.4f}  sign_hit={sign_hit:.3f}  α₂std={alpha2_std:.3f}", flush=True)

    # 保存曲线 + metrics CSV
    plot_curves(rows_out, Path(f"{args.prefix}_curve.png"))
    with open(f"{args.prefix}_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"metrics CSV -> {args.prefix}_metrics.csv  最佳 val_corr={best_corr:.4f}")


if __name__ == "__main__":
    main()
