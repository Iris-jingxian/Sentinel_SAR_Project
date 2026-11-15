"""features.py — 场景不变自聚焦特征（谱相关 / 相位梯度斜率）。

经典相位梯度法（classical_baseline.py）里  G = mean_r( X·conj(roll(X,1)) )  这一步，
把场景自身 clean 谱相位沿 range 平均抵消掉，只留下相位误差梯度（斜率 ∝ α₂，符号 = α₂ 符号）。
RAFnet 塌缩的根因就是没做这步、直接吃原始 [实,虚] 谱（场景相关）。

关键结论（2026-11-14 实测）：
  相位梯度 dphi(u) = 8π²α₂u/n_az + 场景噪声(u)，其中**只有「斜率 c1」这一个自由度是场景不变的**；
  c2（二次项）≈1.7 是场景相关常数、c3+ 更是场景噪声。所以任何高维模型（512 维曲线、6 维多项式系数）
  都会过拟合场景噪声、跨场景崩掉（test corr≈0.5），只有「只取斜率」的经典法能到 0.857。

因此网络输入就是斜率（1 个标量），模型学会「斜率 → α₂」的线性映射，等价经典法、但以学习方式得到。
"""
import numpy as np


def spectral_correlation(img: np.ndarray, axis: int = 1) -> np.ndarray:
    """G(u) = mean_range( X[u]·conj(X[u-Δu]) )，复 1D，长度 n_az。"""
    X = np.fft.fft(img, axis=axis)
    return np.mean(X * np.conj(np.roll(X, 1, axis=axis)), axis=1 - axis)


def phase_gradient_slope(img: np.ndarray, axis: int = 1) -> float:
    """相位梯度斜率 c1 = d(angle G)/du，场景不变的唯一自由度（∝ α₂）。

    推导：angle(G(u)) ≈ dφ/du·Δu = 8π²·α₂·u / n_az，故 slope = 8π²α₂/n_az。
    掩掉 |u|≥0.4（roll 断点 / Nyquist 边界污染）。
    """
    img = img / (float(np.max(np.abs(img))) + 1e-8)
    n_az = img.shape[axis]
    G = spectral_correlation(img, axis)
    dphi = np.unwrap(np.angle(G))
    u = np.fft.fftfreq(n_az)
    keep = np.abs(u) < 0.45  # 同 classical_baseline.py；polyfit 最小二乘对 roll 断点鲁棒
    slope = float(np.polyfit(u[keep], dphi[keep], 1)[0])
    return slope


def phase_gradient_feature(img: np.ndarray, axis: int = 1) -> np.ndarray:
    """网络输入：斜率标量，shape (1,)。模型学 α₂ = w·slope + b（w 应收敛到 n_az/(8π²)）。

    物理界截断：α₂∈[-2.2, 2.2] ⇒ slope = α₂·8π²/n_az ∈ [-0.34, 0.34]。
    |slope|>1 是相位解卷失败（dphi 绕了 2π 造成的伪斜率，实测能到 ±10），
    会让 MSE 被 (±66)²≈4400 的残差支配、把模型压到 0。截到 ±0.5 保留全部正常样本、剔除解卷离群。
    """
    slope = float(np.clip(phase_gradient_slope(img, axis), -0.5, 0.5))
    return np.asarray([slope], dtype=np.float32)
