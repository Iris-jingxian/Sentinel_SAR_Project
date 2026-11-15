"""phase_error.py — 方位向多项式相位误差模型的单一真相源。

相位模型（与老师 demo_defocus.m 的散焦公式等价，仅系数刻度不同）:

    φ(u) = Σ_{k=2}^{K+1} α_k · (2πu)^k ,   u = fftfreq(n_az) ∈ [-0.5, 0.5)
    I_defocus = IFFT_az( FFT_az(I_clean) · exp(+jφ) )    # 注入散焦
    I_clean   = IFFT_az( FFT_az(I_defocus) · exp(-jφ) )  # 补偿 / 反演

与老师 demo_defocus.m 的换算（老师用 φ = Σ dfc_k · na^k, na = -n_az/2 : n_az/2-1）:

    dfc_k = α_k · (2π / n_az)^k ,   α_k = dfc_k · (n_az / 2π)^k

所有脚本（数据生成 / 恢复 / 补偿 / 评估 / 可视化 / 训练）都应从这里 import，
不要在各自文件里复制这套公式，避免再次出现「生成与补偿公式不一致」的 bug。
"""

from __future__ import annotations

import numpy as np


def phase_vector(n_az: int, alpha) -> np.ndarray:
    """方位频域相位误差向量 φ(u)，长度 n_az，顺序与 np.fft.fft 输出一致。

    alpha[k-2] 是 k 阶系数（k = 2 .. len(alpha)+1）。
    """
    u = np.fft.fftfreq(n_az, d=1.0)
    w = 2.0 * np.pi * u
    phi = np.zeros_like(w, dtype=np.float64)
    for i, a in enumerate(alpha):
        phi += float(a) * (w ** (i + 2))
    return phi


def _phase_like(phi: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = phi.shape[0]
    return phi.reshape(shape)


def apply_phase_error(img: np.ndarray, alpha, axis: int = 1) -> np.ndarray:
    """注入散焦：I_defocus = IFFT_az( FFT_az(img) · exp(+jφ) )。"""
    n_az = img.shape[axis]
    phi = phase_vector(n_az, alpha)
    x = np.fft.fft(img, axis=axis)
    y = np.fft.ifft(x * np.exp(1j * _phase_like(phi, axis, img.ndim)), axis=axis)
    return y.astype(np.complex64, copy=False)


def apply_phase_compensation(img: np.ndarray, alpha, axis: int = 1, n_orders: int = None) -> np.ndarray:
    """补偿散焦：I_clean = IFFT_az( FFT_az(img) · exp(-jφ) )。

    n_orders: 只使用前 n_orders 个系数（其余置 0）。训练只监督前 n_orders 阶，未监督的高阶
    预测停留在随机初始化、量级远超真值，用模型预测补偿时必须置 0（否则重聚焦被假高阶相位毁掉）。
    用真实 alpha 补偿时不要传（=全 6 阶，理想上界）。
    """
    n_az = img.shape[axis]
    alpha = np.asarray(alpha, dtype=np.float64)
    if n_orders is not None:
        alpha = alpha.copy()
        alpha[n_orders:] = 0.0
    phi = phase_vector(n_az, alpha)
    x = np.fft.fft(img, axis=axis)
    y = np.fft.ifft(x * np.exp(-1j * _phase_like(phi, axis, img.ndim)), axis=axis)
    return y.astype(np.complex64, copy=False)


def phase_rms(alpha, n_az: int = 1024) -> float:
    """相位误差 RMS (rad)。"""
    phi = phase_vector(n_az, alpha)
    return float(np.sqrt(np.mean(phi ** 2)))


def peak_qpe_rad(alpha) -> float:
    """带边缘(u=±0.5)二次相位峰值 Q = α₂·π² (rad)。"""
    return float(abs(alpha[0]) * np.pi ** 2)


def alpha_to_dfc(alpha, n_az: int) -> np.ndarray:
    """归一化系数 α → 老师 demo_defocus.m 的等效 dfc_k = α_k·(2π/n_az)^k。"""
    alpha = np.asarray(alpha, dtype=np.float64)
    k = np.arange(2, 2 + alpha.size)
    return alpha * (2.0 * np.pi / n_az) ** k


def dfc_to_alpha(dfc, n_az: int) -> np.ndarray:
    """老师 demo_defocus.m 的 dfc_k → 归一化系数 α_k = dfc_k·(n_az/2π)^k。"""
    dfc = np.asarray(dfc, dtype=np.float64)
    k = np.arange(2, 2 + dfc.size)
    return dfc * (n_az / (2.0 * np.pi)) ** k


def phase_vector_torch(n_az: int, alpha):
    """torch 版相位向量（可微），供 train_full.py 的无监督补偿使用。

    alpha: (..., K) tensor，返回 (..., n_az) tensor。
    """
    import torch
    w = 2.0 * torch.pi * torch.fft.fftfreq(n_az, d=1.0).to(alpha.device)
    phi = torch.zeros_like(w)
    for i in range(alpha.shape[-1]):
        phi = phi + alpha[..., i:i + 1] * (w ** (i + 2))
    return phi
