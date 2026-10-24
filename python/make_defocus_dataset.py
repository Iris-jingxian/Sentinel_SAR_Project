#!/usr/bin/env python3
"""
make_defocus_dataset.py

从清晰复图像注入多项式方位向相位误差，生成自聚焦网络训练样本。

输入:
  - 清晰复图像 .mat（变量 roi_complex / image_complex / data / img）或 .npy
  - 或包含多张清晰图的目录
输出:
  out_dir/
    <scene_id>/
      <sample_id>/
        input_defocus.mat   # complex single, (H,W)
        label_alpha.mat     # float32, (K,), α2..α(K+1)
        meta.json
    manifest.csv

相位误差模型（方位频域，u ∈ [-0.5, 0.5)）:
  φ(u) = Σ_{k=2}^{K+1} α_k (2πu)^k
  I_defocus = IFFT_az( FFT_az(I_clean) * exp(jφ) )
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError

VAR_CANDIDATES = ("roi_complex", "image_complex", "data", "img")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_complex(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        npz = np.load(path, allow_pickle=False)
        arr = next((npz[k] for k in VAR_CANDIDATES if k in npz), None)
        if arr is None:
            raise KeyError(f"{path} 中未找到 {VAR_CANDIDATES}")
    elif suffix == ".mat":
        try:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
            arr = next((mat[k] for k in VAR_CANDIDATES if k in mat), None)
            if arr is None:
                for k, v in mat.items():
                    if not k.startswith("__") and isinstance(v, np.ndarray) and np.iscomplexobj(v):
                        arr = v
                        break
            if arr is None:
                raise KeyError(f"{path} 中未找到复数变量")
        except (MatReadError, NotImplementedError, ValueError):
            import h5py
            with h5py.File(path, "r") as f:
                arr = next((f[k][()] for k in VAR_CANDIDATES if k in f), None)
                if arr is None:
                    raise KeyError(f"{path} 是 v7.3 但未找到 {VAR_CANDIDATES}")
                if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
                    arr = arr["real"] + 1j * arr["imag"]
    else:
        raise ValueError(f"不支持的输入格式: {path}")

    arr = np.asarray(arr)
    if not np.iscomplexobj(arr):
        raise ValueError(f"{path} 不是复数数组，dtype={arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{path} 期望二维复图像，实际 shape={arr.shape}")
    return arr.astype(np.complex64, copy=False)


def save_complex_mat(path: Path, arr: np.ndarray, name: str) -> None:
    savemat(path, {name: np.asarray(arr, dtype=np.complex64)}, do_compression=True)


def save_alpha_mat(path: Path, alpha: np.ndarray, name: str = "alpha") -> None:
    savemat(path, {name: np.asarray(alpha, dtype=np.float32)}, do_compression=True)


# ---------------------------------------------------------------------------
# 相位误差
# ---------------------------------------------------------------------------
def poly_phase_vector(n_az: int, alpha: np.ndarray) -> np.ndarray:
    u = np.fft.fftfreq(n_az, d=1.0)
    w = 2.0 * np.pi * u
    phi = np.zeros_like(w, dtype=np.float64)
    for i, a in enumerate(alpha):
        phi += float(a) * (w ** (i + 2))
    return phi


def inject_defocus(img: np.ndarray, alpha: np.ndarray, azimuth_axis: int = 1) -> np.ndarray:
    x = np.fft.fft(img, axis=azimuth_axis)
    n_az = x.shape[azimuth_axis]
    phi = poly_phase_vector(n_az, alpha)
    shape = [1, 1]
    shape[azimuth_axis] = n_az
    y = np.fft.ifft(x * np.exp(1j * phi.reshape(shape)), axis=azimuth_axis)
    return y.astype(np.complex64, copy=False)


def phase_rms(alpha: np.ndarray, n_az: int = 1024) -> float:
    u = np.fft.fftfreq(n_az, d=1.0)
    w = 2.0 * np.pi * u
    phi = np.zeros_like(w)
    for i, a in enumerate(alpha):
        phi += float(a) * (w ** (i + 2))
    return float(np.sqrt(np.mean(phi ** 2)))


def sample_alpha(rng: np.random.Generator, strength: str, order: int) -> np.ndarray:
    ranges = {"weak": (0.3, 0.8), "medium": (0.8, 1.8), "strong": (1.8, 4.0)}
    if strength not in ranges:
        raise ValueError(f"strength 必须是 {list(ranges)}")
    lo, hi = ranges[strength]
    target = rng.uniform(lo, hi)
    alpha = rng.normal(size=order - 1) / np.arange(2, order + 1) ** 0.5
    base = phase_rms(alpha)
    if base < 1e-12:
        alpha = np.zeros(order - 1)
        alpha[0] = target / (2 * np.pi * 0.5) ** 2
        return alpha
    return alpha * (target / base)


def add_complex_noise(img: np.ndarray, snr_db, rng: np.random.Generator) -> np.ndarray:
    if snr_db is None or not np.isfinite(snr_db):
        return img
    p_sig = float(np.mean(np.abs(img) ** 2))
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    sigma = math.sqrt(p_noise / 2.0)
    noise = sigma * (rng.normal(size=img.shape) + 1j * rng.normal(size=img.shape))
    return (img + noise).astype(np.complex64, copy=False)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene-prefix", type=str, default="sim_phase")
    p.add_argument("--samples-per-image", type=int, default=20)
    p.add_argument("--order", type=int, default=7)
    p.add_argument("--strengths", type=str, default="weak,medium,strong")
    p.add_argument("--snr-db", type=float, default=None)
    p.add_argument("--azimuth-axis", type=int, default=1, choices=[0, 1])
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def iter_images(path: Path):
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in {".mat", ".npy", ".npz"}:
                yield p


def scene_id_from(prefix: str, src: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", src.stem)
    return f"{prefix}_{stem}" if prefix else stem


def main() -> None:
    args = parse_args()
    if not (2 <= args.order <= 7):
        raise ValueError("--order 必须在 2..7")
    strengths = [s.strip() for s in args.strengths.split(",") if s.strip()]
    for s in strengths:
        if s not in {"weak", "medium", "strong"}:
            raise ValueError(f"未知强度: {s}")

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for src in iter_images(args.input):
        clean = load_complex(src)
        scene_id = scene_id_from(args.scene_prefix, src)
        scene_dir = args.out / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.samples_per_image):
            strength = strengths[i % len(strengths)]
            alpha = sample_alpha(rng, strength, args.order)
            defocus = add_complex_noise(
                inject_defocus(clean, alpha, args.azimuth_axis), args.snr_db, rng
            )

            sample_id = f"{scene_id}_{i:04d}"
            sample_dir = scene_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            input_path = sample_dir / "input_defocus.mat"
            label_path = sample_dir / "label_alpha.mat"
            meta_path = sample_dir / "meta.json"

            save_complex_mat(input_path, defocus, "input_defocus")
            save_alpha_mat(label_path, alpha, "alpha")

            rms = phase_rms(alpha)
            meta = {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "source_image": str(src),
                "shape": list(defocus.shape),
                "azimuth_axis": args.azimuth_axis,
                "order": args.order,
                "alpha": alpha.astype(float).tolist(),
                "alpha_orders": list(range(2, args.order + 1)),
                "phase_rms_rad": rms,
                "defocus_strength": strength,
                "snr_db": args.snr_db,
                "seed": args.seed,
                "dtype": "complex64",
                "label_type": "poly_coeff",
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            rows.append({
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": "",
                "input_path": str(input_path),
                "label_path": str(label_path),
                "meta_path": str(meta_path),
                "phase_rms_rad": rms,
                "defocus_strength": strength,
                "snr_db": args.snr_db if args.snr_db is not None else "",
                "source_image": str(src),
            })
            print(f"[OK] {sample_id}  {strength:6s}  rms={rms:.3f}")

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n共 {len(rows)} 个样本 -> {manifest}")


if __name__ == "__main__":
    main()
