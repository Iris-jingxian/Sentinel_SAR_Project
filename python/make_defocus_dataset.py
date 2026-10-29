#!/usr/bin/env python3
"""
make_defocus_dataset.py

从清晰复图像注入方位向多项式相位误差，生成自聚焦网络训练样本。

相位误差模型（方位频域，u ∈ [-0.5, 0.5)，w = 2πu）:
  φ(u) = Σ_{k=2}^{K+1} α_k (2πu)^k
  I_defocus = IFFT_az( FFT_az(I_clean) * exp(jφ) )

散焦强度由「二次相位误差」主导：α₂ 通过带边缘（u=±0.5）的二次相位峰值 Q 决定，
  α₂ = Q / π² 。
高阶项（k≥3）只作为带边缘峰值不超过 higher_order_frac·Q 的小扰动，保证二次项
是主要散焦来源（即真正的方位向散焦/展宽），同时高阶系数非零供网络回归。

数据构建（遵循「原图是标签，标签加散焦当输入」）:
  干净原图 I_clean 是标签（真值），I_clean 注入相位误差后得到 I_defocus 作为输入。

输出结构:
  out_dir/
    <scene_id>/
      clean.mat                # 干净原图（真值/标签，同一场景所有样本共用）
      <sample_id>/
        input_defocus.mat      # 散焦复图像（网络输入），complex64 (H,W)
        label_alpha.mat        # 相位误差系数 α，float32 (K,)，α2..α(K+1)
        meta.json
    manifest.csv               # 含 split / clean_path 列
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError

sys.path.append(str(Path(__file__).parent / "python"))
from phase_error import (  # noqa: E402
    alpha_to_dfc,
    apply_phase_error,
    peak_qpe_rad,
    phase_rms,
)

VAR_CANDIDATES = ("roi_complex", "image_complex", "data", "img")

# 各档位「带边缘二次相位峰值」Q = α₂·π² (rad)。
# 实测：Q=8 rad -> 方位向 -3dB 主瓣 ~7 像素；Q=20 rad -> ~21 像素，散焦肉眼明显。
QPE_RANGES = {
    "weak": (1.5, 4.0),
    "medium": (4.0, 10.0),
    "strong": (10.0, 22.0),
}


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
def sample_alpha(rng: np.random.Generator, strength: str, order: int,
                 higher_order_frac: float = 0.15,
                 higher_order_decay: float = 0.5) -> np.ndarray:
    """二次相位主导的相位误差系数采样。

    α₂（二次项）由带边缘二次相位峰值 Q = α₂·π² 决定，Q 在 QPE_RANGES[strength] 内均匀采样。

    高阶项（k=3..order）带边缘相位峰值上限按阶递减：
        peak_k ~ Uniform[0, higher_order_frac · Q · higher_order_decay^(k-3)]
    老师 demo_defocus.m 里高阶项相对二次项很小（三次 ~2.6%、四次 ~0.7%，即每阶约 ×0.25），
    这里默认 decay=0.5 介于「物理真实」与「高阶仍可被网络回归」之间，可用 --higher-order-decay 调节。
    二次项始终是主要散焦来源，高阶只是带符号的小扰动供网络回归。
    """
    if strength not in QPE_RANGES:
        raise ValueError(f"strength 必须是 {list(QPE_RANGES)}")
    lo, hi = QPE_RANGES[strength]
    q = float(rng.uniform(lo, hi))
    alpha = np.zeros(order - 1, dtype=np.float64)
    alpha[0] = float(rng.choice([-1.0, 1.0])) * q / (np.pi ** 2)
    for k in range(3, order + 1):
        peak_max = higher_order_frac * q * (higher_order_decay ** (k - 3))
        peak_k = float(rng.uniform(0.0, peak_max))  # 该阶带边缘相位峰值
        alpha[k - 2] = float(rng.choice([-1.0, 1.0])) * peak_k / (np.pi ** k)
    return alpha


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
    p.add_argument("--samples-per-image", type=int, default=100)
    p.add_argument("--order", type=int, default=7)
    p.add_argument("--strengths", type=str, default="weak,medium,strong")
    p.add_argument("--higher-order-frac", type=float, default=0.15)
    p.add_argument("--higher-order-decay", type=float, default=0.5,
                   help="高阶项带边缘峰值随阶递减的衰减比（老师物理模型约 0.25，默认 0.5）")
    p.add_argument("--snr-db", type=float, default=None)
    p.add_argument("--azimuth-axis", type=int, default=1, choices=[0, 1])
    p.add_argument("--split-json", type=Path, default=None,
                   help="场景划分 JSON（train/val/test 的 scene_id 列表），用于填 manifest 的 split 列")
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


def load_split_map(path: Path) -> dict[str, str]:
    """读取 {scene_id: split} 映射。"""
    spec = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for split, scenes in spec.items():
        if split in {"version", "task"}:
            continue
        for s in scenes:
            mapping[s if isinstance(s, str) else s["scene_id"]] = split
    return mapping


def main() -> None:
    args = parse_args()
    if not (2 <= args.order <= 7):
        raise ValueError("--order 必须在 2..7")
    strengths = [s.strip() for s in args.strengths.split(",") if s.strip()]
    for s in strengths:
        if s not in QPE_RANGES:
            raise ValueError(f"未知强度: {s}")

    split_map = load_split_map(args.split_json) if args.split_json else {}

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for src in iter_images(args.input):
        clean = load_complex(src)
        scene_id = scene_id_from(args.scene_prefix, src)
        scene_dir = args.out / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # 干净原图即「标签」（同一场景所有样本共用），存一份到场景目录
        clean_path = scene_dir / "clean.mat"
        save_complex_mat(clean_path, clean, "roi_complex")

        for i in range(args.samples_per_image):
            strength = strengths[i % len(strengths)]
            alpha = sample_alpha(rng, strength, args.order,
                                 args.higher_order_frac, args.higher_order_decay)
            defocus = add_complex_noise(
                apply_phase_error(clean, alpha, args.azimuth_axis), args.snr_db, rng
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
            q = peak_qpe_rad(alpha)
            meta = {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "source_image": str(src),
                "shape": list(defocus.shape),
                "azimuth_axis": args.azimuth_axis,
                "order": args.order,
                "alpha": alpha.astype(float).tolist(),
                "dfc_equivalent": alpha_to_dfc(alpha, defocus.shape[args.azimuth_axis]).tolist(),
                "alpha_orders": list(range(2, args.order + 1)),
                "peak_qpe_rad": q,
                "phase_rms_rad": rms,
                "defocus_strength": strength,
                "snr_db": args.snr_db,
                "seed": args.seed,
                "dtype": "complex64",
                "label_type": "poly_coeff",
                "clean_path": str(clean_path),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            rows.append({
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": split_map.get(scene_id, ""),
                "input_path": str(input_path),
                "label_path": str(label_path),
                "clean_path": str(clean_path),
                "meta_path": str(meta_path),
                "peak_qpe_rad": q,
                "phase_rms_rad": rms,
                "defocus_strength": strength,
                "snr_db": args.snr_db if args.snr_db is not None else "",
                "source_image": str(src),
            })
            print(f"[OK] {sample_id}  {strength:6s}  Q={q:5.1f} rad  rms={rms:.3f}")

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n共 {len(rows)} 个样本 -> {manifest}")


if __name__ == "__main__":
    main()
