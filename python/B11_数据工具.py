"""SAR 数据文件、参数表和复数矩阵的公共读写工具。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat


def write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("CSV 记录不能为空")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            })


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def require_complex(array: np.ndarray, name: str = "signal") -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or not np.iscomplexobj(array) or array.size == 0:
        raise ValueError(f"{name} 必须是非空二维复数矩阵")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含有 NaN 或 Inf")
    return np.asarray(array, dtype=np.complex64)


def save_sample(path: Path, arrays: dict[str, np.ndarray], *, mat: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if mat:
        savemat(path.with_suffix(".mat"), arrays, do_compression=True, oned_as="row")


def configure_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def amplitude_db(array: np.ndarray, reference: float, floor_db: float = -50.0) -> np.ndarray:
    floor = 10.0 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(np.abs(array) / max(reference, 1e-12), floor))


def validate_rois(rois: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> None:
    height, width = shape
    if len(rois) < 3:
        raise ValueError("至少需要 3 个互不重叠的 ROI")
    for index, (x, y, w, h) in enumerate(rois):
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError(f"ROI {index + 1} 越界：{(x, y, w, h)}，影像尺寸 {width}×{height}")
        for px, py, pw, ph in rois[:index]:
            if x < px + pw and px < x + w and y < py + ph and py < y + h:
                raise ValueError("ROI 之间有重叠，请选择三个不同区域")
