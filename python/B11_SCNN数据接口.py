"""SCNN 一维滑窗数据接口；依据 CN110333489A，项目约定见随附说明。"""

from __future__ import annotations

import numpy as np


def nonnegative_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not array.size or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} 必须是非空、有限、非负的实数幅度数组")
    return array.astype(np.float64)


def linear_upsample(values: np.ndarray, factor: int) -> np.ndarray:
    """保留原始采样点及两个端点，不作周期延拓。"""
    values = nonnegative_array(values, "values")
    if values.ndim != 1 or values.size < 2:
        raise ValueError("输入须为至少两个点的一维序列")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("插值倍数须为正整数")
    return np.interp(np.arange((values.size - 1) * factor + 1) / factor,
                     np.arange(values.size), values)


def make_windows(signal: np.ndarray, target: np.ndarray, *, factor: int = 8,
                 window: int = 50, stride: int = 8, amax: float = 1000.0) -> dict:
    """非负幅度序列 → 单通道窗口与中心标量；尺度只由输入窗口计算。"""
    signal = nonnegative_array(signal, "signal")
    target = nonnegative_array(target, "target")
    if signal.shape != target.shape or signal.ndim != 1:
        raise ValueError("输入和标签须为等长一维序列")
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (window, stride)):
        raise ValueError("窗长和步长须为正整数")
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("amax 须为有限正数")
    interpolated = linear_upsample(signal, factor)
    label = linear_upsample(target, factor)
    if window > interpolated.size:
        raise ValueError("窗长超过插值后的序列长度")
    starts = np.arange(0, interpolated.size - window + 1, stride, dtype=np.int32)
    raw = np.lib.stride_tricks.sliding_window_view(interpolated, window)[starts]
    # 偶数窗长 50 采用零基索引 25，左侧 25 点、右侧 24 点。
    centers = starts + window // 2
    scales = raw.max(axis=1)
    # 全零输入保留 scale=0 并使用除数 1，避免 NaN，不凭标签改输入尺度。
    divisors = np.where(scales > 0, scales, 1.0)
    normalized = raw * (amax / divisors[:, None])
    normalized_target = label[centers] * (amax / divisors)
    if not np.isfinite(normalized).all() or not np.isfinite(normalized_target).all():
        raise ValueError("归一化发生数值溢出")
    return {
        "input": normalized[:, None].astype(np.float32),
        "target": normalized_target[:, None].astype(np.float32),
        "scale": scales.astype(np.float64),
        "start_interpolated": starts,
        "center_interpolated": centers,
        "center_original": centers.astype(np.float64) / factor,
        "zero_input": (scales == 0).astype(np.uint8),
    }


def restore_centers(prediction: np.ndarray, scales: np.ndarray, amax: float = 1000.0) -> np.ndarray:
    """撤销输入窗口的缩放；返回中心点幅度，不补造未预测的边界。"""
    prediction = nonnegative_array(prediction, "prediction").reshape(-1)
    scales = nonnegative_array(scales, "scales").reshape(-1)
    if prediction.shape != scales.shape or not np.isfinite(amax) or amax <= 0:
        raise ValueError("预测、尺度或 amax 不匹配")
    return prediction * np.where(scales > 0, scales, 1.0) / amax


def directional_label(shape: tuple[int, int], targets: np.ndarray,
                      spacing_xy: np.ndarray, axis: int) -> np.ndarray:
    """只截去当前方向的旁瓣；另一方向仍保留 sinc 响应。"""
    if axis not in (0, 1):
        raise ValueError("axis 只能为 0（方位）或 1（距离）")
    targets = np.asarray(targets)
    spacing = np.asarray(spacing_xy)
    if (len(shape) != 2 or min(shape) < 1 or targets.ndim != 2 or targets.shape[1] != 4
            or not len(targets) or not np.isfinite(targets).all()
            or spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError("场景、点目标或第一零点间隔无效")
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    label = np.zeros(shape, dtype=np.complex128)
    for x, y, amplitude, phase in targets:
        response = np.sinc((xx - x) / spacing[0]) * np.sinc((yy - y) / spacing[1])
        mask = np.abs(yy - y) < spacing[1] if axis == 0 else np.abs(xx - x) < spacing[0]
        label += amplitude * np.exp(1j * phase) * response * mask
    return label.astype(np.complex64)


def combine_amplitude(original: np.ndarray, rsva: np.ndarray, scnn: np.ndarray,
                      threshold: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """专利 [0264] 的幅度域分支约定；RSVA 结果和 CNN 预测须由调用者提供。"""
    original = nonnegative_array(original, "original")
    rsva = nonnegative_array(rsva, "rsva")
    scnn = nonnegative_array(scnn, "scnn")
    if original.shape != rsva.shape or original.shape != scnn.shape:
        raise ValueError("三个幅度数组尺寸必须相同")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold 须在 [0,1] 内")
    ratio = np.divide(np.abs(rsva - original), original,
                      out=np.full_like(original, np.inf), where=original > 0)
    choose_rsva = ratio > threshold
    result = np.where(choose_rsva, rsva, scnn)
    # 专利未规定零分母；项目对原始零幅度输出零，并单独记录约定。
    result[original == 0] = 0
    return result.astype(np.float32), choose_rsva
