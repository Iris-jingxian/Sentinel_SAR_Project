"""固定场景清单加载、分区损失和回归评测。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from B11_数据工具 import file_hash


def load_split(manifest_path: Path, split: str):
    if split not in ("train", "val", "test"):
        raise ValueError("未知集合")
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec["config"]["window"] != 50 or spec["config"]["amax"] != 1000:
        raise ValueError("SCNN 数据配置不匹配")
    groups = {}
    ids = set()
    for record in spec["records"]:
        sid, group, category = record["scene_id"], record["source_scene_id"], record["split"]
        if sid in ids or (group in groups and groups[group] != category):
            raise ValueError("场景重复或来源跨集合")
        if category not in ("train", "val", "test"):
            raise ValueError("清单含未知集合")
        ids.add(sid)
        groups[group] = category
    chunks, labels, metadata = [], [], []
    for record in spec["records"]:
        if record["split"] != split:
            continue
        p = (manifest_path.parent / record["file"]).resolve()
        if p.parent != manifest_path.parent.resolve() or file_hash(p) != record["file_sha256"]:
            raise ValueError("窗口路径或校验值错误")
        with np.load(p, allow_pickle=False) as d:
            x, y = d["input"].copy(), d["target"].copy()
        if x.shape != (len(y), 1, 50) or y.shape != (len(y), 1):
            raise ValueError("张量形状不匹配")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(y < 0):
            raise ValueError("窗口或标签数值异常")
        if len(x) != record["window_count"]:
            raise ValueError("窗口计数错误")
        chunks.append(x)
        labels.append(y)
        metadata.extend([record["scene_id"]] * len(y))
    if not chunks:
        raise ValueError("集合为空")
    x, y = np.concatenate(chunks), np.concatenate(labels)
    if len(x) != spec["window_counts"][split] or len(chunks) != spec["scene_counts"][split]:
        raise ValueError("集合计数不符")
    return torch.from_numpy(x), torch.from_numpy(y), np.asarray(metadata)


def balanced_loss(prediction, target):
    """分别平均非零标签和零标签的MSE，防止背景数量支配梯度。"""
    error = ((prediction - target) / 1000.0).square()
    positive = target > 0
    terms = [error[mask].mean() for mask in (positive, ~positive) if mask.any()]
    return torch.stack(terms).mean()


def regression_metrics(prediction, target):
    p = np.asarray(prediction, dtype=np.float64).reshape(-1) / 1000
    y = np.asarray(target, dtype=np.float64).reshape(-1) / 1000
    mask = y > 0
    mse = float(np.mean((p - y) ** 2))
    positive = float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else None
    negative = float(np.mean(p[~mask] ** 2)) if (~mask).any() else None
    parts = [v for v in (positive, negative) if v is not None]
    return {"mse": mse, "positive_mse": positive, "background_mse": negative,
            "balanced_mse": float(np.mean(parts)), "positive_count": int(mask.sum()),
            "count": len(y), "negative_prediction_count": int((p < 0).sum())}


@torch.no_grad()
def predict_batches(model, x, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(batch.to(device)).cpu() for batch in x.split(batch_size)])
