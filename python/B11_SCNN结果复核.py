"""用随附权重复算分类场景误差，并逐层抽查数值梯度。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from B11_SCNN网络 import load_scnn
from B11_SCNN训练 import gradient_probe
from B11_SCNN训练工具 import load_split, predict_batches, regression_metrics
from B11_数据工具 import write_csv, write_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


def verify(output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = ROOT / 'data/B11_SCNN滑窗样例/B11_SCNN滑窗清单.json'
    weights = ROOT / 'weights/B11_SCNN.pth'
    model = load_scnn(weights)
    x, y, _ = load_split(manifest, 'train')
    generator = torch.Generator().manual_seed(77)
    groups = [torch.where(y[:, 0] > 0)[0], torch.where(y[:, 0] == 0)[0]]
    indices = torch.cat([g[torch.randperm(len(g), generator=generator)[:32]] for g in groups])
    gradient = gradient_probe(model, x[indices], y[indices])
    xt, yt, scene_ids = load_split(manifest, 'test')
    prediction = predict_batches(model, xt).numpy()
    reloaded = predict_batches(load_scnn(weights), xt).numpy()
    np.testing.assert_array_equal(prediction, reloaded)
    records = json.loads(manifest.read_text(encoding='utf-8'))['records']
    lookup = {r['scene_id']: r['scenario'] for r in records}
    categories = np.asarray([lookup[s] for s in scene_ids])
    rows = []
    for category in sorted(set(categories)):
        mask = categories == category
        for name, values in (
            ('SCNN', prediction), ('identity_center', xt[:, 0, 25].numpy()),
            ('all_zero', np.zeros(len(yt))),
        ):
            rows.append({'scenario': category, 'method': name,
                         'scene_count': len(set(scene_ids[mask])),
                         **regression_metrics(values[mask], yt.numpy()[mask])})
    write_csv(output / 'B11_分类场景对照.csv', rows)
    write_json(output / 'B11_梯度数值核对.json', {
        'weight_sha256': file_hash(weights), 'manifest_sha256': file_hash(manifest),
        'sample_count': len(indices), 'training_window_indices': indices.tolist(),
        **gradient,
    })
    metrics = regression_metrics(prediction, yt.numpy())
    report = {'weight_sha256': file_hash(weights), 'manifest_sha256': file_hash(manifest),
              'reload_max_abs_error': float(np.max(np.abs(prediction - reloaded))),
              'gradient_tensor_count': gradient['checked_tensor_count'],
              'gradient_max_relative_error': gradient['relative_error'],
              'test_metrics': metrics}
    write_json(output / 'B11_权重复核.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/B11_SCNN复核')
    verify(parser.parse_args().output)
