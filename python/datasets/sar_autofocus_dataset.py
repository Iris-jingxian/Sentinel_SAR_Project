import csv
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

def _load_complex(path: Path) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True)
    key = "input_defocus" if "input_defocus" in mat else next(
        k for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray) and np.iscomplexobj(v)
    )
    return np.asarray(mat[key])

def _load_alpha(path: Path) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True)
    return np.asarray(mat["alpha"], dtype=np.float32).ravel()

def _normalize_complex(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scale = float(np.max(np.abs(img))) + eps
    return img / scale

class SARAutofocusDataset(Dataset):
    def __init__(self, manifest: Path, split: str = "train", root: Path = Path("."), split_json: Optional[Path] = None, normalize: bool = True):
        self.root = Path(root)
        self.normalize = normalize
        with Path(manifest).open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            
        # 如果没有 split_json，就直接用全部数据测试
        if split_json is not None:
            spec = json.loads(Path(split_json).read_text(encoding="utf-8"))
                        # 兼容字典列表和纯字符串列表
            scenes = {x["scene_id"] if isinstance(x, dict) else x for x in spec.get(split, [])}
            rows = [r for r in rows if r["scene_id"] in scenes]

        assert rows, f"split={split} 无样本"
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        ip = Path(r["input_path"]); ip = ip if ip.is_absolute() else self.root / ip
        lp = Path(r["label_path"]); lp = lp if lp.is_absolute() else self.root / lp

        img = _load_complex(ip)
        if self.normalize:
            img = _normalize_complex(img)

        alpha = _load_alpha(lp)
        x = np.stack([img.real, img.imag], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(alpha)
