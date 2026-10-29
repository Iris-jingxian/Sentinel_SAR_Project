import sys
import csv
import time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent / "python"))
from datasets.sar_autofocus_dataset import SARAutofocusDataset
from model_rafnet import RAFnet
from phase_error import apply_phase_compensation

def calculate_contrast(img_complex):
    """计算图像对比度 (标准差 / 均值)"""
    amp = np.abs(img_complex)
    return np.std(amp) / (np.mean(amp) + 1e-8)

def calculate_entropy(img_complex):
    """计算图像熵"""
    amp = np.abs(img_complex)
    amp = amp / (np.sum(amp) + 1e-12)
    return -np.sum(amp * np.log2(amp + 1e-12))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAFnet().to(device)
    model.load_state_dict(torch.load("model_best.pth", map_location=device))
    model.eval()

    manifest_path = Path("data/sim/defocus/unified/manifest.csv")
    dataset = SARAutofocusDataset(manifest_path, split="test", normalize=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    print(f"开始批量评估 {len(dataset)} 个测试样本...\n")

    for i, (x, _, alpha_true) in enumerate(dataloader):
        x, alpha_true = x.to(device), alpha_true.to(device)
        row = dataset.rows[i]
        
        # 读取原始复数图
        img_mat = loadmat(row["input_path"])
        raw_complex = img_mat["input_defocus"]

        # 干净原图（标签），用于计算「聚焦后与原图」的相关性
        clean_mat = loadmat(row["clean_path"])
        clean_complex = clean_mat["roi_complex"] if "roi_complex" in clean_mat else next(
            v for k, v in clean_mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and np.iscomplexobj(v)
        )

        # 推理与计时
        start_time = time.time()
        with torch.no_grad():
            pred_alpha = model(x)
        infer_time = (time.time() - start_time) * 1000 # 转换为毫秒

        # 网络补偿（相位模型统一来自 python/phase_error.py）
        img_compensated = apply_phase_compensation(
            raw_complex, pred_alpha.cpu().numpy().flatten(), axis=1)

        # 计算指标
        ent_defocus = calculate_entropy(raw_complex)
        ent_pred = calculate_entropy(img_compensated)
        con_defocus = calculate_contrast(raw_complex)
        con_pred = calculate_contrast(img_compensated)
        corr_pred = float(np.corrcoef(np.abs(img_compensated).ravel(),
                                      np.abs(clean_complex).ravel())[0, 1])

        results.append({
            "sample_id": row["sample_id"],
            "strength": row["defocus_strength"],
            "entropy_defocus": round(ent_defocus, 4),
            "entropy_pred": round(ent_pred, 4),
            "contrast_defocus": round(con_defocus, 4),
            "contrast_pred": round(con_pred, 4),
            "corr_with_clean": round(corr_pred, 6),
            "infer_time_ms": round(infer_time, 2)
        })
        
        if (i+1) % 20 == 0:
            print(f"已处理 {i+1}/{len(dataset)} 个样本...")

    # 保存为 CSV
    csv_path = Path("data/evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n评估完成！结果已保存至 {csv_path}")

if __name__ == "__main__":
    main()
