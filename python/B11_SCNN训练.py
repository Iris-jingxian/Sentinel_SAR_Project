"""B-11 小样本训练、梯度验收与独立集合评估。"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from B11_SCNN网络 import SCNNB11, load_scnn
from B11_SCNN训练工具 import load_split, balanced_loss, predict_batches, regression_metrics
from B11_数据工具 import write_csv, write_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


def gradient_probe(model, x, y):
    """逐参数张量检查梯度，并各取最大绝对梯度元素核对中心差分。"""
    double = copy.deepcopy(model).cpu().double().eval()
    xd, yd = x.cpu().double(), y.cpu().double()
    double.zero_grad(set_to_none=True)
    balanced_loss(double(xd), yd).backward()
    rows = []
    eps = 1e-7
    for name, parameter in double.named_parameters():
        g = parameter.grad
        if g is None or not torch.isfinite(g).all() or float(g.norm()) == 0:
            raise ValueError(f"参数 {name} 的梯度无效或为零")
        index = int(g.abs().argmax())
        analytic = float(g.flatten()[index])
        original = float(parameter.detach().flatten()[index])
        values = []
        with torch.no_grad():
            try:
                for direction in (1, -1):
                    parameter.flatten()[index] = original + direction * eps
                    values.append(float(balanced_loss(double(xd), yd)))
            finally:
                parameter.flatten()[index] = original
        numeric = (values[0] - values[1]) / (2 * eps)
        relative = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-12)
        if not np.isfinite(relative) or relative > 1e-4:
            raise ValueError(f"参数 {name} 的有限差分不一致：{relative:.6g}")
        rows.append({"parameter": name, "gradient_norm": float(g.norm()),
                     "finite": True, "flat_index": index, "analytic": analytic,
                     "numeric": numeric, "relative_error": relative})
    return {"layers": rows, "epsilon": eps, "relative_tolerance": 1e-4,
            "relative_error": max(row["relative_error"] for row in rows),
            "checked_tensor_count": len(rows),
            "method": "每个参数张量各抽查一个元素；双精度中心差分", "passed": True}


def overfit_check(x, y, device, steps, seed):
    generator = torch.Generator().manual_seed(seed)
    pos = torch.where(y[:, 0] > 0)[0]
    neg = torch.where(y[:, 0] == 0)[0]
    indices = torch.cat([indices[torch.randperm(len(indices), generator=generator)[:32]] for indices in (pos, neg)])
    xb, yb = x[indices].to(device), y[indices].to(device)
    model = SCNNB11(dropout=0).to(device)
    gradient = gradient_probe(model, xb, yb)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    initial = float(balanced_loss(model(xb), yb).detach())
    history = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps-1:
            history.append({"step": step+1, "balanced_mse": float(loss.detach())})
    model.eval()
    final = float(balanced_loss(model(xb), yb).detach())
    return {"sample_count": len(indices), "indices": indices.tolist(), "initial_loss": initial,
            "final_loss": final, "loss_ratio": final / initial, "passed": final < initial * .05,
            "criterion": "固定64个训练窗口，损失降低至少95%；不是测试集效果",
            "gradient_check": gradient, "history": history}


def train(args):
    if args.epochs < 1 or args.overfit_steps < 1 or args.batch_size < 1:
        raise ValueError("训练轮数、步数和批大小必须为正")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("输出目录非空，请指定新的 --output 保存另一轮实验")
    x, y, scenes = load_split(args.manifest, "train")
    xv, yv, val_scenes = load_split(args.manifest, "val")
    run = {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
           "learning_rate": .001, "weight_decay": 1e-5, "dropout": .1,
           "device": device, "torch": str(torch.__version__), "python": platform.python_version(),
           "manifest_sha256": file_hash(args.manifest),
           "train_scenes": sorted(set(scenes)), "val_scenes": sorted(set(val_scenes)),
           "selection": "minimum validation balanced MSE; test evaluated once after selection"}
    write_json(out / "B11_训练配置.json", run)
    overfit = overfit_check(x, y, device, args.overfit_steps, args.seed)
    write_json(out / "B11_小样本过拟合与梯度.json", overfit)
    print("overfit", overfit["initial_loss"], overfit["final_loss"], overfit["passed"], flush=True)
    if not overfit["passed"]: raise ValueError("小样本过拟合未通过；停止后续训练")
    torch.manual_seed(args.seed)
    model = SCNNB11().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    best, selected, history = float("inf"), None, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(x), generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_loss(model(x[indices].to(device)), y[indices].to(device))
            if not torch.isfinite(loss): raise ValueError("训练损失出现NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
        train_m = regression_metrics(predict_batches(model, x).numpy(), y.numpy())
        val_m = regression_metrics(predict_batches(model, xv).numpy(), yv.numpy())
        history.append({"epoch": epoch+1, "train_balanced_mse": train_m["balanced_mse"],
                        "val_balanced_mse": val_m["balanced_mse"],
                        "train_mse": train_m["mse"], "val_mse": val_m["mse"]})
        if val_m["balanced_mse"] < best:
            best, selected = val_m["balanced_mse"], epoch+1
            torch.save({k: v.detach().cpu() for k,v in model.state_dict().items()}, out / "B11_SCNN最佳权重.pth")
        write_csv(out / "B11_训练日志.csv", history)
        print(f"epoch {epoch+1}: train={train_m['balanced_mse']:.6g} val={best:.6g}", flush=True)
    model = load_scnn(out / "B11_SCNN最佳权重.pth", device)
    before = predict_batches(model, xv[:32])
    reloaded = load_scnn(out / "B11_SCNN最佳权重.pth", device)
    after = predict_batches(reloaded, xv[:32])
    reload_error = float((before - after).abs().max())
    if reload_error > 1e-6: raise ValueError("权重重新加载输出不一致")
    # 测试集合直到模型选择结束才加载。
    xt, yt, test_scenes = load_split(args.manifest, "test")
    predictions = predict_batches(model, xt).numpy()
    reports = []
    for name, pred in (("SCNN_raw", predictions), ("SCNN_nonnegative", np.maximum(predictions,0)),
                       ("identity_center", xt[:,0,25].numpy()), ("all_zero", np.zeros(len(yt)))):
        reports.append({"method": name, **regression_metrics(pred, yt.numpy())})
    write_csv(out / "B11_测试集回归对照.csv", reports)
    scene_rows=[]
    for scene in sorted(set(test_scenes)):
        mask = test_scenes == scene
        scene_rows.append({"scene_id": scene, **regression_metrics(predictions[mask], yt.numpy()[mask])})
    write_csv(out / "B11_测试场景明细.csv", scene_rows)
    np.savez_compressed(out / "B11_测试窗口预测.npz", prediction=predictions, target=yt.numpy(), scene_id=test_scenes)
    result = {"b11_engineering_passed": True, "selected_epoch": selected,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter()-start,
              "reload_max_abs_error": reload_error, "overfit_passed": overfit["passed"],
              "gradient_passed": overfit["gradient_check"]["passed"],
              "checkpoint_sha256": file_hash(out / "B11_SCNN最佳权重.pth"),
              "test_scenes": sorted(set(test_scenes)), "test_results": reports,
              "scope": "SCNN仿真幅度滑窗原型；不是完整CNN+RSVA复现或真实场景性能验收"}
    write_json(out / "B11_训练验收.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/B11_SCNN滑窗样例/B11_SCNN滑窗清单.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/B11_SCNN复现")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--overfit-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", choices=("cpu","cuda","auto"), default="cpu")
    train(parser.parse_args())
