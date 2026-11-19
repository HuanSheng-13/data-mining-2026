"""Autoencoder 重建误差异常检测实验入口。

用法示例：
    python scripts/run_autoencoder.py --category hazelnut --data_root <路径>
    python scripts/run_autoencoder.py --all --data_root <路径>
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import (AnomalyImageDataset, TestImageDataset, load_test_labels,
                         split_normal_train_dataset, training_context, resolve_data_root)
from src.autoencoder import AutoEncoderDetector
from src.provenance import environment, file_manifest
from src.metrics import (evaluate_binary, normal_quantile_threshold, plot_roc_curves,
                         plot_score_distribution)


def run_category(args, category: str, results_all: dict):
    print(f"\n========== 类别: {category} ==========")
    data_root = resolve_data_root(args.data_root)
    labels = load_test_labels(os.path.join(data_root, "image_anomaly_labels.json"))

    normal_ds = AnomalyImageDataset(data_root, category, "train", subdir="good",
                                    transform=AnomalyImageDataset.ae_transform())
    train_ds, calibration_ds = split_normal_train_dataset(
        normal_ds, calibration_fraction=args.calibration_fraction,
        seed=args.seed if getattr(args, "split_seed", None) is None else args.split_seed
    )
    test_ds = TestImageDataset(data_root, category, labels,
                               transform=AnomalyImageDataset.ae_transform())
    print(f"训练(正常): {len(train_ds)} 张 | 正常校准: {len(calibration_ds)} 张 | "
          f"测试: {len(test_ds)} 张")

    # DataLoader 的 shuffle 也使用显式 Generator，保证每次训练的样本顺序一致。
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=train_generator)
    calibration_loader = DataLoader(calibration_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    t0 = time.time()
    model = AutoEncoderDetector(device=args.device, epochs=args.epochs,
                                lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    history = model.fit(train_loader, verbose=True)
    print(f"训练耗时 {time.time()-t0:.1f}s")
    # 保存权重（供 PatchCore 特征复用 / 复现）
    ckpt_path = os.path.join(args.out_dir, category, "autoencoder.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    model.save(ckpt_path, training_context(train_ds))
    print(f"AE 权重已保存: {ckpt_path}")

    # 阈值仅由未参与拟合的正常样本确定；此处不访问任何测试标签。
    calibration_scores, _ = model.predict(calibration_loader)
    threshold = normal_quantile_threshold(calibration_scores, args.threshold_quantile)
    np.savez(os.path.join(os.path.dirname(ckpt_path), "calibration_scores.npz"),
             normal_scores=calibration_scores,
             files=[normal_ds.files[i] for i in calibration_ds.indices],
             quantile=args.threshold_quantile,
             threshold=threshold)
    print(f"正常校准集 q={args.threshold_quantile:.3f} 阈值: {threshold:.6f}")

    t0 = time.time()
    y_score, _ = model.predict(test_loader)
    y_true = np.asarray([1 if labels[f"{category}/test/{f}"]
                         ["label"] == "bad" else 0 for f in test_ds.files])
    print(f"测试打分耗时 {time.time()-t0:.1f}s")

    metrics = evaluate_binary(y_true, y_score, threshold=threshold)
    metrics["model"] = "Autoencoder"
    metrics["category"] = category
    metrics["final_train_mse"] = float(history[-1])
    metrics["threshold_source"] = "held_out_normal_calibration"
    metrics["threshold_quantile"] = args.threshold_quantile
    print(f"AUROC={metrics['auroc']:.4f} | AP={metrics['ap']:.4f} | "
          f"F1(q{args.threshold_quantile:.2f} normal)={metrics['f1']:.4f} | "
          f"threshold={metrics['threshold']:.4f}")

    out = os.path.join(args.out_dir, category)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(out, "scores.npz"), y_true=y_true, y_score=y_score,
             files=test_ds.files)
    # 逐图结果让指标差异能追溯到具体样本，避免只对比四舍五入后的 F1。
    with open(os.path.join(out, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "y_true", "score", "threshold", "y_pred", "outcome"])
        for name, truth, score in zip(test_ds.files, y_true, y_score):
            pred = int(score >= threshold)
            outcome = ("TP" if pred else "FN") if truth else ("FP" if pred else "TN")
            writer.writerow([name, int(truth), float(score), threshold, pred, outcome])
    fit_files = [normal_ds.files[i] for i in train_ds.indices]
    cal_files = [normal_ds.files[i] for i in calibration_ds.indices]
    paths = ([f"{category}/train/good/{name}" for name in normal_ds.files]
             + [f"{category}/test/{name}" for name in test_ds.files]
             + ["image_anomaly_labels.json"])
    # 不复制原始图像。清单足以区分数据、划分、配置和数值环境差异。
    evidence = {"environment": environment(args.device),
                "configuration": {**model.configuration(),
                    "split_seed": args.seed if getattr(args, "split_seed", None) is None else args.split_seed,
                    "calibration_fraction": args.calibration_fraction,
                    "threshold_quantile": args.threshold_quantile},
                "fit_files": fit_files, "calibration_files": cal_files,
                "test_files": test_ds.files, "training_context": training_context(train_ds),
                "data_manifest": file_manifest(data_root, paths),
                "train_mse_history": history}
    with open(os.path.join(out, "reproducibility.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    plot_roc_curves({f"AE-{category}": (y_true, y_score)},
                    os.path.join(out, "roc_curve.png"))
    plot_score_distribution(y_true, y_score,
                            os.path.join(out, "score_dist.png"),
                            title=f"Autoencoder {category} Score Distribution")

    results_all[category] = metrics
    return metrics


def main():
    p = argparse.ArgumentParser(description="Autoencoder 图像异常检测基线")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--category", type=str, default="hazelnut",
                   choices=["hazelnut", "zipper"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=None,
                   help="独立划分种子；默认跟随 seed，统计初始化波动时固定为42")
    p.add_argument("--threads", type=int, default=None,
                   help="显式设置CPU运算线程；不保证不同平台逐位一致")
    p.add_argument("--calibration_fraction", type=float, default=0.1,
                   help="从正常训练样本中固定划出的阈值校准比例")
    p.add_argument("--threshold_quantile", type=float, default=0.99,
                   help="正常校准分数分位数；高于此阈值判为异常")
    # 统一写入本任务的结果目录，避免与任务 1 等其他任务混在一起。
    p.add_argument("--out_dir", type=str, default="results/task2/autoencoder")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.threads is not None and args.threads < 1):
        p.error("epochs、batch_size、threads 必须为正数")
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    os.makedirs(args.out_dir, exist_ok=True)
    results_all = {}
    cats = ["hazelnut", "zipper"] if args.all else [args.category]
    for cat in cats:
        run_category(args, cat, results_all)

    print("\n========== 汇总 ==========")
    for cat, m in results_all.items():
        print(f"{cat}: AUROC={m['auroc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f}")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                       for kk, vv in v.items()} for k, v in results_all.items()},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
