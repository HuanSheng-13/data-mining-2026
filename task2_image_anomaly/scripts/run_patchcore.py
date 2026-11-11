"""PatchCore 异常检测实验入口。

用法示例：
    python scripts/run_patchcore.py --category hazelnut --data_root <路径>
    python scripts/run_patchcore.py --category zipper  --data_root <路径>
    # 两个类别一次性跑完：
    python scripts/run_patchcore.py --all --data_root <路径>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import AnomalyImageDataset, TestImageDataset, load_test_labels
from src.patchcore import PatchCore
from src.metrics import (evaluate_binary, plot_roc_curves,
                         plot_score_distribution, save_patch_heatmap)


def denorm(tensor: torch.Tensor, mode: str = "imagenet") -> np.ndarray:
    """反归一化回 0-255 RGB 数组。mode='imagenet' 用于标准归一化，'unit' 用于 [0,1]。"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # HxWx3
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean) * 255.0
    else:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def run_category(args, category: str, results_all: dict):
    print(f"\n========== 类别: {category} ==========")
    data_root = args.data_root
    label_path = os.path.join(data_root, "image_anomaly_labels.json")
    labels = load_test_labels(label_path)

    # AE 骨干使用 [0,1] 像素域变换；ImageNet 骨干使用标准归一化
    transform = (AnomalyImageDataset.ae_transform() if args.backbone == "ae"
                 else None)
    train_ds = AnomalyImageDataset(data_root, category, "train", subdir="good",
                                   transform=transform)
    test_ds = TestImageDataset(data_root, category, labels, transform=transform)
    print(f"训练(正常): {len(train_ds)} 张 | 测试: {len(test_ds)} 张")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    # 0) 构造特征提取器
    if args.backbone == "ae":
        from src.autoencoder import AutoEncoderDetector
        ae = AutoEncoderDetector(device=args.device, epochs=args.ae_epochs,
                                 lr=args.lr, batch_size=args.batch_size, seed=42)
        # 默认复用 run_autoencoder.py 在同一任务目录下保存的权重。
        # out_dir 默认为 results/task2/patchcore，因此权重目录是它的同级 autoencoder。
        task_result_dir = os.path.dirname(os.path.normpath(args.out_dir))
        default_ae = os.path.join(task_result_dir, "autoencoder", category, "autoencoder.pt")
        ae_path = args.ae_weights or default_ae
        if os.path.exists(ae_path):
            ae.load(ae_path)
            print(f"加载 AE 权重: {ae_path}")
        else:
            print("训练自监督 Autoencoder 特征提取器 ...")
            ae.fit(train_loader, verbose=True)
            os.makedirs(os.path.dirname(ae_path), exist_ok=True)
            ae.save(ae_path)
            print(f"AE 权重已保存: {ae_path}")
        feature_extractor = lambda x: ae.model.encode_patches(x.to(args.device))
    else:
        from src.patchcore import WideResNetFeatureExtractor
        feature_extractor = WideResNetFeatureExtractor("wide_resnet50_2", args.device)

    # 1) 特征提取 + 记忆库构建
    t0 = time.time()
    model = PatchCore(device=args.device, coreset_ratio=args.coreset_ratio,
                      max_pool=args.max_pool, top_k=args.top_k,
                      feature_extractor=feature_extractor)
    feats = []
    for x, _ in train_loader:
        feats.append(model.extract_features(x))
    feats = torch.cat(feats, dim=0)
    model.build_memory_bank(feats)
    n_core = model.memory_bank.shape[0]
    print(f"记忆库规模: {n_core} 个 patch 特征 | 特征提取+构建耗时 {time.time()-t0:.1f}s")

    # 2) 测试打分
    t0 = time.time()
    y_true, y_score, patch_maps, imgs = [], [], [], []
    for x, y, fname in test_loader:
        s, p = model.predict(x)
        y_true.extend(y.tolist())
        y_score.extend(s.tolist())
        patch_maps.extend(p)
        dmode = "unit" if args.backbone == "ae" else "imagenet"
        for i in range(len(fname)):
            imgs.append((fname[i], denorm(x[i], dmode)))
    print(f"测试打分耗时 {time.time()-t0:.1f}s")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    metrics = evaluate_binary(y_true, y_score)
    metrics["model"] = f"PatchCore-{args.backbone}"
    metrics["backbone"] = args.backbone
    metrics["category"] = category
    metrics["memory_bank_size"] = int(n_core)
    print(f"AUROC={metrics['auroc']:.4f} | AP={metrics['ap']:.4f} | "
          f"F1(Youden)={metrics['f1']:.4f} | threshold={metrics['threshold']:.4f}")

    # 3) 保存结果
    out = os.path.join(args.out_dir, category)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(out, "scores.npz"),
             y_true=y_true, y_score=y_score,
             patch_maps=np.stack(patch_maps))

    # 4) 可视化：ROC / 分数分布
    plot_roc_curves({f"PatchCore-{category}": (y_true, y_score)},
                    os.path.join(out, "roc_curve.png"))
    plot_score_distribution(y_true, y_score,
                            os.path.join(out, "score_dist.png"),
                            title=f"PatchCore {category} Score Distribution")

    # 5) 可视化：正常/异常热力图示例
    if args.save_heatmaps:
        vmax = float(np.percentile(np.stack(patch_maps), 99.5))
        example_rows = []
        n_good = n_bad = 0
        for fname, arr in imgs:
            is_bad = labels[f"{category}/test/{fname}"]["label"] == "bad"
            key = "bad" if is_bad else "good"
            if key == "bad" and n_bad < args.examples_per_type:
                example_rows.append((fname, arr, key, True))
                n_bad += 1
            elif key == "good" and n_good < args.examples_per_type:
                example_rows.append((fname, arr, key, False))
                n_good += 1
            if n_good >= args.examples_per_type and n_bad >= args.examples_per_type:
                break
        for fname, arr, key, is_bad in example_rows:
            idx = test_ds.files.index(fname)
            pm = patch_maps[idx]
            save_patch_heatmap(arr, pm,
                               os.path.join(out, f"heatmap_{key}_{fname}"),
                               title=f"{category}/{fname} ({key})", vmax=vmax)

    results_all[category] = metrics
    return metrics


def main():
    p = argparse.ArgumentParser(description="PatchCore 图像异常检测")
    p.add_argument("--data_root", type=str, required=True,
                   help="Image_Anomaly_Detection 目录路径")
    p.add_argument("--category", type=str, default="hazelnut",
                   choices=["hazelnut", "zipper"])
    p.add_argument("--all", action="store_true", help="跑全部类别")
    p.add_argument("--backbone", type=str, default="ae",
                   choices=["ae", "wideresnet"],
                   help="特征提取骨干：ae=自监督自编码器特征（默认，无需联网下载权重）；"
                        "wideresnet=ImageNet预训练WideResNet-50（需联网下载权重）")
    p.add_argument("--ae_weights", type=str, default="",
                   help="复用已训练的 AE 权重路径（可选）")
    p.add_argument("--ae_epochs", type=int, default=60, help="AE 训练轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="AE 学习率")
    p.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--coreset_ratio", type=float, default=0.1)
    p.add_argument("--max_pool", type=int, default=10000)
    # 与报告和已提交 results 保持同一口径：Top-3 既能突出局部缺陷，又比单点最大值稳定。
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--out_dir", type=str, default="results/task2/patchcore")
    p.add_argument("--save_heatmaps", action="store_true")
    p.add_argument("--examples_per_type", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_all = {}
    cats = ["hazelnut", "zipper"] if args.all else [args.category]
    for cat in cats:
        run_category(args, cat, results_all)

    print("\n========== 汇总 ==========")
    for cat, m in results_all.items():
        print(f"{cat}: AUROC={m['auroc']:.4f} AP={m['ap']:.4f} F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} Acc={m['accuracy']:.4f}")
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                       for kk, vv in v.items()} for k, v in results_all.items()},
                  f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
