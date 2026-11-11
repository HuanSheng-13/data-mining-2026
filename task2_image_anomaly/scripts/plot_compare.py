"""生成三类方法（AE重建 / 记忆库 / 融合）的对比 ROC 图与指标汇总表（CSV）。"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 中文字体（macOS 苹方 / 冬青黑体）
for fp in ["/System/Library/Fonts/PingFang.ttc",
           "/System/Library/Fonts/STHeiti Light.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        break
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import evaluate_binary


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/task2")
    p.add_argument("--out_dir", type=str, default="results/task2/compare")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fusion = json.load(open(os.path.join(args.results_dir, "fusion_summary.json")))
    rows = []
    for cat in ["hazelnut", "zipper"]:
        ae = np.load(os.path.join(args.results_dir, "autoencoder", cat, "scores.npz"))
        pc = np.load(os.path.join(args.results_dir, "patchcore", cat, "scores.npz"))
        y_true = ae["y_true"]
        s_ae = minmax(ae["y_score"])
        s_pc = minmax(pc["y_score"])
        alpha = fusion[cat]["alpha"]
        s_fu = alpha * s_pc + (1 - alpha) * s_ae

        plt.figure(figsize=(6.4, 5.2))
        for name, s in [("Autoencoder 重建", s_ae),
                        ("特征记忆库 (PatchCore)", s_pc),
                        (f"分数融合 (α={alpha:.2f})", s_fu)]:
            fpr, tpr, _ = roc_curve(y_true, s)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={roc_auc_score(y_true, s):.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{cat} ROC Comparison")
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"roc_compare_{cat}.png"), dpi=200)
        plt.close()

        for name, s in [("AE", s_ae), ("PatchCore", s_pc), ("Fusion", s_fu)]:
            m = evaluate_binary(y_true, s)
            rows.append({"category": cat, "method": name,
                         "auroc": round(m["auroc"], 4), "ap": round(m["ap"], 4),
                         "f1": round(m["f1"], 4), "precision": round(m["precision"], 4),
                         "recall": round(m["recall"], 4), "accuracy": round(m["accuracy"], 4)})

    import csv
    with open(os.path.join(args.out_dir, "metrics_table.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("对比图与指标表已生成:", args.out_dir)


if __name__ == "__main__":
    main()
