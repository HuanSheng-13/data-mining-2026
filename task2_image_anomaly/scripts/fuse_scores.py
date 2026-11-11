"""分数融合实验：融合 Autoencoder 重建分数与 PatchCore 记忆库距离分数。

用法：python scripts/fuse_scores.py --results_dir results
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import evaluate_binary


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/task2")
    p.add_argument("--alpha_step", type=float, default=0.05)
    args = p.parse_args()

    summary = {}
    for cat in ["hazelnut", "zipper"]:
        ae = np.load(os.path.join(args.results_dir, "autoencoder", cat, "scores.npz"))
        pc = np.load(os.path.join(args.results_dir, "patchcore", cat, "scores.npz"))
        y_true = ae["y_true"]
        s_ae = minmax(ae["y_score"])
        s_pc = minmax(pc["y_score"])

        best = {"alpha": 0.0, "auroc": 0.0, "f1": 0.0}
        for alpha in np.arange(0.0, 1.0 + 1e-9, args.alpha_step):
            fused = alpha * s_pc + (1 - alpha) * s_ae
            m = evaluate_binary(y_true, fused)
            if m["auroc"] > best["auroc"]:
                best = {"alpha": float(alpha), "auroc": m["auroc"], "f1": m["f1"],
                        "ap": m["ap"], "precision": m["precision"],
                        "recall": m["recall"], "accuracy": m["accuracy"],
                        "threshold": m["threshold"]}
        summary[cat] = best
        print(f"{cat}: 最优融合 α={best['alpha']:.2f} | AUROC={best['auroc']:.4f} "
              f"AP={best['ap']:.4f} F1={best['f1']:.4f}")

    with open(os.path.join(args.results_dir, "fusion_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"融合结果已保存到 {args.results_dir}/fusion_summary.json")


if __name__ == "__main__":
    main()
