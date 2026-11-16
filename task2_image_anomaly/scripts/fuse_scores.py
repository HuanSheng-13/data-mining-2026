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
from src.metrics import evaluate_binary, normal_quantile_threshold


def fit_minmax(x: np.ndarray) -> tuple[float, float]:
    """仅在正常校准集上拟合缩放范围，避免测试分数参与预处理。"""
    return float(x.min()), float(x.max())


def apply_minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x - lo) / (hi - lo + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/task2")
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--threshold_quantile", type=float, default=0.99)
    args = p.parse_args()

    summary = {}
    for cat in ["hazelnut", "zipper"]:
        ae = np.load(os.path.join(args.results_dir, "autoencoder", cat, "scores.npz"))
        pc = np.load(os.path.join(args.results_dir, "patchcore", cat, "scores.npz"))
        ae_calibration = np.load(os.path.join(
            args.results_dir, "autoencoder", cat, "calibration_scores.npz"
        ))["normal_scores"]
        pc_calibration = np.load(os.path.join(
            args.results_dir, "patchcore", cat, "calibration_scores.npz"
        ))["normal_scores"]
        y_true = ae["y_true"]
        ae_lo, ae_hi = fit_minmax(ae_calibration)
        pc_lo, pc_hi = fit_minmax(pc_calibration)
        s_ae = apply_minmax(ae["y_score"], ae_lo, ae_hi)
        s_pc = apply_minmax(pc["y_score"], pc_lo, pc_hi)
        c_ae = apply_minmax(ae_calibration, ae_lo, ae_hi)
        c_pc = apply_minmax(pc_calibration, pc_lo, pc_hi)

        # 融合仍是补充性的测试集上界参考；但每个候选的分类阈值只从正常校准集取分位数。
        best = {"alpha": 0.0, "auroc": float("-inf")}
        for alpha in np.arange(0.0, 1.0 + 1e-9, args.alpha_step):
            fused = alpha * s_pc + (1 - alpha) * s_ae
            calibration_fused = alpha * c_pc + (1 - alpha) * c_ae
            threshold = normal_quantile_threshold(calibration_fused, args.threshold_quantile)
            m = evaluate_binary(y_true, fused, threshold=threshold)
            if m["auroc"] > best["auroc"]:
                best = {"alpha": float(alpha), "auroc": m["auroc"], "f1": m["f1"],
                        "ap": m["ap"], "precision": m["precision"],
                        "recall": m["recall"], "accuracy": m["accuracy"],
                        "threshold": m["threshold"],
                        "threshold_source": "held_out_normal_calibration",
                        "threshold_quantile": args.threshold_quantile,
                        "ae_calibration_min": ae_lo, "ae_calibration_max": ae_hi,
                        "patchcore_calibration_min": pc_lo,
                        "patchcore_calibration_max": pc_hi}
        summary[cat] = best
        print(f"{cat}: 最优融合 α={best['alpha']:.2f} | AUROC={best['auroc']:.4f} "
              f"AP={best['ap']:.4f} F1={best['f1']:.4f}")

    with open(os.path.join(args.results_dir, "fusion_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"融合结果已保存到 {args.results_dir}/fusion_summary.json")


if __name__ == "__main__":
    main()
