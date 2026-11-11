"""评估指标与可视化工具。"""
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (auc, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score, roc_auc_score)


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float = None) -> dict:
    """计算阈值无关（AUROC/AP）与阈值相关（F1 等）指标。

    Args:
        y_true: 0=正常, 1=异常
        y_score: 异常分数（越大越异常）
        threshold: 若给定则直接用于二值化；否则用 Youden 最优阈值
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    assert y_true.ndim == 1 and y_score.ndim == 1 and len(y_true) == len(y_score)

    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)

    if threshold is None:
        youden = tpr - fpr
        idx = int(np.argmax(youden))
        threshold = float(thr[idx])
    y_pred = (y_score >= threshold).astype(int)

    return {
        "auroc": auroc,
        "ap": ap,
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int((y_true == 1).sum()),
    }


def plot_roc_curves(results: dict, save_path: str):
    """绘制多模型的 ROC 曲线对比图。results: {model_name: (y_true, y_score)}"""
    plt.figure(figsize=(6.4, 5.2))
    for name, (yt, ys) in results.items():
        fpr, tpr, _ = roc_curve(yt, ys)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={roc_auc_score(yt, ys):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Set")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_score_distribution(y_true: np.ndarray, y_score: np.ndarray, save_path: str,
                            title: str = "Anomaly Score Distribution"):
    """绘制正常/异常样本分数分布直方图。"""
    good = y_score[np.asarray(y_true) == 0]
    bad = y_score[np.asarray(y_true) == 1]
    plt.figure(figsize=(6.4, 4.2))
    bins = np.linspace(min(y_score), max(y_score), 40)
    plt.hist(good, bins=bins, alpha=0.6, label=f"normal (n={len(good)})", color="tab:blue")
    plt.hist(bad, bins=bins, alpha=0.6, label=f"anomaly (n={len(bad)})", color="tab:red")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_patch_heatmap(img: np.ndarray, patch_scores: np.ndarray, save_path: str,
                       title: str = "", vmax: float = None):
    """将 patch 级异常分数叠加为热力图。img: HxWx3 0-255 原图。"""
    h, w = img.shape[:2]
    score_map = patch_scores.reshape(64, 64)
    # 上采样到原图尺寸
    import torch.nn.functional as F
    import torch
    sm = torch.from_numpy(score_map).float().unsqueeze(0).unsqueeze(0)
    sm = F.interpolate(sm, size=(h, w), mode="bilinear", align_corners=False)
    sm = sm.squeeze().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(img, alpha=0.6)
    im2 = axes[1].imshow(sm, cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def grid_visualize(images: List[np.ndarray], titles: List[str], save_path: str,
                   ncols: int = 4):
    """展示示例图网格。"""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(images[i])
            axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
