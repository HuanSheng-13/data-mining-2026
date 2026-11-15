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


def normal_quantile_threshold(normal_scores: np.ndarray, quantile: float = 0.99) -> float:
    """由独立的正常校准集确定固定阈值，不读取测试标签。"""
    normal_scores = np.asarray(normal_scores, dtype=float)
    if normal_scores.ndim != 1 or len(normal_scores) == 0:
        raise ValueError("normal_scores 必须是一维且非空的正常样本分数")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile 必须在 (0, 1) 内")
    return float(np.quantile(normal_scores, quantile))


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """计算 AUROC/AP 与固定阈值下的分类指标。

    ``threshold`` 必须由训练集划出的正常校准集预先确定。函数刻意不再根据
    ``y_true`` 搜索 Youden 阈值，避免将测试标签泄漏到判定规则中。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    assert y_true.ndim == 1 and y_score.ndim == 1 and len(y_true) == len(y_score)

    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("threshold 必须由独立正常校准集确定")
    y_pred = (y_score >= threshold).astype(int)

    return {
        "auroc": auroc,
        "ap": ap,
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
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
                       patch_grid: Tuple[int, int], title: str = "", vmax: float = None):
    """将 patch 级异常分数叠加为热力图。

    ``patch_grid`` 来自本次特征提取的实际网格，而不是假定所有骨干都是 64×64。
    例如默认 AE 为 64×64，WideResNet 的 layer2 为 32×32。
    """
    h, w = img.shape[:2]
    grid_h, grid_w = patch_grid
    if patch_scores.size != grid_h * grid_w:
        raise ValueError(
            f"patch 分数数目 {patch_scores.size} 与网格 {patch_grid} 不一致"
        )
    score_map = patch_scores.reshape(grid_h, grid_w)
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
