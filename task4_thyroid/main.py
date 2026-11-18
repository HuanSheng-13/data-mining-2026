"""Task 4: normal-only novelty detection with held-out normal calibration."""

import argparse
import sys
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task4_thyroid.utils import (
    ROOT,
    SEED,
    COLORS,
    plt,
    save_json,
    provenance,
    save_figure,
    markdown_table,
)

PRIMARY = "LOF"


def resolve_data_dir(data):
    """Locate either flat CSV files or the archive's nested thyroid directory."""
    base = Path(data)
    for candidate in (base, base / "thyroid"):
        if all((candidate / name).is_file() for name in ("train-set.csv", "test-set.csv")):
            return candidate
    raise FileNotFoundError(f"Cannot locate train-set.csv and test-set.csv under {base}")


def calibrated_threshold(scores, alpha):
    """Conservative finite-sample upper quantile; mark anomaly iff score > threshold."""
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= len(scores) else float("inf")


def evaluate(y, score, threshold):
    pred = (score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(y, score)),
        "PR_AP": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "test_FPR": float(fp / (tn + fp)),
    }


def run(data, out, alpha=0.05, seed=SEED):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = resolve_data_dir(data)
    train_path = data / "train-set.csv"
    test_path = data / "test-set.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    cols = list(train.columns)
    if len(cols) != 6 or set(test.columns) != set(cols + ["label"]):
        raise ValueError("Unexpected thyroid schema")
    if (
        not np.isfinite(train.to_numpy()).all()
        or not np.isfinite(test.to_numpy()).all()
    ):
        raise ValueError("Non-finite thyroid data: review data before fitting")
    # Exact feature duplicates stay in one group to prevent fit/calibration leakage.
    groups = pd.util.hash_pandas_object(train, index=False).to_numpy()
    fit_ids, cal_ids = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            train, groups=groups
        )
    )
    scaler = StandardScaler().fit(train.iloc[fit_ids])
    xf = scaler.transform(train.iloc[fit_ids])
    xc = scaler.transform(train.iloc[cal_ids])
    xt = scaler.transform(test[cols])
    models = {
        "LOF": LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=1),
        "IsolationForest": IsolationForest(
            n_estimators=300, max_samples="auto", random_state=seed, n_jobs=1
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
    }
    # Fixed primary method and fixed comparison parameters; no label-based tuning.
    save_json(
        out / "protocol.json",
        {
            "primary_model": PRIMARY,
            "alpha": alpha,
            "selection": "fixed before evaluation; primary LOF captures local normal density",
            "model_parameters": {k: m.get_params() for k, m in models.items()},
            "fit_rows": fit_ids.tolist(),
            "calibration_rows": cal_ids.tolist(),
        },
    )
    cal_scores = {}
    test_scores = {}
    thresholds = {}
    times = {}
    for name, model in models.items():
        start = perf_counter()
        model.fit(xf)
        cal_scores[name] = -model.score_samples(xc)
        test_scores[name] = -model.score_samples(xt)
        thresholds[name] = calibrated_threshold(cal_scores[name], alpha)
        times[name] = perf_counter() - start
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "features": cols,
                "threshold": thresholds[name],
                "alpha": alpha,
                "anomaly_score": "-model.score_samples(scaler.transform(X))",
                "calibration_scores": cal_scores[name],
            },
            out / f"{name}.joblib",
        )
    # Test labels are only accessed for reporting after all fits/thresholds are fixed.
    if not test.label.isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    y = test.label.to_numpy(dtype=int)
    rows = []
    predictions = pd.DataFrame({"row_index": np.arange(len(test)), "label": y})
    cal_frame = pd.DataFrame({"train_row_index": cal_ids})
    for name, score in test_scores.items():
        rows.append(
            {
                "model": name,
                "primary": name == PRIMARY,
                "alpha": alpha,
                "threshold": thresholds[name],
                **evaluate(y, score, thresholds[name]),
                "fit_score_seconds": times[name],
            }
        )
        predictions[name + "_score"] = score
        predictions[name + "_prediction"] = (score > thresholds[name]).astype(int)
        cal_frame[name + "_score"] = cal_scores[name]
    results = pd.DataFrame(rows)
    results.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    cal_frame.to_csv(out / "normal_calibration_scores.csv", index=False)
    sensitivity = []
    for level in [0.01, 0.05, 0.10]:
        t = calibrated_threshold(cal_scores[PRIMARY], level)
        sensitivity.append(
            {"alpha": level, "threshold": t, **evaluate(y, test_scores[PRIMARY], t)}
        )
    pd.DataFrame(sensitivity).to_csv(out / "threshold_sensitivity.csv", index=False)
    audit = {
        "train_rows": len(train),
        "test_rows": len(test),
        "features": cols,
        "fit_rows": len(fit_ids),
        "calibration_rows": len(cal_ids),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_rows": int(test.duplicated().sum()),
        "test_feature_rows_overlapping_train": int(
            pd.util.hash_pandas_object(test[cols], index=False).isin(groups).sum()
        ),
        "test_normal": int((y == 0).sum()),
        "test_anomaly": int((y == 1).sum()),
        "missing_values": int(train.isna().sum().sum() + test.isna().sum().sum()),
        "fit_calibration_group_overlap": int(
            len(set(groups[fit_ids]) & set(groups[cal_ids]))
        ),
        "policy": "official test rows preserved; duplicate training feature vectors grouped for calibration split",
    }
    save_json(out / "data_quality.json", audit)
    save_json(out / "environment.json", provenance([train_path, test_path], seed))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for color, (name, s) in zip(COLORS, test_scores.items()):
        fpr, tpr, _ = roc_curve(y, s)
        pr, re, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, color=color, label=f"{name} ({roc_auc_score(y,s):.3f})")
        axes[1].plot(
            re, pr, color=color, label=f"{name} ({average_precision_score(y,s):.3f})"
        )
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[1].axhline(y.mean(), color="black", ls=":", label="Prevalence")
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC (AUC in legend)",
    )
    axes[1].set(
        xlabel="Recall", ylabel="Precision", title="Precision-recall (AP in legend)"
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
    save_figure(fig, out, "roc_pr")
    cm = confusion_matrix(y, predictions[PRIMARY + "_prediction"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            color="white" if v > cm.max() / 2 else "black",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Disease"],
        yticklabels=["Normal", "Disease"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"LOF, normal calibration alpha={alpha}",
    )
    save_figure(fig, out, "confusion_matrix")
    report = f"""# 任务四：无监督疾病判断实验记录

## 问题与数据

给定仅包含正常样本的训练集，用正常分布学习新颖性检测器，将偏离正常模式的测试样本判为患病（1），否则判为正常（0）。原文件没有真实医学特征含义，因此只使用 feature_1 至 feature_6，不推断指标名称。

训练 {len(train)} 行，测试 {len(test)} 行，测试正常 {audit['test_normal']} 行、患病 {audit['test_anomaly']} 行（患病率 {y.mean():.2%}）。两文件合计 {len(train)+len(test)} 行，与说明的 3772 一致。无缺失值。训练重复 {audit['train_duplicate_rows']} 行，测试重复 {audit['test_duplicate_rows']} 行；测试中 {audit['test_feature_rows_overlapping_train']} 行特征向量也出现在训练中。官方测试集保持原样，重复不等同于可确认的数据错误，也不能据此判断患者身份。

## 无监督方法与阈值

预先指定 LOF 为主方法（novelty=True，35 邻居），以局部正常密度变化识别异常；固定比较 Isolation Forest（300 棵树）与 RBF One-Class SVM（nu=0.05、gamma=scale）。未使用测试标签选择模型或超参数。

仅将正常训练样本按相同特征向量分组，以随机种子 {seed} 划分拟合 {len(fit_ids)} 行和校准 {len(cal_ids)} 行，相同特征不跨分区。StandardScaler 仅在拟合子集上训练，所有检测器仅在该子集拟合；保留校准子集，不在阈值确定后重训。LOF 的新颖性打分仅用于未参与拟合的校准和测试数据。

所有分数取负 score_samples，越大越异常。设正常校准分数 s₁,…,sₘ，固定 alpha={alpha}，采用第 ceil((m+1)(1-alpha)) 个顺序统计量作阈值，新样本分数严格大于阈值则判异常。这是正常样本分布下的误报控制设计，不是从测试患病比例反推阈值。训练/测试分布偏移或重复依赖使独立同分布假设不完全成立，不能保证实际测试误报率恰为 alpha。

## 测试结果

{markdown_table(results.drop(columns=['fit_score_seconds']))}

ROC-AUC 评价异常排序，PR_AP 是 average precision，并非梯形积分 PR-AUC。患病率低，因此不能仅看 accuracy；同时报告 precision、recall、F1、balanced accuracy 和混淆矩阵。预先指定的主方法始终是 LOF，即使其他方法测试分数更高也不更改主方法。

![ROC 和 PR](roc_pr.png)

![主方法混淆矩阵](confusion_matrix.png)

## 阈值敏感性

以下 alpha=1%、5%、10% 均为预设，仅作评价，不用于根据测试标签调整正式阈值。

{markdown_table(pd.DataFrame(sensitivity)[['alpha','precision','recall','F1','test_FPR']])}

## 分析与限制

LOF 使用局部密度，孤立森林使用随机隔离，One-Class SVM 使用正常样本边界，不同假设可能产生不同的漏报与误报。较宽松阈值通常提升召回并增加误报，需结合上表实际结果判断。只有正常训练样本无法针对患病 F1 做有监督调参。本实验只是课程数据上的异常检测评价；给定数据已经经过脱敏变换，无法确认上游标准化是否使用全量数据，当前流程另行保证新增预处理只用训练子集。没有患者标识，无法证明患者级独立性。
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False), flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/thyroid")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/task4/reproduced")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        run(a.data_dir, a.output_dir, a.alpha, a.seed)
