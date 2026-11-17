"""Task 3: causal 12x21 weather windows -> next-step CO2 concentration."""

import argparse
import sys
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task3_forecasting.utils import (
    ROOT,
    SEED,
    COLORS,
    plt,
    save_json,
    provenance,
    save_figure,
    markdown_table,
)

TARGET = "CO2 (ppm)"
WINDOW = 12


def load_weather(path):
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["date"], errors="raise")
    if not dates.is_monotonic_increasing:
        raise ValueError("Weather timestamps must be chronological.")
    duplicate = dates.duplicated(keep="first")
    d = raw.loc[~duplicate].copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    cols = [c for c in d.columns if c != "date"]
    if len(cols) != 21 or TARGET not in cols:
        raise ValueError("Expected 21 weather indicators including CO2 (ppm).")
    values = d[cols].astype(float)
    invalid = ~np.isfinite(values) | values.eq(-9999)
    clean = values.mask(invalid)
    n = len(d)
    b1, b2 = int(n * 0.6), int(n * 0.8)
    # A valid past observation is available before a future forecast is made.
    # No interpolation/backfill; initial gaps use training-only medians.
    medians = clean.iloc[:b1].median()
    if medians.isna().any():
        raise ValueError("A training feature is entirely missing.")
    filled = clean.ffill().fillna(medians)
    audit = {
        "raw_rows": len(raw),
        "rows_after_timestamp_dedup": n,
        "duplicate_timestamps_removed": int(duplicate.sum()),
        "non_10minute_intervals": int(
            d.date.diff().iloc[1:].ne(pd.Timedelta(minutes=10)).sum()
        ),
        "invalid_by_feature": invalid.sum().astype(int).to_dict(),
        "features": cols,
        "raw_boundaries": [0, b1, b2, n],
        "first_timestamp": str(d.date.iloc[0]),
        "last_timestamp": str(d.date.iloc[-1]),
        "policy": "keep first duplicate; causal forward fill inputs; never impute targets; reject non-contiguous windows",
    }
    return d, clean, filled, cols, (b1, b2), audit


def windows(dates, filled, targets, start, stop):
    """Each window and its target lie wholly inside the supplied raw-row partition."""
    arr = np.asarray(filled, dtype=float)
    xs, ys, ids = [], [], []
    ts = np.asarray(dates, dtype="datetime64[ns]")
    step = np.timedelta64(10, "m")
    for t in range(start + WINDOW, stop):
        if not np.isfinite(targets[t]):
            continue
        if not np.all(np.diff(ts[t - WINDOW : t + 1]) == step):
            continue
        xs.append(arr[t - WINDOW : t].reshape(-1))
        ys.append(targets[t])
        ids.append(t)
    if not xs:
        raise ValueError("No eligible windows in this partition.")
    return np.asarray(xs), np.asarray(ys), np.asarray(ids)


def metrics(y, pred):
    return {
        "MAE_ppm": float(mean_absolute_error(y, pred)),
        "RMSE_ppm": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def build_model(kind, params, seed):
    if kind == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    return make_pipeline(
        StandardScaler(),
        HistGradientBoostingRegressor(
            **params, early_stopping=False, random_state=seed
        ),
    )


def run(data, out, seed=SEED):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d, clean, filled, cols, (b1, b2), audit = load_weather(data)
    y = clean[TARGET].to_numpy()
    co2 = cols.index(TARGET)
    train = windows(d.date, filled, y, 0, b1)
    valid = windows(d.date, filled, y, b1, b2)
    test = windows(d.date, filled, y, b2, len(d))
    parts = {"train": train, "validation": valid, "test": test}
    audit["splits"] = {
        k: {
            "samples": len(v[1]),
            "first_target": str(d.date.iloc[v[2][0]]),
            "last_target": str(d.date.iloc[v[2][-1]]),
        }
        for k, v in parts.items()
    }

    # Includes all 21 variables at all 12 lags. Models predict a residual on top of persistence.
    def baseline(x):
        return x[:, -len(cols) + co2]

    xtr, ytr, _ = train
    xv, yv, _ = valid
    candidates = [("Persistence", {})]
    candidates += [("Ridge", {"alpha": a}) for a in [0.1, 1.0, 10.0, 100.0, 1000.0]]
    candidates += [
        (
            "HistGB",
            {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": leaves,
                "l2_regularization": 10.0,
                "min_samples_leaf": 40,
            },
        )
        for leaves in [7, 15]
    ]
    scores = []
    for kind, params in candidates:
        start = perf_counter()
        if kind == "Persistence":
            pred = baseline(xv)
        else:
            model = build_model(kind, params, seed)
            model.fit(xtr, ytr - baseline(xtr))
            pred = baseline(xv) + model.predict(xv)
        row = {
            "model": kind,
            "params": params,
            **metrics(yv, pred),
            "fit_seconds": perf_counter() - start,
        }
        scores.append(row)
        print("validation", row, flush=True)
    # This decision is made solely from validation results, before test prediction.
    best_by_kind = {
        kind: min(
            [s for s in scores if s["model"] == kind], key=lambda s: s["RMSE_ppm"]
        )
        for kind in ["Persistence", "Ridge", "HistGB"]
    }
    selected = min(best_by_kind.values(), key=lambda s: s["RMSE_ppm"])["model"]
    save_json(
        out / "selection.json",
        {
            "criterion": "minimum validation RMSE",
            "selected_model": selected,
            "candidates": scores,
        },
    )
    # Preserve the raw-partition embargo: concatenate eligible train/validation windows.
    xd = np.concatenate([xtr, xv])
    yd = np.concatenate([ytr, yv])
    xt, yt, it = test
    rows = []
    predictions = pd.DataFrame(
        {"row_index": it, "date": d.date.iloc[it].to_numpy(), "actual_CO2_ppm": yt}
    )
    for kind, choice in best_by_kind.items():
        start = perf_counter()
        pred = baseline(xt)
        model = None
        if kind != "Persistence":
            model = build_model(kind, choice["params"], seed)
            model.fit(xd, yd - baseline(xd))
            pred = pred + model.predict(xt)
        rows.append(
            {
                "model": kind,
                "selected_on_validation": kind == selected,
                **metrics(yt, pred),
                "fit_predict_seconds": perf_counter() - start,
            }
        )
        predictions[kind] = pred
        joblib.dump(
            {
                "model": model,
                "kind": kind,
                "features": cols,
                "window": WINDOW,
                "target": TARGET,
                "residual_on_persistence": True,
                "initial_fill_medians": clean.iloc[:b1].median().to_dict(),
                "input_policy": "replace -9999 by NaN; chronological ffill before constructing windows",
            },
            out / f"{kind}.joblib",
        )
    results = pd.DataFrame(rows)
    results.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame([{**s, "params": str(s["params"])} for s in scores]).to_csv(
        out / "validation_search.csv", index=False
    )
    save_json(out / "data_quality.json", audit)
    save_json(out / "environment.json", provenance([data], seed))
    fig, ax = plt.subplots(figsize=(9, 3.3))
    q = predictions.iloc[:288]
    ax.plot(
        pd.to_datetime(q.date),
        q.actual_CO2_ppm,
        color="black",
        label="Observed",
        lw=1.2,
    )
    for color, kind, style in zip(
        COLORS, dict.fromkeys(["Persistence", selected]), [":", "--"]
    ):
        ax.plot(
            pd.to_datetime(q.date), q[kind], label=kind, color=color, ls=style, lw=1
        )
    ax.set(
        ylabel="CO2 (ppm)",
        xlabel="Target timestamp",
        title="First 48 hours of eligible test targets",
    )
    ax.legend()
    fig.autofmt_xdate()
    save_figure(fig, out, "forecast_48h")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(results.model, results.RMSE_ppm, color=COLORS[:3])
    for i, r in results.iterrows():
        ax.text(i, r.RMSE_ppm, f"{r.RMSE_ppm:.3f}", ha="center", va="bottom")
    ax.set(ylabel="Test RMSE (ppm)", ylim=(0, results.RMSE_ppm.max() * 1.2))
    save_figure(fig, out, "test_rmse")
    report = f"""# 任务三：时间序列预测实验记录

目标：过去 12 个时间点的全部 21 个指标组成 252 维输入，预测下一时刻 CO₂ 浓度，单位 ppm。这里预测浓度值，而非浓度差值；模型内部以变化量学习，输出时加回上一时刻浓度。

## 数据与处理

原始 {audit['raw_rows']} 行，删除 {audit['duplicate_timestamps_removed']} 个重复时间戳后 {len(d)} 行；时间跨度 {audit['first_timestamp']} 至 {audit['last_timestamp']}。-9999 是缺失标记，各列数量见 data_quality.json。仅对输入前向填补，首部缺失回退训练中位数；目标缺失不填补。只保留输入至目标均连续、相隔 10 分钟的窗口，跨越采样间断的窗口丢弃。

按原始行时间顺序 60%/20%/20% 划分训练、验证、测试。各分区独立构造窗口，第一批 12 行只作输入，避免窗口重叠到其他分区。标准化统计量只从当前拟合分区学习。测试采用滚动单步预测：预测下一时刻时，前面已观测的测试期指标可作为历史输入，不是一次性预测整段未来。

{markdown_table(pd.DataFrame([{'split':k,**v} for k,v in audit['splits'].items()]))}

## 方法与模型选择

Persistence：下一时刻预测为最近一次 CO₂ 浓度。Ridge：标准化全部滞后特征后做 L2 正则线性回归。HistGB：直方图梯度提升树拟合非线性关系。后两者学习 y(t)-y(t-1)，预测时加回 y(t-1)，仍然使用全部 252 维历史指标。候选网格、验证误差与耗时完整保存于 validation_search.csv。

仅按验证 RMSE 选定 **{selected}**，然后各模型分别使用训练与验证的合格窗口重新拟合，在同一测试集做最终评价。所有模型的标准化器随拟合重新学习；测试数据不参与选择、拟合或早停。

## 测试结果

{markdown_table(results)}

MAE=mean(|y-ŷ|)，RMSE=sqrt(mean((y-ŷ)²))；R² 相对测试集均值计算。误差越小越好，R² 越大越好。最终采用验证集选定的模型，而不是事后选择测试分数最好的模型。

![48 小时预测](forecast_48h.png)

![测试 RMSE](test_rmse.png)

## 分析与限制

短期 CO₂ 具有连续性，Persistence 是必要基线；复杂模型是否有效必须依据与该基线的同口径对比。前 48 小时曲线只作可视化，指标使用全部合格测试窗口。半年时间序列后段分布可能不同于训练期；单次时间划分不足以证明跨年份泛化。正常的真实极端值不作主观删除。逐点预测可用于复核，模型文件仅加载可信来源。
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False), flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/weather")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/task3/reproduced")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        run(a.data_dir / "weather.csv", a.output_dir, a.seed)
