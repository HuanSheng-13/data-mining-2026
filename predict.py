"""Apply a saved task model to new CSV input without refitting."""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def predict_weather(bundle, frame):
    cols = bundle["features"]
    w = bundle["window"]
    times = pd.to_datetime(frame["date"])
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Provide strictly increasing, unique timestamps.")
    values = frame[cols].astype(float).replace([np.inf, -np.inf, -9999], np.nan)
    values = values.ffill().fillna(bundle["initial_fill_medians"]).to_numpy()
    rows = []
    x = []
    for t in range(w - 1, len(frame)):
        if not np.all(
            np.diff(times.iloc[t - w + 1 : t + 1].to_numpy()) == np.timedelta64(10, "m")
        ):
            continue
        rows.append(t)
        x.append(values[t - w + 1 : t + 1].reshape(-1))
    if not x:
        raise ValueError("At least 12 contiguous observations are required.")
    x = np.asarray(x)
    base = x[:, -len(cols) + cols.index(bundle["target"])]
    y = base if bundle["model"] is None else base + bundle["model"].predict(x)
    return pd.DataFrame(
        {
            "forecast_date": times.iloc[rows].to_numpy() + np.timedelta64(10, "m"),
            "predicted_CO2_ppm": y,
        }
    )


def predict_thyroid(bundle, frame):
    x = frame[bundle["features"]].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Thyroid features must be finite.")
    scores = -bundle["model"].score_samples(
        bundle["scaler"].transform(frame[bundle["features"]])
    )
    return pd.DataFrame(
        {
            "anomaly_score": scores,
            "prediction": (scores > bundle["threshold"]).astype(int),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, choices=[3, 4], required=True)
    p.add_argument(
        "--model", type=Path, required=True, help="Trusted local joblib bundle"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bundle = joblib.load(a.model)
    frame = pd.read_csv(a.input)
    result = (predict_weather if a.task == 3 else predict_thyroid)(bundle, frame)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
    print(f"Saved {len(result)} predictions to {a.output}")
