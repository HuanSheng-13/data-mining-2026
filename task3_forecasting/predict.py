"""Apply a saved task3_forecasting model without refitting."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trusted local joblib bundle")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    result = predict_weather(bundle, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
