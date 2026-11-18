"""Apply a saved task4_thyroid model without refitting."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trusted local joblib bundle")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    result = predict_thyroid(bundle, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
