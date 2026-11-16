"""Checks for leakage boundaries, missing targets, calibration and saved-model consistency."""

import json
import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from experiment_common import ROOT
from predict import predict_weather, predict_thyroid
from task3_forecasting.main import load_weather, windows, WINDOW, TARGET, metrics
from task4_thyroid.main import calibrated_threshold, evaluate, resolve_data_dir


class ProtocolTests(unittest.TestCase):
    def test_thyroid_flat_and_nested_data_directories(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                actual = base / "thyroid" if nested else base
                actual.mkdir(exist_ok=True)
                for name in ("train-set.csv", "test-set.csv"):
                    (actual / name).touch()
                self.assertEqual(resolve_data_dir(base), actual)
                self.assertEqual(resolve_data_dir(actual), actual)

    def test_weather_window_boundaries_and_targets(self):
        d, clean, filled, cols, (b1, b2), audit = load_weather(
            ROOT / "data/weather/weather.csv"
        )
        for lo, hi in [(0, b1), (b1, b2), (b2, len(d))]:
            x, y, ids = windows(d.date, filled, clean[TARGET].to_numpy(), lo, hi)
            self.assertEqual(x.shape[1], 252)
            self.assertGreaterEqual(ids.min() - WINDOW, lo)
            self.assertLess(ids.max(), hi)
            np.testing.assert_allclose(y, clean[TARGET].iloc[ids])
            for i in ids:
                self.assertTrue(
                    (
                        d.date.iloc[i - 12 : i + 1].diff().iloc[1:]
                        == pd.Timedelta(minutes=10)
                    ).all()
                )
        self.assertEqual(audit["invalid_by_feature"][TARGET], 34)

    def test_future_is_not_an_input(self):
        dates = pd.date_range("2020-01-01", periods=40, freq="10min")
        a = np.arange(40 * 21, dtype=float).reshape(40, 21)
        y = np.arange(40, dtype=float)
        x, _, _ = windows(dates, a, y, 0, 40)
        changed = a.copy()
        changed[20:] = 1e9
        z, _, _ = windows(dates, changed, y, 0, 40)
        # Target index 20 uses inputs 8..19, so changing row 20 must not affect it.
        np.testing.assert_array_equal(x[:9], z[:9])

    def test_saved_forecast_matches_test(self):
        d, _, _, _, _, _ = load_weather(ROOT / "data/weather/weather.csv")
        truth = pd.read_csv(ROOT / "results/task3/reproduced/test_predictions.csv")
        bundle = joblib.load(ROOT / "results/task3/reproduced/HistGB.joblib")
        sample = truth.iloc[100]
        i = int(sample.row_index)
        result = predict_weather(bundle, d.iloc[:i])
        self.assertAlmostEqual(
            float(result.predicted_CO2_ppm.iloc[-1]), sample.HistGB, places=9
        )
        self.assertEqual(
            pd.Timestamp(result.forecast_date.iloc[-1]), pd.Timestamp(sample.date)
        )

    def test_threshold_matches_conformal_tail(self):
        cal = np.arange(99, dtype=float)
        scores = np.array([0.0, 92.0, 93.0, 94.0, 95.0, 98.0, 99.0])
        p = (1 + (cal[:, None] >= scores).sum(axis=0)) / (len(cal) + 1)
        np.testing.assert_array_equal(
            scores > calibrated_threshold(cal, 0.05), p <= 0.05
        )

    def test_thyroid_no_group_overlap_or_label_dependency(self):
        data_dir = resolve_data_dir(ROOT / "data/thyroid")
        train = pd.read_csv(data_dir / "train-set.csv")
        protocol = json.loads(
            (ROOT / "results/task4/reproduced/protocol.json").read_text()
        )
        h = pd.util.hash_pandas_object(train, index=False)
        self.assertFalse(
            set(h.iloc[protocol["fit_rows"]])
            & set(h.iloc[protocol["calibration_rows"]])
        )
        test = pd.read_csv(data_dir / "test-set.csv")
        b = joblib.load(ROOT / "results/task4/reproduced/LOF.joblib")
        original = predict_thyroid(b, test)
        test["label"] = 1 - test.label
        pd.testing.assert_frame_equal(original, predict_thyroid(b, test))
        saved = pd.read_csv(ROOT / "results/task4/reproduced/test_predictions.csv")
        np.testing.assert_allclose(original.anomaly_score, saved.LOF_score, rtol=1e-12)

    def test_metrics_recompute_from_predictions(self):
        w = pd.read_csv(ROOT / "results/task3/reproduced/test_predictions.csv")
        for _, r in pd.read_csv(
            ROOT / "results/task3/reproduced/metrics.csv"
        ).iterrows():
            for k, v in metrics(w.actual_CO2_ppm, w[r.model]).items():
                self.assertAlmostEqual(v, r[k], places=10)
        t = pd.read_csv(ROOT / "results/task4/reproduced/test_predictions.csv")
        for _, r in pd.read_csv(
            ROOT / "results/task4/reproduced/metrics.csv"
        ).iterrows():
            for k, v in evaluate(t.label, t[r.model + "_score"], r.threshold).items():
                self.assertAlmostEqual(v, r[k], places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
