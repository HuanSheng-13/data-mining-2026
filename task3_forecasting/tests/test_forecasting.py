"""Pytest checks for causal weather windows and saved forecasting models."""
import joblib
import numpy as np
import pandas as pd
import pytest
from task3_forecasting.utils import ROOT
from task3_forecasting.main import load_weather, windows, WINDOW, TARGET, metrics
from task3_forecasting.predict import predict_weather


def test_weather_window_boundaries_and_targets():
    d, clean, filled, cols, (b1, b2), audit = load_weather(ROOT / 'data/weather/weather.csv')
    for lo, hi in [(0, b1), (b1, b2), (b2, len(d))]:
        x, y, ids = windows(d.date, filled, clean[TARGET].to_numpy(), lo, hi)
        assert x.shape[1] == 252
        assert ids.min() - WINDOW >= lo
        assert ids.max() < hi
        np.testing.assert_allclose(y, clean[TARGET].iloc[ids])
        for i in ids:
            assert (d.date.iloc[i - 12:i + 1].diff().iloc[1:] == pd.Timedelta(minutes=10)).all()
    assert audit['invalid_by_feature'][TARGET] == 34


def test_future_is_not_an_input():
    dates = pd.date_range('2020-01-01', periods=40, freq='10min')
    a = np.arange(40 * 21, dtype=float).reshape(40, 21)
    y = np.arange(40, dtype=float)
    x, _, _ = windows(dates, a, y, 0, 40)
    changed = a.copy()
    changed[20:] = 1000000000.0
    z, _, _ = windows(dates, changed, y, 0, 40)
    np.testing.assert_array_equal(x[:9], z[:9])


def test_saved_forecast_matches_test():
    d, _, _, _, _, _ = load_weather(ROOT / 'data/weather/weather.csv')
    truth = pd.read_csv(ROOT / 'results/task3/reproduced/test_predictions.csv')
    bundle = joblib.load(ROOT / 'results/task3/reproduced/HistGB.joblib')
    sample = truth.iloc[100]
    i = int(sample.row_index)
    result = predict_weather(bundle, d.iloc[:i])
    assert float(result.predicted_CO2_ppm.iloc[-1]) == pytest.approx(sample.HistGB, abs=1e-09, rel=0)
    assert pd.Timestamp(result.forecast_date.iloc[-1]) == pd.Timestamp(sample.date)


def test_metrics_recompute_from_predictions():
    w = pd.read_csv(ROOT / 'results/task3/reproduced/test_predictions.csv')
    for _, r in pd.read_csv(ROOT / 'results/task3/reproduced/metrics.csv').iterrows():
        for k, v in metrics(w.actual_CO2_ppm, w[r.model]).items():
            assert v == pytest.approx(r[k], abs=1e-10, rel=0)
