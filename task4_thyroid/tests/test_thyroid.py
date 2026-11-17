"""Pytest checks for normal-only calibration and saved novelty detectors."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from task4_thyroid.utils import ROOT
from task4_thyroid.main import calibrated_threshold, evaluate, resolve_data_dir
from task4_thyroid.predict import predict_thyroid


def test_threshold_matches_conformal_tail():
    cal = np.arange(99, dtype=float)
    scores = np.array([0.0, 92.0, 93.0, 94.0, 95.0, 98.0, 99.0])
    p = (1 + (cal[:, None] >= scores).sum(axis=0)) / (len(cal) + 1)
    np.testing.assert_array_equal(scores > calibrated_threshold(cal, 0.05), p <= 0.05)


def test_thyroid_no_group_overlap_or_label_dependency():
    data_dir = resolve_data_dir(ROOT / 'data/thyroid')
    train = pd.read_csv(data_dir / 'train-set.csv')
    protocol = json.loads((ROOT / 'results/task4/reproduced/protocol.json').read_text())
    h = pd.util.hash_pandas_object(train, index=False)
    assert not set(h.iloc[protocol['fit_rows']]) & set(h.iloc[protocol['calibration_rows']])
    test = pd.read_csv(data_dir / 'test-set.csv')
    b = joblib.load(ROOT / 'results/task4/reproduced/LOF.joblib')
    original = predict_thyroid(b, test)
    test['label'] = 1 - test.label
    pd.testing.assert_frame_equal(original, predict_thyroid(b, test))
    saved = pd.read_csv(ROOT / 'results/task4/reproduced/test_predictions.csv')
    np.testing.assert_allclose(original.anomaly_score, saved.LOF_score, rtol=1e-12)


@pytest.mark.parametrize("nested", [False, True])
def test_thyroid_flat_and_nested_data_directories(tmp_path, nested):
    actual = tmp_path / "thyroid" if nested else tmp_path
    actual.mkdir(exist_ok=True)
    for name in ("train-set.csv", "test-set.csv"):
        (actual / name).touch()
    assert resolve_data_dir(tmp_path) == actual
    assert resolve_data_dir(actual) == actual


def test_metrics_recompute_from_predictions():
    t = pd.read_csv(ROOT / 'results/task4/reproduced/test_predictions.csv')
    for _, r in pd.read_csv(ROOT / 'results/task4/reproduced/metrics.csv').iterrows():
        for k, v in evaluate(t.label, t[r.model + '_score'], r.threshold).items():
            assert v == pytest.approx(r[k], abs=1e-10, rel=0)
