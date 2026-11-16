"""Cache regressions: a calibration split must never inherit fitted weights."""
import importlib
import sys
import os
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np
import torch
from PIL import Image

from task2_image_anomaly.src.autoencoder import AutoEncoderDetector
from task2_image_anomaly.src.dataset import (
    AnomalyImageDataset, split_normal_train_dataset, training_context, resolve_data_root,
)


@pytest.fixture
def normals(tmp_path):
    folder = tmp_path / "hazelnut/train/good"
    folder.mkdir(parents=True)
    for i in range(20):
        Image.new("RGB", (8, 8), (i, i, i)).save(folder / f"{i:03}.png")
    return AnomalyImageDataset(str(tmp_path), "hazelnut", "train", "good")


def test_checkpoint_rejects_other_split_changed_content_and_legacy(normals, tmp_path):
    fit, _ = split_normal_train_dataset(normals, seed=42)
    other_fit, _ = split_normal_train_dataset(normals, seed=123)
    context = training_context(fit)
    other_context = training_context(other_fit)
    assert other_context != context
    model = AutoEncoderDetector(seed=42, epochs=1)
    path = tmp_path / "ae.pt"
    model.save(path, context)
    loaded = AutoEncoderDetector(seed=42, epochs=1).load(path, context)
    assert all(torch.equal(a, b) for a, b in zip(model.model.parameters(), loaded.model.parameters()))
    with pytest.raises(ValueError, match="不匹配"):
        loaded.load(path, other_context)
    Image.new("RGB", (8, 8), "red").save(
        Path(normals.root) / "hazelnut/train/good" / normals.files[fit.indices[0]])
    with pytest.raises(ValueError, match="不匹配"):
        loaded.load(path, training_context(fit))
    with pytest.raises(ValueError, match="不匹配"):
        AutoEncoderDetector(seed=123, epochs=1).load(path, context)
    torch.save({"state_dict": model.model.state_dict(), "config": model.configuration()}, path)
    with pytest.raises(ValueError, match="不匹配"):
        loaded.load(path, context)


@pytest.mark.parametrize("script", ["run_autoencoder", "run_patchcore"])
def test_cli_seed_reaches_model(script, normals, tmp_path):
    # Exercise the actual entry points through dataset split/loader construction.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module(f"task2_image_anomaly.scripts.{script}")
    args = Namespace(data_root=str(tmp_path), calibration_fraction=.1, seed=123,
                     batch_size=2, device="cpu", epochs=1, ae_epochs=1, lr=.001,
                     backbone="ae")
    class StopAtModel(Exception):
        pass
    def construct(**kwargs):
        assert kwargs["seed"] == 123
        raise StopAtModel
    model_target = (f"{module.__name__}.AutoEncoderDetector" if script == "run_autoencoder"
                    else "src.autoencoder.AutoEncoderDetector")
    with patch.object(module, "resolve_data_root", return_value=str(tmp_path)), \
         patch.object(module, "load_test_labels", return_value={}), \
         patch.object(module, "AnomalyImageDataset", return_value=normals), \
         patch.object(module, "TestImageDataset", return_value=normals), \
         patch(model_target, side_effect=construct):
        with pytest.raises(StopAtModel):
            module.run_category(args, "hazelnut", {})


def test_outer_archive_directory_is_resolved(tmp_path):
    inner = tmp_path / "Image_Anomaly_Detection"
    inner.mkdir()
    (inner / "image_anomaly_labels.json").write_text("{}")
    assert Path(resolve_data_root(str(tmp_path))) == inner
    assert Path(resolve_data_root(str(inner))) == inner


def test_cached_cli_matches_fresh_and_retrains_after_seed_change(tmp_path):
    root = tmp_path / "data"
    good = root / "hazelnut/train/good"
    test = root / "hazelnut/test"
    good.mkdir(parents=True)
    test.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (32, 32), (20 * i, 30, 40)).save(good / f"{i}.png")
    labels = {}
    for i, label in enumerate(["good", "bad"]):
        name = f"{i}.png"
        Image.new("RGB", (32, 32), (100 * i, 30, 40)).save(test / name)
        labels[f"hazelnut/test/{name}"] = {"label": label}
    (root / "image_anomaly_labels.json").write_text(json.dumps(labels))
    output = tmp_path / "results/patchcore"
    script = Path(__file__).resolve().parents[1] / "scripts/run_patchcore.py"
    command = [sys.executable, str(script), "--data_root", str(root),
               "--out_dir", str(output), "--backbone", "ae", "--ae_epochs", "1",
               "--batch_size", "2", "--max_pool", "16", "--device", "cpu",
               "--seed"]
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONIOENCODING="utf-8")
    def run(seed):
        return subprocess.run(command + [str(seed)], check=True, capture_output=True,
                              encoding="utf-8", env=environment).stdout
    run(7)
    path = output / "hazelnut/scores.npz"
    with np.load(path) as scores:
        first = scores["y_score"].copy()
    assert "加载匹配的 AE 权重" in run(7)
    with np.load(path) as scores:
        np.testing.assert_array_equal(first, scores["y_score"])
    assert "忽略过期 AE 缓存" in run(123)
    checkpoint = torch.load(tmp_path / "results/autoencoder/hazelnut/autoencoder.pt",
                            weights_only=True)
    assert checkpoint["config"]["seed"] == 123
