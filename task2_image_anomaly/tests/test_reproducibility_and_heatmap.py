"""Pytest regressions for initialization, calibration and heatmap grids."""

import numpy as np
import torch

from task2_image_anomaly.src.autoencoder import AutoEncoderDetector
from task2_image_anomaly.src.metrics import (
    evaluate_binary,
    normal_quantile_threshold,
    save_patch_heatmap,
)
from task2_image_anomaly.src.patchcore import PatchCore


def test_seed_is_applied_before_autoencoder_initialization():
    first = AutoEncoderDetector(device="cpu", seed=42)
    second = AutoEncoderDetector(device="cpu", seed=42)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(left, right)


def test_threshold_comes_from_normal_calibration_not_test_labels():
    threshold = normal_quantile_threshold(np.array([0.1, 0.2, 0.4, 0.5]), 0.99)
    metrics = evaluate_binary(np.array([0, 1]), np.array([0.2, 0.8]), threshold)
    assert metrics["threshold"] == threshold


def test_heatmap_uses_actual_32_by_32_patch_grid(tmp_path):
    # A WideResNet-style grid has 1024 patches, rather than the AE's 4096.
    def feature_extractor(images):
        count = images.shape[0] * 1024 * 4
        return torch.arange(count, dtype=torch.float32).reshape(images.shape[0], 1024, 4)

    model = PatchCore(device="cpu", max_pool=16, coreset_ratio=0.5,
                      feature_extractor=feature_extractor)
    model.build_memory_bank(model.extract_features(torch.zeros(2, 3, 256, 256)))
    _, patch_scores = model.predict(torch.zeros(1, 3, 256, 256))
    assert model.patch_grid == (32, 32)

    output = tmp_path / "wide.png"
    save_patch_heatmap(np.zeros((256, 256, 3), dtype=np.uint8), patch_scores[0],
                       output, patch_grid=model.patch_grid)
    assert output.exists() and output.stat().st_size > 0
