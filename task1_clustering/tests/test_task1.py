"""Meaningful data, cache and evaluation boundary checks; no network required."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.datasets import make_blobs

from task1_clustering.clustering import cluster_features, transform_features
from task1_clustering.data import CLASSES, DataError, audit_dataset, read_labels
from task1_clustering.evaluation import evaluate
from task1_clustering.features import cache_key, extract_features, prepare_weights, resolve_device


@pytest.fixture
def small_dataset(tmp_path):
    root = tmp_path / "data" / "Cluster"
    images = root / "dataset"
    images.mkdir(parents=True)
    labels = {}
    rng = np.random.default_rng(7)
    for index, category in enumerate(CLASSES * 2):
        filename = f"{index + 1:03d}.png"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(images / filename)
        labels[filename] = category
    (root / "cluster_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return root.parent


def test_real_archive_layout_and_stable_manifest(small_dataset):
    dataset = audit_dataset(small_dataset, expected_per_class=2)
    assert len(dataset.samples) == 12
    assert [s.sample_id for s in dataset.samples] == [f"{i:03d}.png" for i in range(1, 13)]
    assert dataset.audit["class_counts"] == dict.fromkeys(CLASSES, 2)
    assert dataset.audit["image_sizes"] == {"64x48": 12}
    assert read_labels(dataset) == list(CLASSES * 2)
    assert audit_dataset(dataset.root, expected_per_class=2).fingerprint == dataset.fingerprint


def test_category_folder_layout(tmp_path):
    for index, category in enumerate(CLASSES):
        folder = tmp_path / category
        folder.mkdir()
        Image.new("RGB", (20, 30), (index * 30, 40, 60)).save(folder / "sample.png")
    dataset = audit_dataset(tmp_path, expected_per_class=1)
    assert dataset.audit["layout"] == "category_folders"
    assert read_labels(dataset) == list(CLASSES)


def test_missing_dataset_is_actionable(tmp_path):
    with pytest.raises(DataError, match="Extract Cluster.zip"):
        audit_dataset(tmp_path)


def test_bad_count_is_not_silently_accepted(small_dataset):
    with pytest.raises(DataError, match="Expected 600 images, found 12"):
        audit_dataset(small_dataset)


def test_corrupt_image_is_not_skipped(small_dataset):
    path = small_dataset / "Cluster/dataset/001.png"
    path.write_bytes(b"not an image")
    with pytest.raises(DataError, match="Cannot decode image 001.png"):
        audit_dataset(small_dataset, expected_per_class=2)


@pytest.mark.parametrize("change", ["missing", "unknown", "unbalanced", "duplicate_key"])
def test_invalid_labels(small_dataset, change):
    path = small_dataset / "Cluster/cluster_labels.json"
    labels = json.loads(path.read_text())
    if change == "missing":
        labels.pop("001.png")
    elif change == "unknown":
        labels["001.png"] = "unknown"
    elif change == "unbalanced":
        labels["001.png"] = "cable"
    if change == "duplicate_key":
        text = json.dumps(labels)[:-1] + ', "001.png": "bottle"}'
    else:
        text = json.dumps(labels)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DataError):
        audit_dataset(small_dataset, expected_per_class=2)


def test_duplicates_are_recorded_without_dropping(small_dataset):
    root = small_dataset / "Cluster/dataset"
    (root / "002.png").write_bytes((root / "001.png").read_bytes())
    dataset = audit_dataset(small_dataset, expected_per_class=2)
    assert dataset.audit["duplicate_content_groups"] == [["001.png", "002.png"]]
    assert len(dataset.samples) == 12


def test_cache_alignment_content_invalidation_and_recovery(small_dataset, tmp_path):
    dataset = audit_dataset(small_dataset, expected_per_class=2)
    cache = tmp_path / "cache"
    args = (dataset.samples, "handcrafted", cache, torch.device("cpu"), 4)
    first, first_info = extract_features(*args)
    second, second_info = extract_features(*args)
    np.testing.assert_array_equal(first, second)
    assert not first_info["cache_hit"] and second_info["cache_hit"]
    assert first.shape == (12, 58)
    # Valid arrays in the wrong sample order must never be reused.
    path = cache / f"handcrafted-{first_info['cache_key']}.npz"
    np.savez_compressed(path, sample_ids=np.array([s.sample_id for s in dataset.samples][::-1]), features=first)
    recovered, info = extract_features(*args)
    assert not info["cache_hit"]
    np.testing.assert_array_equal(first, recovered)
    path.write_bytes(b"truncated cache")
    recovered, info = extract_features(*args)
    assert not info["cache_hit"]
    np.testing.assert_array_equal(first, recovered)
    Image.new("RGB", (48, 64), "red").save(dataset.samples[0].path)
    changed = audit_dataset(small_dataset, expected_per_class=2)
    assert cache_key(changed.samples, "handcrafted", None) != first_info["cache_key"]
    assert cache_key(dataset.samples, "resnet18", "weight-a") != cache_key(dataset.samples, "resnet18", "weight-b")


def test_changing_evaluation_labels_does_not_change_features_or_predictions(small_dataset, tmp_path):
    dataset = audit_dataset(small_dataset, expected_per_class=2)
    raw, info = extract_features(dataset.samples, "handcrafted", tmp_path / "cache", torch.device("cpu"), 4)
    features, _ = transform_features(raw, "standard_l2")
    predictions, _ = cluster_features(features, "kmeans", 42)
    mapping = dict(zip(CLASSES, CLASSES[1:] + CLASSES[:1], strict=True))
    label_path = dataset.labels_path
    labels = json.loads(label_path.read_text())
    label_path.write_text(json.dumps({key: mapping[value] for key, value in labels.items()}))
    relabeled = audit_dataset(small_dataset, expected_per_class=2)
    after, after_info = extract_features(relabeled.samples, "handcrafted", tmp_path / "cache", torch.device("cpu"), 4)
    assert relabeled.fingerprint == dataset.fingerprint
    assert after_info["cache_key"] == info["cache_key"] and after_info["cache_hit"]
    np.testing.assert_array_equal(raw, after)
    after_features, _ = transform_features(after, "standard_l2")
    after_predictions, _ = cluster_features(after_features, "kmeans", 42)
    np.testing.assert_array_equal(predictions, after_predictions)


def test_cluster_id_permutation_and_optimal_acc():
    features, y = make_blobs(n_samples=60, n_features=8, centers=6, cluster_std=0.1, random_state=5)
    truth = [CLASSES[label] for label in y]
    predicted = np.array([11, 4, 9, 0, 18, 2])[y]
    before, _ = evaluate(features, y, truth)
    after, details = evaluate(features, predicted, truth)
    for metric in ("ari", "nmi", "acc", "silhouette", "davies_bouldin"):
        assert after[metric] == pytest.approx(before[metric])
    assert after["acc"] == 1.0
    assert details["matched_labels"] == truth
    # A many-to-one majority vote would incorrectly give 1.0 for this partition.
    mixed = y.copy()
    mixed[y == 1] = 0
    mixed[np.flatnonzero(y == 0)[:5]] = 1
    result, _ = evaluate(features, mixed, truth)
    assert result["acc"] < 1.0


@pytest.mark.parametrize("algorithm", ["kmeans", "ward"])
def test_fixed_seed_reproducibility_and_six_clusters(algorithm):
    features, _ = make_blobs(n_samples=60, n_features=8, centers=6, random_state=7)
    features, _ = transform_features(features, "l2_pca95_l2")
    first, _ = cluster_features(features, algorithm, 42)
    second, _ = cluster_features(features, algorithm, 42)
    np.testing.assert_array_equal(first, second)
    assert len(set(first)) == 6


def test_missing_or_unofficial_weights_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        prepare_weights(tmp_path / "missing.pth", tmp_path / "checkpoints")
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"fake")
    with pytest.raises(ValueError, match="checksum"):
        prepare_weights(fake, tmp_path / "checkpoints")


def test_cuda_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_device("cuda")


def test_all_predictions_are_frozen_before_external_evaluation(small_dataset, tmp_path, monkeypatch):
    """Exercise orchestration on a small dataset; this does not test pretrained quality."""
    from task1_clustering import main as entrypoint

    output = tmp_path / "output"
    real_audit = audit_dataset
    real_evaluate = evaluate
    synthetic, _ = make_blobs(n_samples=12, n_features=16, centers=6, random_state=7)
    monkeypatch.setattr(entrypoint, "audit_dataset", lambda root: real_audit(root, expected_per_class=2))
    monkeypatch.setattr(entrypoint, "prepare_weights", lambda *args: (Path("unused"), "synthetic-test-only"))
    monkeypatch.setattr(entrypoint, "extract_features", lambda *args: (synthetic.copy(), {"test_only": True}))

    def checked_evaluation(*args):
        frozen = list((output / "all").glob("*/seed_*/predictions.csv"))
        assert len(frozen) == 16
        assert all("true_label" not in path.read_text() for path in frozen)
        return real_evaluate(*args)

    monkeypatch.setattr(entrypoint, "evaluate", checked_evaluation)
    # Real figure rendering is covered by the full-data command; avoid test image overhead.
    monkeypatch.setattr(entrypoint, "contingency_plot", lambda *args: None)
    monkeypatch.setattr(entrypoint, "projection_plot", lambda *args: None)
    monkeypatch.setattr(entrypoint, "comparison_plot", lambda *args: None)
    monkeypatch.setattr(entrypoint, "image_galleries", lambda *args: {"mismatch_count": 0})
    assert entrypoint.main(["--data-dir", str(small_dataset), "--output-dir", str(output),
                            "--device", "cpu", "--experiment", "all"]) == 0
    status = json.loads((output / "all/status.json").read_text())
    assert status == {"state": "complete", "run_count": 16}
    assert (output / "all/summary.csv").is_file()
    assert (output / "all/report.md").is_file()


@pytest.mark.parametrize("location", ["same", "child", "parent"])
def test_output_must_not_overlap_original_data(small_dataset, location):
    from task1_clustering.main import main
    output = {"same": small_dataset, "child": small_dataset / "output", "parent": small_dataset.parent}[location]
    before = sorted(path.relative_to(small_dataset).as_posix() for path in small_dataset.rglob("*"))
    assert main(["--data-dir", str(small_dataset), "--output-dir", str(output)]) == 1
    after = sorted(path.relative_to(small_dataset).as_posix() for path in small_dataset.rglob("*"))
    assert before == after
