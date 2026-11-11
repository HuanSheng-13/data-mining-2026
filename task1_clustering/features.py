"""Frozen visual features, deterministic geometry and content-addressed caching."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import local_binary_pattern
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode

from task1_clustering.data import ImageSample, file_sha256

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
PREPROCESS = {
    "version": 1, "rgb": True, "resize_short_edge": 256, "center_crop": 224,
    "interpolation": "bilinear", "antialias": True,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; install a compatible wheel or use --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                        "cpu" if requested == "auto" else requested)


def configure_reproducibility(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def geometry() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
    ])


def prepare_weights(weights_path: Path | None, checkpoint_dir: Path) -> tuple[Path, str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = weights_path.resolve() if weights_path else checkpoint_dir / WEIGHTS.url.rsplit("/", 1)[1]
    if not path.is_file():
        if weights_path is not None:
            raise ValueError(f"Weights file does not exist: {path}")
        try:
            torch.hub.load_state_dict_from_url(
                WEIGHTS.url, model_dir=str(checkpoint_dir), check_hash=True,
                map_location="cpu", weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot download official ResNet-18 weights: {exc}. Download {WEIGHTS.url} "
                "and pass --weights-path, or place the file in output-dir/checkpoints/."
            ) from exc
    digest = file_sha256(path)
    expected_prefix = WEIGHTS.url.rsplit("-", 1)[1].split(".")[0]
    if not digest.startswith(expected_prefix):
        raise ValueError(f"Weights checksum does not match official IMAGENET1K_V1: {path}")
    return path, digest


def cache_key(samples: tuple[ImageSample, ...], extractor: str, weight_digest: str | None,
              execution: dict | None = None) -> str:
    spec = {
        "samples": [(sample.sample_id, sample.sha256) for sample in samples],
        "extractor": extractor, "weights_sha256": weight_digest,
        "execution": execution,
        "preprocess": PREPROCESS, "hsv_bins": 16, "lbp": [8, 1, "uniform"],
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "pillow", "numpy", "scikit-image")},
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _deep_features(samples, weights_path, device, batch_size) -> np.ndarray:
    model = resnet18(weights=None)
    try:
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    except Exception as exc:
        raise ValueError(f"Cannot load ResNet-18 state dictionary: {exc}") from exc
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval().to(device)
    preprocess = WEIGHTS.transforms()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            images = []
            for sample in samples[start:start + batch_size]:
                with Image.open(sample.path) as image:
                    images.append(preprocess(image.convert("RGB")))
            batch = torch.stack(images).to(device)
            blocks.append(model(batch).cpu().numpy())
            print(f"ResNet-18: {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    del model
    return np.concatenate(blocks).astype(np.float32)


def _handcrafted_features(samples) -> np.ndarray:
    crop = geometry()
    rows = []
    for sample in samples:
        with Image.open(sample.path) as image:
            rgb = crop(image.convert("RGB"))
            hsv = rgb2hsv(np.asarray(rgb, dtype=np.float32) / 255.0)
            gray = np.asarray(rgb.convert("L"))
        histograms = []
        for channel in range(3):
            hist, _ = np.histogram(hsv[:, :, channel], bins=16, range=(0.0, 1.0))
            histograms.append(hist / hist.sum())
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(11))
        histograms.append(hist / hist.sum())
        rows.append(np.concatenate(histograms))
    return np.asarray(rows, dtype=np.float32)


def extract_features(samples, extractor, cache_dir, device, batch_size,
                     weights_path=None, weight_digest=None) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    execution = {"device": str(device), "batch_size": batch_size} if extractor == "resnet18" else None
    key = cache_key(samples, extractor, weight_digest, execution)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{extractor}-{key}.npz"
    expected_ids = np.asarray([sample.sample_id for sample in samples])
    cache_hit = False
    features = None
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as cached:
                if np.array_equal(cached["sample_ids"], expected_ids):
                    features = cached["features"]
                    cache_hit = True
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # A truncated/stale cache can be regenerated from the original images.
    dimension = 512 if extractor == "resnet18" else 58
    if features is not None and (features.shape != (len(samples), dimension) or
                                 not np.isfinite(features).all()):
        features, cache_hit = None, False
    if features is None:
        if extractor == "resnet18":
            features = _deep_features(samples, weights_path, device, batch_size)
        elif extractor == "handcrafted":
            features = _handcrafted_features(samples)
        else:
            raise ValueError(f"Unknown feature extractor: {extractor}")
        if features.shape != (len(samples), dimension) or not np.isfinite(features).all():
            raise ValueError("Feature extraction returned invalid shape or non-finite values.")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, sample_ids=expected_ids, features=features)
        temporary.replace(path)
    return features, {
        "cache_key": key, "cache_hit": cache_hit, "raw_dimension": dimension,
        "elapsed_seconds": time.perf_counter() - started,
        "weights": WEIGHTS.name if extractor == "resnet18" else None,
        "weights_sha256": weight_digest,
    }
