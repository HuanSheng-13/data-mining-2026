"""Read images and audit evaluation metadata without exposing labels to models."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

CLASSES = ("bottle", "cable", "leather", "pill", "tile", "transistor")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class DataError(ValueError):
    """An input dataset cannot be evaluated as the specified course dataset."""


@dataclass(frozen=True)
class ImageSample:
    sample_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class Dataset:
    root: Path
    samples: tuple[ImageSample, ...]
    labels_path: Path | None
    fingerprint: str
    audit: dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_images(path: Path) -> bool:
    return path.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in path.rglob("*")
    )


def _locate(data_dir: Path) -> tuple[Path, Path | None]:
    # The distributed archive has an extra Cluster/ wrapper and flat PNG files.
    candidates = (data_dir / "dataset", data_dir / "Cluster" / "dataset", data_dir)
    for root in candidates:
        if not root.is_dir():
            continue
        labels_paths = (root.parent / "cluster_labels.json", root / "cluster_labels.json")
        label_file = next((p for p in labels_paths if p.is_file()), None)
        folder_layout = all((root / category).is_dir() for category in CLASSES)
        if label_file is not None or folder_layout:
            return root, label_file
    if any(_has_images(root) for root in candidates):
        raise DataError(
            "Images found, but evaluation labels are missing. Expected cluster_labels.json "
            "beside dataset/, or six category folders: " + ", ".join(CLASSES)
        )
    raise DataError(
        f"No dataset found under {data_dir}. Extract Cluster.zip there; expected "
        "Cluster/dataset/*.png and Cluster/cluster_labels.json (dataset/ also supported)."
    )


def _unique_json_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"Duplicate image key in cluster_labels.json: {key}")
        result[key] = value
    return result


def read_labels(dataset: Dataset) -> list[str]:
    """Read labels in manifest order, for audit/evaluation only."""
    if dataset.labels_path is None:
        labels = [Path(sample.sample_id).parts[0] for sample in dataset.samples]
    else:
        try:
            mapping = json.loads(
                dataset.labels_path.read_text(encoding="utf-8-sig"),
                object_pairs_hook=_unique_json_object,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"Cannot read labels: {exc}") from exc
        if not isinstance(mapping, dict):
            raise DataError("cluster_labels.json must map image filenames to category strings.")
        ids = {sample.sample_id for sample in dataset.samples}
        if set(mapping) != ids:
            missing = sorted(ids - set(mapping))[:5]
            extra = sorted(set(mapping) - ids)[:5]
            raise DataError(f"Labels and images differ; missing labels={missing}, extra labels={extra}")
        labels = [mapping[sample.sample_id] for sample in dataset.samples]
    if any(not isinstance(label, str) or label not in CLASSES for label in labels):
        raise DataError(f"Labels must belong to the six categories: {CLASSES}")
    return labels


def audit_dataset(data_dir: Path, expected_per_class: int = 100) -> Dataset:
    """Check every image; strict counts prevent silently changing the experiment."""
    data_dir = data_dir.resolve()
    root, labels_path = _locate(data_dir)
    paths = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    expected_total = len(CLASSES) * expected_per_class
    if len(paths) != expected_total:
        raise DataError(f"Expected {expected_total} images, found {len(paths)} in {root}")
    samples, sizes, modes = [], Counter(), Counter()
    duplicates: dict[str, list[str]] = defaultdict(list)
    fingerprint = hashlib.sha256()
    for path in paths:
        sample_id = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise DataError(f"Cannot decode image {sample_id}: {exc}") from exc
        digest = file_sha256(path)
        samples.append(ImageSample(sample_id, path, digest))
        duplicates[digest].append(sample_id)
        fingerprint.update(f"{sample_id}\0{digest}\n".encode("utf-8"))
    dataset = Dataset(root, tuple(samples), labels_path, fingerprint.hexdigest(), {})
    counts = Counter(read_labels(dataset))
    if any(counts[category] != expected_per_class for category in CLASSES):
        raise DataError(f"Expected {expected_per_class} images per category; found {dict(counts)}")
    audit = {
        "image_count": len(samples), "class_counts": dict(sorted(counts.items())),
        "image_sizes": dict(sorted(sizes.items())), "image_modes": dict(sorted(modes.items())),
        "duplicate_content_groups": [ids for ids in duplicates.values() if len(ids) > 1],
        "dataset_fingerprint": dataset.fingerprint,
        "labels_sha256": file_sha256(labels_path) if labels_path else None,
        "layout": "flat_json" if labels_path else "category_folders",
        "root": str(root),
    }
    return Dataset(root, dataset.samples, labels_path, dataset.fingerprint, audit)
