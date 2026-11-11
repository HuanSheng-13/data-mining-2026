"""Headless report figures; projections and matched classes never feed clustering."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from task1_clustering.features import geometry


def contingency_plot(details: dict, output: Path) -> None:
    table = np.asarray(details["contingency"])
    figure, axis = plt.subplots(figsize=(7, 5))
    graphic = axis.imshow(table, cmap="Blues", vmin=0)
    axis.set_xticks(range(table.shape[1]), labels=details["cluster_ids"])
    axis.set_yticks(range(table.shape[0]), labels=details["classes"])
    axis.set(xlabel="Cluster ID (arbitrary)", ylabel="True category", title="Category / cluster contingency")
    for row, column in np.ndindex(table.shape):
        axis.text(column, row, str(table[row, column]), ha="center", va="center",
                  color="white" if table[row, column] > table.max() / 2 else "black")
    figure.colorbar(graphic, ax=axis, label="Image count")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def projection_plot(features, predictions, labels, sample_ids, output_dir: Path) -> None:
    with threadpool_limits(limits=1):
        coordinates = PCA(n_components=min(2, features.shape[1]), svd_solver="full").fit_transform(features)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates, np.zeros(len(features))))
    pd.DataFrame({"sample_id": sample_ids, "pc1": coordinates[:, 0], "pc2": coordinates[:, 1],
                  "true_label": labels, "cluster": predictions}).to_csv(output_dir / "projection.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, values, title in zip(axes, (np.asarray(labels), predictions),
                                   ("True categories", "Predicted clusters"), strict=True):
        for index, value in enumerate(sorted(set(values))):
            subset = values == value
            axis.scatter(coordinates[subset, 0], coordinates[subset, 1], s=16, alpha=0.75,
                         color=plt.get_cmap("tab10")(index), label=str(value))
        axis.set(title=title, xlabel="PC1", ylabel="PC2")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("PCA projection of clustering features (visualization only)")
    figure.tight_layout()
    figure.savefig(output_dir / "projection.png", dpi=160)
    plt.close(figure)


def _show(axis, sample, caption, crop=None):
    with Image.open(sample.path) as image:
        rgb = image.convert("RGB")
        if crop is not None:
            rgb = crop(rgb)
        rgb.thumbnail((256, 256))
        axis.imshow(np.asarray(rgb))
    axis.set_title(caption, fontsize=8)
    axis.axis("off")


def image_galleries(samples, features, predictions, true_labels, details, output_dir: Path) -> dict:
    figure, axes = plt.subplots(6, 5, figsize=(12, 13), squeeze=False)
    representatives = {}
    for row, cluster in enumerate(sorted(set(predictions))):
        indices = np.flatnonzero(predictions == cluster)
        center = features[indices].mean(axis=0)
        order = indices[np.argsort(np.linalg.norm(features[indices] - center, axis=1), kind="stable")]
        representatives[str(int(cluster))] = [samples[i].sample_id for i in order[:5]]
        for column, axis in enumerate(axes[row]):
            if column < len(order):
                i = order[column]
                _show(axis, samples[i], f"C{cluster}: {samples[i].sample_id}\ntrue: {true_labels[i]}")
            else:
                axis.axis("off")
    figure.suptitle("Five nearest images to each cluster mean (original images)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / "representatives.png", dpi=140)
    plt.close(figure)

    mistakes = [i for i, (truth, matched) in enumerate(zip(true_labels, details["matched_labels"], strict=True))
                if truth != matched]
    selected = mistakes[:12]  # Stable manifest order; not cherry-picked by visual appearance.
    if selected:
        figure, axes = plt.subplots(len(selected), 2, figsize=(7, 2.4 * len(selected)), squeeze=False)
        crop = geometry()
        for row, i in enumerate(selected):
            caption = f"{samples[i].sample_id}: {true_labels[i]} -> {details['matched_labels'][i]} (C{predictions[i]})"
            _show(axes[row, 0], samples[i], caption + "\nOriginal")
            _show(axes[row, 1], samples[i], "Actual 224 x 224 center crop", crop)
        figure.tight_layout()
        figure.savefig(output_dir / "confusions.png", dpi=120)
        plt.close(figure)
    else:
        figure, axis = plt.subplots(figsize=(7, 2))
        axis.text(0.5, 0.5, "No mismatches under the evaluation-only one-to-one mapping.", ha="center", va="center")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / "confusions.png", dpi=120)
        plt.close(figure)
    return {"representative_ids": representatives,
            "mismatch_count": len(mistakes), "displayed_mismatch_ids": [samples[i].sample_id for i in selected]}


def comparison_plot(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(summary))
    width = 0.25
    for offset, metric in enumerate(("ari", "nmi", "acc")):
        axis.bar(positions + (offset - 1) * width, summary[f"{metric}_mean"], width,
                 yerr=summary[f"{metric}_std"], capsize=3, label=metric.upper())
    axis.set_xticks(positions, labels=[name.split("_")[0] for name in summary["experiment"]])
    axis.set(ylabel="Score", xlabel="Fixed experiment", title="External metrics (mean and sample standard deviation)")
    axis.set_ylim(min(-0.05, float(summary["ari_mean"].min()) - 0.05), 1.08)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
