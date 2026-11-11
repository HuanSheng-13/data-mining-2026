"""Evaluation only: permutation-invariant metrics and one-to-one cluster matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from threadpoolctl import threadpool_limits

from task1_clustering.data import CLASSES


def evaluate(features: np.ndarray, predictions: np.ndarray, true_labels: list[str]) -> tuple[dict, dict]:
    if len(true_labels) != len(predictions) or len(features) != len(predictions):
        raise ValueError("Features, predictions and evaluation labels must have equal length.")
    clusters = np.unique(predictions)
    table = np.zeros((len(CLASSES), len(clusters)), dtype=np.int64)
    for category, cluster in zip(true_labels, predictions, strict=True):
        table[CLASSES.index(category), np.searchsorted(clusters, cluster)] += 1
    rows, columns = linear_sum_assignment(table, maximize=True)
    mapping = {int(clusters[column]): CLASSES[row] for row, column in zip(rows, columns, strict=True)}
    matched_labels = [mapping.get(int(cluster), "unmatched") for cluster in predictions]
    with threadpool_limits(limits=1):
        metrics = {
            "ari": float(adjusted_rand_score(true_labels, predictions)),
            "nmi": float(normalized_mutual_info_score(true_labels, predictions, average_method="arithmetic")),
            "acc": float(table[rows, columns].sum() / len(predictions)),
            "silhouette": float(silhouette_score(features, predictions, metric="euclidean")),
            "davies_bouldin": float(davies_bouldin_score(features, predictions)),
            "cluster_sizes": {str(int(cluster)): int((predictions == cluster).sum()) for cluster in clusters},
        }
    return metrics, {"classes": list(CLASSES), "cluster_ids": [int(c) for c in clusters],
                     "contingency": table.tolist(), "cluster_to_class": mapping,
                     "matched_labels": matched_labels}
