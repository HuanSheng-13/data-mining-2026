"""Label-free transformations and clustering; no evaluation labels are accepted."""

from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

EXPERIMENTS = {
    "A_handcrafted_kmeans": ("handcrafted", "standard_l2", "kmeans"),
    "B_resnet18_kmeans": ("resnet18", "l2", "kmeans"),
    "C_resnet18_pca_kmeans": ("resnet18", "l2_pca95_l2", "kmeans"),
    "D_resnet18_pca_ward": ("resnet18", "l2_pca95_l2", "ward"),
}
MAIN_EXPERIMENT = "C_resnet18_pca_kmeans"
KMEANS_PARAMS = dict(n_clusters=6, init="k-means++", n_init=20,
                     max_iter=300, tol=1e-4, algorithm="lloyd")


def transform_features(features: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected a finite 2D feature matrix.")
    started = time.perf_counter()
    explained_variance = None
    with threadpool_limits(limits=1):
        if method == "standard_l2":
            transformed = normalize(StandardScaler().fit_transform(features))
        elif method == "l2":
            transformed = normalize(features)
        elif method == "l2_pca95_l2":
            pca = PCA(n_components=0.95, svd_solver="full", whiten=False)
            transformed = normalize(pca.fit_transform(normalize(features)))
            explained_variance = float(pca.explained_variance_ratio_.sum())
        else:
            raise ValueError(f"Unknown transformation: {method}")
    if not np.isfinite(transformed).all() or np.any(np.linalg.norm(transformed, axis=1) == 0):
        raise ValueError("Transformation produced zero/non-finite vectors; inspect image features.")
    return transformed, {
        "method": method, "dimension": transformed.shape[1],
        "explained_variance_ratio": explained_variance,
        "elapsed_seconds": time.perf_counter() - started,
    }


def cluster_features(features: np.ndarray, algorithm: str, seed: int) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        if algorithm == "kmeans":
            model = KMeans(**KMEANS_PARAMS, random_state=seed).fit(features)
            metadata = {"inertia": float(model.inertia_), "iterations": int(model.n_iter_),
                        "parameters": {**KMEANS_PARAMS, "random_state": seed}}
        elif algorithm == "ward":
            model = AgglomerativeClustering(n_clusters=6, linkage="ward", metric="euclidean").fit(features)
            metadata = {"inertia": None, "iterations": None,
                        "parameters": {"n_clusters": 6, "linkage": "ward", "metric": "euclidean"}}
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    labels = model.labels_.astype(np.int64)
    if np.unique(labels).size != 6:
        raise ValueError("Clustering did not produce six nonempty clusters; inspect duplicate/constant features.")
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["deterministic"] = algorithm == "ward"
    return labels, metadata
