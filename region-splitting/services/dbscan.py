"""DBSCAN density-based clustering for the region-splitting module.

Mirrors the service interface used by :class:`LeidenService` and
:class:`LouvianService` (``build_graph`` / ``get_clusters`` /
``generate_visualization``) while exposing a standalone, testable toolkit
(``run_dbscan``, metrics, sensitivity sweeps, grid search) underneath.

Unlike the graph-based community detection used by Leiden/Louvain, DBSCAN
clusters spatial point clouds (here: road-network node coordinates) by
density. Noise points (label ``-1``) are reported separately and are
explicitly excluded from all internal metric calculations.
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from .base import BaseClusteringService
from .traci import TraciService


@dataclass
class DBSCANConfig:
    """Hyperparameters for DBSCAN clustering.

    Attributes:
        eps: Neighborhood radius. Maximum distance between two samples for
            one to be considered in the neighborhood of the other. CRITICAL
            parameter — must be tuned per dataset.
        min_samples: Minimum number of samples in a neighborhood for a core
            point. Higher values produce denser clusters and more noise.
        metric: Distance metric. Supports "euclidean", "cosine",
            "manhattan" and "precomputed" (distance matrix passed directly).
        metric_params: Additional keyword arguments for the metric function.
        algorithm: Nearest-neighbor algorithm: "auto", "ball_tree",
            "kd_tree" or "brute". Use "brute" for high-dimensional or
            precomputed metrics.
        leaf_size: Leaf size for BallTree/KDTree. Affects speed and memory.
        n_jobs: Parallelism for neighbor search. -1 uses all cores.
        use_embedding: If True, run DBSCAN on a low-dimensional embedding
            (e.g. UMAP/PCA coordinates) rather than the raw feature matrix.
        embedding_key: Key in ``data.obsm`` (or equivalent) for the
            embedding to use.
        label_noise_as: Value assigned to noise points. Conventionally -1.
            Some downstream tools expect 0; set accordingly.
    """

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    metric_params: dict | None = None
    algorithm: str = "auto"
    leaf_size: int = 30
    n_jobs: int = -1
    use_embedding: bool = True
    embedding_key: str = "X_umap"
    label_noise_as: int = -1

    def with_updates(self, **kwargs: Any) -> "DBSCANConfig":
        """Return a copy of this config with the given fields overridden."""
        return DBSCANConfig(**{**self.__dict__, **kwargs})


@dataclass
class DBSCANResult:
    """Result of a single DBSCAN run.

    Attributes:
        labels: Per-sample cluster assignment (int, noise = -1).
        n_clusters: Number of clusters found (excluding noise).
        n_noise: Number of noise points.
        noise_fraction: Fraction of points labelled as noise.
        core_sample_indices: Indices of core points.
        config: The DBSCANConfig used for the run.
        runtime_seconds: Wall-clock time of the run.
    """

    labels: np.ndarray
    n_clusters: int
    n_noise: int
    noise_fraction: float
    core_sample_indices: np.ndarray
    config: DBSCANConfig
    runtime_seconds: float


@dataclass
class SuggestedEps:
    """Suggested ``eps`` value derived from the k-distance plot.

    Attributes:
        eps: Suggested neighborhood radius.
        confidence: "confident" or "ambiguous" — reflects whether the elbow
            in the k-distance plot is well-defined.
    """

    eps: float
    confidence: str


@dataclass
class GridSearchResult:
    """Outcome of a DBSCAN parameter grid search.

    Attributes:
        best_eps: ``eps`` value of the best-scoring run.
        best_min_samples: ``min_samples`` value of the best-scoring run.
        best_score: Score of the best run under the requested metric.
        metric: Name of the scoring metric used.
        results: DataFrame with one row per (eps, min_samples) combination.
    """

    best_eps: float
    best_min_samples: int
    best_score: float
    metric: str
    results: pd.DataFrame


@dataclass
class DBSCANEvaluationReport:
    """Full evaluation report for a DBSCAN result.

    Attributes:
        labels: Per-sample cluster assignment (noise = -1).
        n_clusters: Number of clusters found (excluding noise).
        n_noise: Number of noise points.
        noise_fraction: Fraction of points labelled as noise.
        core_sample_indices: Indices of core points.
        config: The DBSCANConfig used for the run.
        runtime_seconds: Wall-clock time of the run.
        internal_metrics: Dict of metrics computed without ground truth.
        external_metrics: Dict of metrics computed against ground truth, or
            None when no ground truth was provided.
    """

    labels: np.ndarray
    n_clusters: int
    n_noise: int
    noise_fraction: float
    core_sample_indices: np.ndarray
    config: DBSCANConfig
    runtime_seconds: float
    internal_metrics: dict[str, Any]
    external_metrics: dict[str, Any] | None = None

    def summary(self) -> None:
        """Print a human-readable table of evaluation results."""
        line = "  " + "-" * 56
        print(line)
        print("  DBSCAN evaluation summary")
        print(line)
        print(f"  n_clusters      : {self.n_clusters}")
        print(f"  n_noise         : {self.n_noise}")
        print(f"  noise_fraction  : {self.noise_fraction:.4f}")
        print(f"  runtime_seconds : {self.runtime_seconds:.4f}")

        internal = self.internal_metrics or {}
        if internal:
            print(line)
            print("  Internal metrics (non-noise points)")
            print(line)
            for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
                value = internal.get(key)
                print(f"  {key:<20}: {'n/a' if value is None else f'{value:.4f}'}")
            sizes = internal.get("cluster_size_statistics")
            if sizes and sizes.get("n_clusters"):
                print(
                    "  cluster sizes     : "
                    f"min={sizes.get('min')}, max={sizes.get('max')}, "
                    f"mean={sizes.get('mean'):.2f}, "
                    f"median={sizes.get('median')}, gini={sizes.get('gini'):.3f}"
                )

        external = self.external_metrics
        if external:
            print(line)
            print("  External metrics (noise excluded)")
            print(line)
            for key, value in external.items():
                if isinstance(value, float):
                    print(f"  {key:<20}: {value:.4f}")
                else:
                    print(f"  {key:<20}: {value}")
        print(line)


def _extract_features(data: Any, config: DBSCANConfig) -> np.ndarray:
    """Extract the feature matrix from ``data`` using the accessor pattern.

    Supports (in order of preference): an ``obsm`` embedding keyed by
    ``config.embedding_key``, a raw ``X`` attribute, or a plain
    2-D array-like.
    """
    if hasattr(data, "obsm") and config.use_embedding and config.embedding_key in data.obsm:
        return np.asarray(data.obsm[config.embedding_key], dtype=float)
    if hasattr(data, "X"):
        return np.asarray(data.X, dtype=float)
    return np.asarray(data, dtype=float)


def _validate_distance_matrix(matrix: np.ndarray) -> None:
    """Validate that ``matrix`` is a square distance matrix, not a similarity."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"metric='precomputed' requires a square distance matrix, got shape {matrix.shape}")

    if matrix.size and not np.allclose(matrix, matrix.T, atol=1e-8):
        warnings.warn(
            "Distance matrix is not symmetric — check that it is a true distance matrix.",
            stacklevel=2,
        )

    diagonal = np.diag(matrix)
    if matrix.size and np.any(np.abs(diagonal) > 1e-8):
        warnings.warn(
            "Distance matrix has non-zero diagonal — distances should be zero on the diagonal.",
            stacklevel=2,
        )

    if matrix.size and np.nanmax(matrix) > 1.0:
        warnings.warn(
            "Input contains values > 1.0 — this looks like a similarity matrix. "
            "DBSCAN expects distances (larger = further apart).",
            stacklevel=2,
        )


def run_dbscan(data: Any, config: DBSCANConfig) -> DBSCANResult:
    """Run DBSCAN on ``data`` and return a :class:`DBSCANResult`.

    Args:
        data: Feature matrix (n_samples, n_features) or an object exposing
            ``obsm``/``X`` attributes (AnnData-like). When
            ``config.metric == "precomputed"``, ``data`` must be a square
            distance matrix.
        config: DBSCAN hyperparameters.

    Returns:
        DBSCANResult with cluster labels and run metadata.
    """
    start = time.perf_counter()

    features = _extract_features(data, config)
    if config.metric == "precomputed":
        _validate_distance_matrix(features)

    dbscan = DBSCAN(
        eps=config.eps,
        min_samples=config.min_samples,
        metric=config.metric,
        metric_params=config.metric_params,
        algorithm=config.algorithm,
        leaf_size=config.leaf_size,
        n_jobs=config.n_jobs,
    )
    labels = dbscan.fit_predict(features)
    runtime_seconds = time.perf_counter() - start

    noise_mask = labels == config.label_noise_as
    n_noise = int(np.sum(noise_mask))
    n_clusters = len(set(labels[~noise_mask])) if labels.size else 0
    noise_fraction = n_noise / labels.size if labels.size else 0.0

    return DBSCANResult(
        labels=np.asarray(labels),
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_fraction=noise_fraction,
        core_sample_indices=np.asarray(dbscan.core_sample_indices_, dtype=int),
        config=config,
        runtime_seconds=runtime_seconds,
    )


def _gini(sizes: Sequence[int]) -> float | None:
    """Gini coefficient of a cluster-size distribution (0 = perfectly even)."""
    values = np.sort(np.asarray(sizes, dtype=float))
    if values.size == 0:
        return None
    if values.size == 1:
        return 0.0
    cumsum = np.cumsum(values)
    return float((values.size + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / values.size)


def _cluster_pairwise_distance_mean(
    features: np.ndarray, labels: np.ndarray, cluster_id: int, metric: str
) -> float | None:
    """Mean pairwise distance within a single cluster (non-noise points)."""
    mask = labels == cluster_id
    members = features[mask]
    if members.shape[0] < 2:
        return None
    if metric == "precomputed":
        chunk = members[:, mask]
        upper = chunk[np.triu_indices(members.shape[0], k=1)]
    else:
        dist = pairwise_distances(members, metric=metric)
        upper = dist[np.triu_indices(members.shape[0], k=1)]
    return float(np.mean(upper))


def compute_internal_metrics(features: np.ndarray, result: DBSCANResult, config: DBSCANConfig) -> dict[str, Any]:
    """Compute clustering-quality metrics that need no ground truth.

    All internal metrics are computed on non-noise points only (labels !=
    ``config.label_noise_as``). Metric scores that are undefined for the
    given clustering (e.g. silhouette with fewer than 2 clusters) are
    returned as None and reported via a warning.

    Returns:
        Dict with keys: silhouette, davies_bouldin, calinski_harabasz,
        noise_fraction, cluster_size_statistics, density_metrics.
    """
    labels = result.labels
    non_noise = labels != config.label_noise_as

    if labels.size == 0:
        n_samples = 0
    else:
        n_samples = int(np.sum(non_noise))

    n_clusters = result.n_clusters
    silhouette = None
    davies_bouldin = None
    calinski_harabasz = None

    if n_clusters < 2:
        warnings.warn(
            f"n_clusters={n_clusters} < 2 — silhouette, davies_bouldin and "
            "calinski_harabasz are undefined and set to None.",
            stacklevel=2,
        )
    elif n_samples < 2:
        warnings.warn("Fewer than 2 non-noise points — internal metrics are undefined.", stacklevel=2)
    else:
        sub_features = features[non_noise]
        sub_labels = labels[non_noise]

        if n_clusters < n_samples:
            silhouette_kwargs = {"metric": config.metric if config.metric != "precomputed" else "precomputed"}
            if config.metric_params:
                silhouette_kwargs["metric_params"] = config.metric_params
            silhouette = float(silhouette_score(sub_features, sub_labels, **silhouette_kwargs))
        davies_bouldin = float(davies_bouldin_score(sub_features, sub_labels))
        if n_clusters < n_samples:
            calinski_harabasz = float(calinski_harabasz_score(sub_features, sub_labels))

    cluster_ids = sorted({int(label) for label in labels[non_noise]})
    sizes = [int(np.sum(labels == cid)) for cid in cluster_ids]

    size_stats = {
        "distribution": {cid: size for cid, size in zip(cluster_ids, sizes)},
        "min": min(sizes) if sizes else None,
        "max": max(sizes) if sizes else None,
        "mean": float(np.mean(sizes)) if sizes else None,
        "median": float(np.median(sizes)) if sizes else None,
        "std": float(np.std(sizes)) if sizes else None,
        "gini": _gini(sizes),
        "n_clusters": len(sizes),
    }

    intra_cluster = {}
    for cid in cluster_ids:
        mean_dist = _cluster_pairwise_distance_mean(features, labels, cid, config.metric)
        if mean_dist is not None:
            intra_cluster[cid] = mean_dist
    overall_intra = float(np.mean(list(intra_cluster.values()))) if intra_cluster else None

    nn_metric = config.metric if config.metric != "precomputed" else "precomputed"
    mean_nn = None
    if features.shape[0] >= 2:
        nn = NearestNeighbors(n_neighbors=2, metric=nn_metric, metric_params=config.metric_params)
        nn.fit(features)
        distances, _ = nn.kneighbors(features)
        mean_nn = float(np.mean(distances[:, 1]))

    n_core = int(result.core_sample_indices.size)
    core_ratio = n_core / n_samples if n_samples > 0 else None

    density_metrics = {
        "mean_intra_cluster_distance": overall_intra,
        "per_cluster_mean_intra_distance": intra_cluster,
        "mean_nearest_neighbor_distance": mean_nn,
        "core_point_ratio": core_ratio,
        "n_core_points": n_core,
    }

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "noise_fraction": result.noise_fraction,
        "cluster_size_statistics": size_stats,
        "density_metrics": density_metrics,
    }


def compute_external_metrics(labels: np.ndarray, ground_truth: np.ndarray, config: DBSCANConfig) -> dict[str, Any]:
    """Compute metrics that require ground-truth labels.

    Noise points are excluded from all computations — they do not represent
    a true class and would otherwise skew the scores.

    Returns:
        Dict with keys: ari, nmi, ami, fowlkes_mallows, homogeneity,
        completeness, v_measure, purity.
    """
    ground_truth = np.asarray(ground_truth)
    if ground_truth.shape != labels.shape:
        raise ValueError(f"ground_truth shape {ground_truth.shape} does not match labels shape {labels.shape}")

    mask = labels != config.label_noise_as
    pred = labels[mask]
    truth = ground_truth[mask]

    if pred.size == 0:
        warnings.warn("No non-noise points — external metrics are undefined.", stacklevel=2)
        return {
            "ari": None,
            "nmi": None,
            "ami": None,
            "fowlkes_mallows": None,
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
            "purity": None,
        }

    ari = float(adjusted_rand_score(truth, pred))
    nmi = float(normalized_mutual_info_score(truth, pred, average_method="arithmetic"))
    ami = float(adjusted_mutual_info_score(truth, pred))
    fmi = float(fowlkes_mallows_score(truth, pred))
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth, pred)

    purity = 0.0
    for cid in set(pred):
        cluster_truth = truth[pred == cid]
        counts = np.bincount(cluster_truth)
        purity += float(counts.max())
    purity /= pred.size

    return {
        "ari": ari,
        "nmi": nmi,
        "ami": ami,
        "fowlkes_mallows": fmi,
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "purity": purity,
    }


def evaluate_dbscan(
    data: Any,
    result: DBSCANResult,
    config: DBSCANConfig,
    ground_truth: np.ndarray | None = None,
) -> DBSCANEvaluationReport:
    """Compute all applicable metrics for a DBSCAN result.

    Args:
        data: Feature matrix used for the run (see :func:`run_dbscan`).
        result: Result of :func:`run_dbscan`.
        config: DBSCAN hyperparameters used for the run.
        ground_truth: Optional ground-truth labels. When provided, external
            metrics are computed with noise points excluded.

    Returns:
        A DBSCANEvaluationReport with internal and (optionally) external
        metrics.
    """
    features = _extract_features(data, config)
    internal = compute_internal_metrics(features, result, config)
    external = compute_external_metrics(result.labels, ground_truth, config) if ground_truth is not None else None

    return DBSCANEvaluationReport(
        labels=result.labels,
        n_clusters=result.n_clusters,
        n_noise=result.n_noise,
        noise_fraction=result.noise_fraction,
        core_sample_indices=result.core_sample_indices,
        config=config,
        runtime_seconds=result.runtime_seconds,
        internal_metrics=internal,
        external_metrics=external,
    )


def _sweep_row(data: Any, config: DBSCANConfig, eps: float, min_samples: int) -> dict[str, Any]:
    """Run DBSCAN for a single parameter combo and collect scoring columns."""
    run_config = config.with_updates(eps=eps, min_samples=min_samples)
    result = run_dbscan(data, run_config)
    report = evaluate_dbscan(data, result, run_config)

    row = {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise,
        "noise_fraction": result.noise_fraction,
    }
    for column in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        row[column] = report.internal_metrics.get(column)
    return row


def eps_sensitivity_analysis(
    data: Any, config: DBSCANConfig, eps_range: Sequence[float], fixed_min_samples: int | None = None
) -> pd.DataFrame:
    """Sweep ``eps`` over ``eps_range`` and record clustering quality.

    Args:
        data: Feature matrix (see :func:`run_dbscan`).
        config: Base DBSCAN configuration to sweep.
        eps_range: Values of ``eps`` to evaluate (e.g. ``np.linspace(0.1, 2.0, 30)``).
        fixed_min_samples: ``min_samples`` to use for every run. Defaults to
            ``config.min_samples``.

    Returns:
        DataFrame with columns: eps, min_samples, n_clusters, n_noise,
        noise_fraction, silhouette, davies_bouldin, calinski_harabasz.
    """
    min_samples = fixed_min_samples if fixed_min_samples is not None else config.min_samples
    rows = [_sweep_row(data, config, float(eps), min_samples) for eps in eps_range]
    return pd.DataFrame(rows)


def min_samples_sensitivity_analysis(
    data: Any, config: DBSCANConfig, min_samples_range: Sequence[int], fixed_eps: float | None = None
) -> pd.DataFrame:
    """Sweep ``min_samples`` over ``min_samples_range``.

    Args:
        data: Feature matrix (see :func:`run_dbscan`).
        config: Base DBSCAN configuration to sweep.
        min_samples_range: Values of ``min_samples`` to evaluate (e.g. ``range(2, 20)``).
        fixed_eps: ``eps`` to use for every run. Defaults to ``config.eps``.

    Returns:
        DataFrame with the same columns as :func:`eps_sensitivity_analysis`.
    """
    eps = fixed_eps if fixed_eps is not None else config.eps
    rows = [_sweep_row(data, config, eps, int(ms)) for ms in min_samples_range]
    return pd.DataFrame(rows)


def k_distance_plot_data(data: Any, config: DBSCANConfig, k: int | None = None) -> np.ndarray:
    """Compute sorted k-th nearest-neighbor distances for eps selection.

    Args:
        data: Feature matrix (see :func:`run_dbscan`).
        config: DBSCAN configuration whose metric determines the distance.
        k: Rank of the neighbor (0-indexed). Defaults to
            ``config.min_samples - 1``.

    Returns:
        Sorted (ascending) array of k-th nearest-neighbor distances. The
        "elbow" of this curve is a good choice for ``eps``. Callers are
        responsible for plotting.
    """
    features = _extract_features(data, config)
    k = config.min_samples - 1 if k is None else int(k)

    if k < 1:
        raise ValueError("k must be >= 1 (0 = the point itself).")
    if features.shape[0] < k + 1:
        raise ValueError(f"n_samples={features.shape[0]} is too small for k={k} nearest neighbors.")

    metric = config.metric if config.metric != "precomputed" else "precomputed"
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, metric_params=config.metric_params)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    kth_distances = np.sort(distances[:, k])
    return kth_distances


def _max_curvature_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Fallback elbow detection: point farthest from the chord between endpoints."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 0
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    chord = p2 - p1
    chord_norm = np.linalg.norm(chord)
    if chord_norm == 0:
        return y.size // 2
    distances = []
    for point in np.column_stack([x, y]):
        dist = np.abs(np.cross(chord, p1 - point)) / chord_norm
        distances.append(dist)
    return int(np.argmax(distances))


def suggest_eps(data: Any, config: DBSCANConfig) -> SuggestedEps:
    """Suggest an ``eps`` value from the k-distance plot elbow.

    Uses the kneedle algorithm when ``kneed`` is available, falling back to
    a max-curvature (farthest-from-chord) heuristic otherwise.

    Args:
        data: Feature matrix (see :func:`run_dbscan`).
        config: Base DBSCAN configuration (determines ``min_samples`` and
            the metric used for the k-distance plot).

    Returns:
        SuggestedEps with the suggested radius and a confidence note.
    """
    distances = k_distance_plot_data(data, config)
    x = np.arange(distances.size, dtype=float)
    y = distances

    if y.size < 3:
        return SuggestedEps(eps=float(y[-1]) if y.size else config.eps, confidence="ambiguous")

    elbow_index = None
    try:
        from kneed import KneeLocator

        kneedle = KneeLocator(x, y, curve="convex", direction="increasing", interp_method="interp1d")
        if kneedle.knee is not None:
            elbow_index = int(kneedle.knee)
    except Exception:
        elbow_index = None

    if elbow_index is None:
        elbow_index = _max_curvature_elbow(x, y)

    suggested = float(y[elbow_index])

    dynamic_range = float(y[-1] - y[0]) if y.size > 1 else 0.0
    mean = float(np.mean(y))
    flatness = dynamic_range / mean if mean > 0 else 0.0
    confidence = "confident" if flatness > 0.15 else "ambiguous"
    if confidence == "ambiguous":
        warnings.warn(
            "The k-distance curve is fairly flat — the elbow is ambiguous. "
            "Consider inspecting k_distance_plot_data() directly.",
            stacklevel=2,
        )

    return SuggestedEps(eps=suggested, confidence=confidence)


def cluster_stability_score(
    data: Any,
    config: DBSCANConfig,
    n_bootstrap: int = 20,
    sample_fraction: float = 0.8,
) -> dict[str, Any]:
    """Estimate clustering stability by bootstrapping.

    Re-runs DBSCAN on ``n_bootstrap`` random subsamples of the data and
    measures agreement (adjusted Rand index) between each subsample result
    and the full-data result.

    Args:
        data: Feature matrix (see :func:`run_dbscan`).
        config: DBSCAN configuration used for every run.
        n_bootstrap: Number of bootstrap replicates.
        sample_fraction: Fraction of points retained per subsample.

    Returns:
        Dict with keys: mean_ari, std_ari, ari_scores, n_bootstrap.
    """
    features = _extract_features(data, config)
    n_total = features.shape[0]
    if n_total == 0:
        raise ValueError("Cannot compute stability on an empty dataset.")

    full_result = run_dbscan(data, config)
    full_labels = full_result.labels
    rng = np.random.default_rng(42)
    subsample_size = max(1, int(n_total * sample_fraction))

    ari_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_total, size=subsample_size, replace=False)
        subsample = features[indices]
        bootstrap_result = run_dbscan(subsample, config)
        ari = adjusted_rand_score(full_labels[indices], bootstrap_result.labels)
        ari_scores.append(ari)

    ari_scores = np.asarray(ari_scores)
    return {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "ari_scores": ari_scores,
        "n_bootstrap": n_bootstrap,
    }


def min_samples_for(n_nodes: int) -> int:
    """Pick a DBSCAN ``min_samples`` scaled to the number of points."""
    return max(3, min(8, n_nodes // 10))


def choose_eps(coords: np.ndarray, min_samples: int) -> list[float]:
    """Log-spaced ``eps`` candidates derived from the nearest-neighbor scale."""
    nn = NearestNeighbors(n_neighbors=min(min_samples, max(2, len(coords) - 1)))
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    k = min(min_samples - 1, max(1, len(coords) - 1))
    base = float(np.median(distances[:, k])) if len(coords) > min_samples else 1.0
    return np.geomspace(base * 0.1, base * 10.0, num=24).tolist()


def cluster_bounds(n_nodes: int, mean_min: int = 8, mean_max: int = 40) -> tuple[int, int]:
    """Target a cluster count such that mean cluster size is in ``[mean_min, mean_max]``."""
    low = max(2, n_nodes // mean_max)
    high = max(low + 1, n_nodes // mean_min)
    return low, high


def auto_tune_config(
    coords: np.ndarray,
    max_noise_fraction: float = 0.20,
    min_samples: int | None = None,
    mean_min: int = 8,
    mean_max: int = 40,
) -> DBSCANConfig:
    """Auto-tune ``eps``/``min_samples`` for a point cloud.

    Sweeps log-spaced ``eps`` candidates (see :func:`choose_eps`) and picks
    the run whose cluster count falls inside :func:`cluster_bounds` with the
    lowest noise fraction. When no candidate lands in range, the candidate
    with the most clusters is used. Noise is not reassigned here — callers
    that need full coverage should pass ``reassign_noise=True`` to
    ``DBSCANService.get_clusters``.

    Args:
        coords: (n_samples, n_features) point cloud.
        max_noise_fraction: Reject candidates with more noise than this.
        min_samples: Fixed ``min_samples``; auto-scaled when None.
        mean_min: Lower bound of the target mean cluster size (fine cells use
            a smaller value, e.g. 4).
        mean_max: Upper bound of the target mean cluster size.

    Returns:
        A DBSCANConfig with tuned ``eps``/``min_samples``.
    """
    n_nodes = len(coords)
    min_samples = min_samples or min_samples_for(n_nodes)
    low, high = cluster_bounds(n_nodes, mean_min=mean_min, mean_max=mean_max)
    candidates = choose_eps(coords, min_samples)

    in_range: list[tuple[float, int, float]] = []
    out_of_range: list[tuple[float, int, float]] = []
    for eps in candidates:
        trial = run_dbscan(coords, DBSCANConfig(eps=eps, min_samples=min_samples))
        n = trial.n_clusters
        frac = trial.noise_fraction
        if frac > max_noise_fraction or n < 1:
            continue
        entry = (eps, n, frac)
        if low <= n <= high:
            in_range.append(entry)
        else:
            out_of_range.append(entry)

    if in_range:
        eps, _, _ = min(in_range, key=lambda e: e[2])  # lowest noise in range
    elif out_of_range:
        eps, _, _ = max(out_of_range, key=lambda e: e[1])  # most clusters
    else:
        eps = candidates[len(candidates) // 2]

    return DBSCANConfig(eps=eps, min_samples=min_samples)


_SCORING_DIRECTION = {
    "silhouette": "maximize",
    "calinski_harabasz": "maximize",
    "davies_bouldin": "minimize",
    "noise_fraction": "minimize",
}


def dbscan_grid_search(
    data: Any,
    eps_values: Sequence[float],
    min_samples_values: Sequence[int],
    metric: str = "silhouette",
    config_base: DBSCANConfig | None = None,
) -> GridSearchResult:
    """Grid-search over ``eps`` and ``min_samples`` combinations.

    Args:
        data: Feature matrix (see :func:`run_dbscan`).
        eps_values: Candidate ``eps`` values.
        min_samples_values: Candidate ``min_samples`` values.
        metric: Scoring metric. One of "silhouette", "calinski_harabasz",
            "davies_bouldin" or "noise_fraction".
        config_base: Base configuration; defaults to DBSCANConfig().

    Returns:
        GridSearchResult with the best parameter combination and a full
        results DataFrame (columns: eps, min_samples, n_clusters, n_noise,
        noise_fraction, silhouette, davies_bouldin, calinski_harabasz).
    """
    if metric not in _SCORING_DIRECTION:
        raise ValueError(f"Unknown scoring metric '{metric}'. Choose from {sorted(_SCORING_DIRECTION)}.")

    config = config_base or DBSCANConfig()
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            rows.append(_sweep_row(data, config, float(eps), int(min_samples)))

    results = pd.DataFrame(rows)

    scored = results[results[metric].notna()]
    if scored.empty:
        raise RuntimeError(f"No grid-search run produced a valid '{metric}' score — check the parameter range.")

    maximize = _SCORING_DIRECTION[metric] == "maximize"
    best_index = scored[metric].idxmax() if maximize else scored[metric].idxmin()
    best = results.loc[best_index]

    return GridSearchResult(
        best_eps=float(best["eps"]),
        best_min_samples=int(best["min_samples"]),
        best_score=float(best[metric]),
        metric=metric,
        results=results,
    )


class DBSCANService(BaseClusteringService):
    """DBSCAN clustering over road-network node coordinates.

    Mirrors the :class:`LeidenService` / :class:`LouvianService` interface:
    ``build_graph`` collects network nodes and their coordinates,
    ``get_clusters`` runs DBSCAN over the coordinates and groups nodes by
    cluster label, and ``generate_visualization`` renders the assignment.

    Noise points (label ``-1``) are stored under the ``"-1"`` cluster key
    in the output so nothing is silently dropped.
    """

    def __init__(
        self,
        net_config_path: str,
        traci_service: TraciService,
        config: DBSCANConfig | None = None,
    ):
        super().__init__(net_config_path, traci_service)
        self.config = config or DBSCANConfig()
        self.node_ids: list[str] = []
        self.coords: list[tuple[float, float]] = []
        self.graph = nx.Graph()

    def build_graph(self):
        print("  Constructing graph from network nodes (coordinates)...")
        for node in self.net.getNodes():
            nid = node.getID()
            x, y = node.getCoord()
            self.node_ids.append(nid)
            self.coords.append((float(x), float(y)))
            self.graph.add_node(nid, x=float(x), y=float(y))

        for edge in self.net.getEdges():
            if edge.isSpecial():
                continue
            self.graph.add_edge(
                edge.getFromNode().getID(),
                edge.getToNode().getID(),
                id=edge.getID(),
            )

        print(f"  Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        return self.graph

    def get_clusters(self, reassign_noise: bool = False):
        if not self.coords:
            raise RuntimeError("No coordinates collected — call build_graph() first.")

        coords = np.asarray(self.coords, dtype=float)
        result = run_dbscan(coords, self.config)
        report = evaluate_dbscan(coords, result, self.config)

        clusters = {}
        for node_id, label in zip(self.node_ids, result.labels):
            clusters.setdefault(str(int(label)), []).append(node_id)

        metrics = {
            "n_clusters": result.n_clusters,
            "n_noise": result.n_noise,
            "noise_fraction": result.noise_fraction,
            "eps": self.config.eps,
            "min_samples": self.config.min_samples,
        }
        metrics.update(report.internal_metrics)

        if reassign_noise:
            clusters, metrics = self._reassign_noise(coords, result, clusters, metrics)

        return {"clusters": clusters, "metrics": metrics}

    def _reassign_noise(self, coords, result, clusters, metrics):
        """Assign noise points to their nearest cluster centroid.

        Produces full network coverage so no intersection is dropped from the
        meta-policy (used for the production RL pipeline).
        """
        noise_key = str(self.config.label_noise_as)
        noise_nodes = clusters.pop(noise_key, [])

        cluster_centroids = {}
        for cid, nodes in clusters.items():
            indices = [self.node_ids.index(node) for node in nodes]
            cluster_centroids[cid] = np.mean(coords[indices], axis=0)

        if not cluster_centroids:
            return clusters, metrics

        for node in noise_nodes:
            index = self.node_ids.index(node)
            point = coords[index]
            nearest = min(
                cluster_centroids.items(),
                key=lambda item: float(np.linalg.norm(point - item[1])),
            )[0]
            clusters[nearest].append(node)

        metrics["n_noise"] = 0
        metrics["noise_fraction"] = 0.0
        return clusters, metrics

    def generate_visualization(self, clusters: dict, output_path: str):
        pos = {node: (data["x"], data["y"]) for node, data in self.graph.nodes(data=True)}

        partition_map = defaultdict(lambda: -1)
        for cid, nodes in clusters.items():
            for node in nodes:
                partition_map[node] = int(cid)

        colors = [partition_map[n] for n in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        nx.draw(
            self.graph,
            pos,
            node_size=8,
            node_color=colors,
            with_labels=False,
            cmap=plt.cm.tab20,
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
