"""Tests for DBSCAN clustering (``services/dbscan.py``).

DBSCAN is the only density-based algorithm in the region-splitting module, so
these tests exercise the standalone toolkit (``run_dbscan``, metrics,
diagnostics) rather than the SUMO-bound service classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from services.dbscan import (
    DBSCANConfig,
    DBSCANEvaluationReport,
    DBSCANResult,
    GridSearchResult,
    dbscan_grid_search,
    eps_sensitivity_analysis,
    evaluate_dbscan,
    min_samples_sensitivity_analysis,
    run_dbscan,
)
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, silhouette_score


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    return X, y


@pytest.fixture
def base_config():
    return DBSCANConfig(eps=0.5, min_samples=5)


def test_dbscan_basic_clustering(blobs, base_config):
    X, _ = blobs
    result = run_dbscan(X, base_config)

    assert result.n_clusters == 3
    assert result.noise_fraction < 0.05
    assert len(result.labels) == len(X)


def test_dbscan_detects_noise(blobs, base_config):
    X, _ = blobs
    rng = np.random.default_rng(0)
    noise_pts = rng.uniform(-15, 15, size=(20, X.shape[1]))
    X_with_noise = np.vstack([X, noise_pts])

    result = run_dbscan(X_with_noise, base_config)

    assert result.n_noise > 0
    assert result.labels[-20:].min() == -1


def test_dbscan_precomputed_metric(blobs, base_config):
    X, _ = blobs
    distance_matrix = pairwise_distances(X, metric="euclidean")

    precomputed = run_dbscan(distance_matrix, base_config.with_updates(metric="precomputed"))
    euclidean = run_dbscan(X, base_config.with_updates(metric="euclidean"))

    assert np.array_equal(precomputed.labels, euclidean.labels)
    assert precomputed.n_clusters == euclidean.n_clusters


def test_dbscan_all_noise(blobs):
    X, _ = blobs
    config = DBSCANConfig(eps=0.0001, min_samples=5)

    result = run_dbscan(X, config)

    assert result.n_clusters == 0
    assert np.all(result.labels == config.label_noise_as)
    assert result.noise_fraction == 1.0

    report = evaluate_dbscan(X, result, config)
    assert report.internal_metrics["silhouette"] is None
    assert report.internal_metrics["davies_bouldin"] is None
    assert report.internal_metrics["calinski_harabasz"] is None


def test_dbscan_single_cluster(blobs):
    X, _ = blobs
    config = DBSCANConfig(eps=999, min_samples=5)

    result = run_dbscan(X, config)

    assert result.n_clusters == 1
    assert result.n_noise == 0


def test_silhouette_excludes_noise(blobs, base_config):
    X, _ = blobs
    rng = np.random.default_rng(0)
    noise_pts = rng.uniform(-15, 15, size=(20, X.shape[1]))
    X_with_noise = np.vstack([X, noise_pts])

    result = run_dbscan(X_with_noise, base_config)
    report = evaluate_dbscan(X_with_noise, result, base_config)

    non_noise = result.labels != base_config.label_noise_as
    expected = silhouette_score(
        X_with_noise[non_noise],
        result.labels[non_noise],
        metric="euclidean",
    )

    assert report.internal_metrics["silhouette"] is not None
    assert np.isclose(report.internal_metrics["silhouette"], expected)


def test_eps_sensitivity_returns_dataframe(blobs, base_config):
    X, _ = blobs
    df = eps_sensitivity_analysis(X, base_config, np.linspace(0.1, 1.0, 4))

    expected_columns = {
        "eps",
        "min_samples",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 4

    ms_df = min_samples_sensitivity_analysis(X, base_config, range(2, 6))
    assert len(ms_df) == 4
    assert set(ms_df["min_samples"]) == {2, 3, 4, 5}


def test_grid_search_returns_best_params(blobs):
    X, _ = blobs
    gs = dbscan_grid_search(X, [0.2, 0.5, 0.8], [3, 5, 8], metric="silhouette")

    assert isinstance(gs, GridSearchResult)
    assert gs.best_eps in [0.2, 0.5, 0.8]
    assert gs.best_min_samples in [3, 5, 8]
    assert gs.best_score is not None
    expected_columns = {
        "eps",
        "min_samples",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
    }
    assert expected_columns.issubset(set(gs.results.columns))
    assert len(gs.results) == 9


def test_dbscan_result_matches_interface(blobs, base_config):
    X, _ = blobs
    result = run_dbscan(X, base_config)

    assert isinstance(result, DBSCANResult)
    for field in (
        "labels",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "core_sample_indices",
        "config",
        "runtime_seconds",
    ):
        assert hasattr(result, field), f"DBSCANResult missing field: {field}"
    assert result.config is base_config
    assert result.runtime_seconds >= 0
    assert set(result.labels).issubset({-1, 0, 1, 2})


def test_evaluation_report_matches_interface(blobs, base_config):
    X, _ = blobs
    result = run_dbscan(X, base_config)
    report = evaluate_dbscan(X, result, base_config)

    assert isinstance(report, DBSCANEvaluationReport)
    for field in (
        "labels",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "config",
        "runtime_seconds",
        "internal_metrics",
    ):
        assert hasattr(report, field), f"DBSCANEvaluationReport missing field: {field}"

    for key in (
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "noise_fraction",
        "cluster_size_statistics",
        "density_metrics",
    ):
        assert key in report.internal_metrics, f"internal metric missing: {key}"

    report.summary()  # must not raise
