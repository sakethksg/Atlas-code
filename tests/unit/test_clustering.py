"""Unit tests for the spherical K-Means failure clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from afdad.failures.clustering import FailureClustering
from afdad.utils.models import FailureCluster


@pytest.fixture
def temp_centroids_path(tmp_path: Path) -> Path:
    return tmp_path / "centroids.npy"


@pytest.fixture
def clustering_cfg(temp_centroids_path: Path) -> DictConfig:
    return OmegaConf.create({
        "n_clusters": 3,
        "cluster_names": ["Syntax", "Runtime", "Logic"],
        "centroids_path": str(temp_centroids_path),
        "min_samples_for_fit": 4,
        "ema_decay": 0.1,
        "seed": 42,
    })


def test_clustering_heuristic_init(clustering_cfg: DictConfig) -> None:
    # Test heuristic initialization with too few samples
    clustering = FailureClustering(clustering_cfg)
    
    # Mock embeddings of size (3, 1024)
    embeddings = np.random.randn(3, 1024).astype(np.float32)
    clustering.fit(embeddings)
    
    # Centroids should be saved and have shape (3, 1024)
    assert clustering._centroids is not None
    assert clustering._centroids.shape == (3, 1024)
    assert Path(clustering_cfg.centroids_path).exists()


def test_clustering_predict_and_partial_fit(clustering_cfg: DictConfig) -> None:
    clustering = FailureClustering(clustering_cfg)
    
    # Fit initial mock centroids (3 clusters, 10 dimensions for simplicity)
    clustering_cfg.embedding_dim = 10
    embeddings = np.random.randn(5, 10).astype(np.float32)
    clustering.fit(embeddings)
    
    # Predict a sample
    sample = np.random.randn(10).astype(np.float32)
    cluster = clustering.predict(sample)
    assert cluster in ["Syntax", "Runtime", "Logic"]
    
    # Incrementally update with partial_fit
    old_centroids = clustering._centroids.copy()
    new_sample = np.random.randn(1, 10).astype(np.float32)
    clustering.partial_fit(new_sample)
    
    # Centroids should be updated
    assert not np.array_equal(old_centroids, clustering._centroids)
    
    # Check stats
    stats = clustering.get_cluster_stats()
    assert len(stats) == 3
    rates = clustering.get_failure_rates()
    assert sum(rates.values()) == pytest.approx(1.0)
