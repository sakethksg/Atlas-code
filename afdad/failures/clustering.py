"""
Spherical K-Means failure clustering.

Clusters failure embeddings into predefined categories:
Syntax, Runtime, Logic, Edge Cases, Algorithm Design, Efficiency.

Persists centroids to disk for cross-session consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from afdad.utils.logging import get_logger
from afdad.utils.models import ClusterInfo, FailureCluster


class FailureClustering:
    """Spherical K-Means clustering for failure embeddings.

    Parameters
    ----------
    cfg:
        Clustering configuration with ``n_clusters``, ``cluster_names``,
        ``centroids_path``, and ``min_samples_for_fit``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.n_clusters: int = cfg.n_clusters
        self.cluster_names: list[str] = list(cfg.cluster_names)
        self.centroids_path = Path(cfg.centroids_path)
        self.min_samples: int = cfg.min_samples_for_fit
        self.logger = get_logger()

        self._kmeans: KMeans | None = None
        self._centroids: np.ndarray | None = None
        self._cluster_counts: dict[str, int] = {
            name: 0 for name in self.cluster_names
        }

        # Try loading persisted centroids
        self._load_centroids()

    # ── Public API ────────────────────────────────────────────

    def predict(self, embedding: np.ndarray) -> str:
        """Predict the cluster for a single failure embedding.

        Parameters
        ----------
        embedding:
            Normalised embedding vector.

        Returns
        -------
        str
            Cluster name.
        """
        if self._centroids is None:
            self.logger.warning(
                "No centroids fitted yet — returning UNKNOWN cluster"
            )
            return FailureCluster.UNKNOWN.value

        embedding_norm = normalize(embedding.reshape(1, -1))
        # Cosine similarity → pick nearest centroid
        similarities = embedding_norm @ self._centroids.T
        cluster_idx = int(np.argmax(similarities))

        name = self.cluster_names[cluster_idx]
        self._cluster_counts[name] += 1
        return name

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit Spherical K-Means on a batch of failure embeddings.

        Parameters
        ----------
        embeddings:
            2-D array of shape ``(n_samples, embedding_dim)``.
        """
        n_samples = embeddings.shape[0]

        if n_samples < self.min_samples:
            self.logger.info(
                f"Only {n_samples} samples — need {self.min_samples} to fit. "
                "Using heuristic initialisation instead."
            )
            self._heuristic_init(embeddings)
            return

        self.logger.info(
            f"Fitting Spherical K-Means with {n_samples} samples, "
            f"{self.n_clusters} clusters"
        )

        embeddings_norm = normalize(embeddings)

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        self._kmeans.fit(embeddings_norm)
        self._centroids = normalize(self._kmeans.cluster_centers_)
        self._save_centroids()

        self.logger.info("Clustering fitted and centroids saved.")

    def partial_fit(self, new_embeddings: np.ndarray) -> None:
        """Incrementally update centroids with new failure embeddings.

        Uses exponential moving average to update centroids without
        re-fitting from scratch.

        Parameters
        ----------
        new_embeddings:
            New embeddings to incorporate.
        """
        if self._centroids is None:
            self.fit(new_embeddings)
            return

        new_norm = normalize(new_embeddings)
        alpha = 0.1  # EMA decay factor

        for emb in new_norm:
            similarities = emb.reshape(1, -1) @ self._centroids.T
            nearest = int(np.argmax(similarities))
            self._centroids[nearest] = (
                (1 - alpha) * self._centroids[nearest] + alpha * emb
            )

        # Re-normalise centroids (spherical constraint)
        self._centroids = normalize(self._centroids)
        self._save_centroids()

    def get_cluster_stats(self) -> list[ClusterInfo]:
        """Return current cluster statistics."""
        total = sum(self._cluster_counts.values()) or 1
        return [
            ClusterInfo(
                cluster_name=name,
                total_failures=count,
                failure_rate=count / total,
            )
            for name, count in self._cluster_counts.items()
        ]

    def get_failure_rates(self) -> dict[str, float]:
        """Return failure rates per cluster (for adaptive curriculum)."""
        total = sum(self._cluster_counts.values()) or 1
        return {
            name: count / total
            for name, count in self._cluster_counts.items()
        }

    # ── Persistence ───────────────────────────────────────────

    def _save_centroids(self) -> None:
        """Save centroids to disk."""
        if self._centroids is not None:
            self.centroids_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(self.centroids_path), self._centroids)

    def _load_centroids(self) -> None:
        """Load centroids from disk if they exist."""
        if self.centroids_path.exists():
            self._centroids = np.load(str(self.centroids_path))
            self.logger.info(
                f"Loaded centroids from {self.centroids_path}"
            )

    def _heuristic_init(self, embeddings: np.ndarray) -> None:
        """Initialise centroids from available samples when too few for K-Means."""
        embeddings_norm = normalize(embeddings)
        n = embeddings_norm.shape[0]

        if n >= self.n_clusters:
            # Pick evenly spaced samples as initial centroids
            indices = np.linspace(0, n - 1, self.n_clusters, dtype=int)
            self._centroids = embeddings_norm[indices]
        else:
            # Pad with random unit vectors
            dim = embeddings_norm.shape[1]
            pad = np.random.randn(self.n_clusters - n, dim).astype(np.float32)
            pad = normalize(pad)
            self._centroids = np.vstack([embeddings_norm, pad])

        self._centroids = normalize(self._centroids)
        self._save_centroids()
