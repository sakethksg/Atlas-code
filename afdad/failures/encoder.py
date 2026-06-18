"""
Failure embedding encoder — generates vector representations of failures.

Uses sentence-transformers (bge-large-en-v1.5) to encode structured
failure descriptions for clustering and similarity search.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig

from afdad.utils.logging import get_logger


class FailureEncoder:
    """Encodes failure descriptions into dense vector embeddings.

    Parameters
    ----------
    cfg:
        Failure memory configuration with ``embedding_model`` and
        ``embedding_dim`` fields.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.model_name: str = cfg.embedding_model
        self.embedding_dim: int = cfg.embedding_dim
        self.logger = get_logger()
        self._model: Any | None = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return

        self.logger.info(
            f"Loading embedding model: [bold]{self.model_name}[/bold]"
        )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self.logger.info("Embedding model loaded.")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single failure description.

        Parameters
        ----------
        text:
            Structured failure description string.

        Returns
        -------
        np.ndarray
            1-D embedding vector of shape ``(embedding_dim,)``.
        """
        self._load_model()
        embedding = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embedding, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of failure descriptions.

        Parameters
        ----------
        texts:
            List of failure description strings.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(len(texts), embedding_dim)``.
        """
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def build_failure_text(
        task: str,
        code: str,
        error_type: str | None,
        stderr: str,
        traceback: str,
    ) -> str:
        """Construct a structured text representation of a failure.

        This combines all relevant failure information into a single
        string suitable for embedding.
        """
        parts = [
            f"Task: {task[:200]}",
            f"Error Type: {error_type or 'Unknown'}",
            f"Traceback: {traceback[:500]}" if traceback else "",
            f"Stderr: {stderr[:300]}" if stderr else "",
            f"Code Snippet: {code[:300]}",
        ]
        return "\n".join(p for p in parts if p)
