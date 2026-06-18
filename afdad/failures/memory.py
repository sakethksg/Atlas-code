"""
Failure memory — SQLite-backed persistence for all failure records.

Stores task, code, traceback, failure cluster, embedding, and repair trace.
Supports querying by cluster and nearest-neighbour similarity search.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from afdad.utils.logging import get_logger
from afdad.utils.models import FailureCluster, FailureRecord


class FailureMemory:
    """SQLite-backed failure memory database.

    Parameters
    ----------
    cfg:
        Failure memory configuration with ``db_path`` field.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.db_path = Path(cfg.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ── Database Initialisation ───────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Create the failures table if it doesn't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                record_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                code TEXT NOT NULL,
                traceback TEXT DEFAULT '',
                failure_cluster TEXT DEFAULT 'Unknown',
                embedding BLOB,
                repair_trace TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_failure_cluster
            ON failures(failure_cluster)
        """)
        conn.commit()
        self.logger.info(f"Failure memory initialised at {self.db_path}")

    # ── Write Operations ──────────────────────────────────────

    def store(self, record: FailureRecord) -> None:
        """Persist a failure record to the database.

        Parameters
        ----------
        record:
            The failure record to store.
        """
        conn = self._get_conn()
        embedding_blob = (
            np.array(record.embedding, dtype=np.float32).tobytes()
            if record.embedding
            else None
        )
        repair_json = (
            json.dumps(record.repair_trace)
            if record.repair_trace
            else None
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO failures
            (record_id, task, code, traceback, failure_cluster,
             embedding, repair_trace, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.task,
                record.code,
                record.traceback,
                record.failure_cluster.value,
                embedding_blob,
                repair_json,
                record.created_at.isoformat(),
            ),
        )
        conn.commit()
        self.logger.debug(f"Stored failure record: {record.record_id}")

    def store_batch(self, records: list[FailureRecord]) -> None:
        """Persist multiple failure records."""
        for record in records:
            self.store(record)

    # ── Read Operations ───────────────────────────────────────

    def get_by_cluster(
        self, cluster: str, limit: int = 100
    ) -> list[FailureRecord]:
        """Retrieve failure records by cluster name.

        Parameters
        ----------
        cluster:
            Failure cluster name.
        limit:
            Maximum number of records to return.

        Returns
        -------
        list[FailureRecord]
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM failures
            WHERE failure_cluster = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (cluster, limit),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_all(self, limit: int = 1000) -> list[FailureRecord]:
        """Retrieve all failure records."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM failures ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_all_embeddings(self) -> np.ndarray:
        """Retrieve all stored embeddings as a numpy array.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_records, embedding_dim)`` or
            empty array if no embeddings exist.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT embedding FROM failures WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return np.array([], dtype=np.float32)

        embeddings = [
            np.frombuffer(row["embedding"], dtype=np.float32)
            for row in rows
        ]
        return np.vstack(embeddings)

    def count_by_cluster(self) -> dict[str, int]:
        """Return failure counts per cluster."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT failure_cluster, COUNT(*) as cnt
            FROM failures
            GROUP BY failure_cluster
            """
        ).fetchall()
        return {row["failure_cluster"]: row["cnt"] for row in rows}

    def total_count(self) -> int:
        """Return total number of failure records."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM failures").fetchone()
        return row["cnt"] if row else 0

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[FailureRecord]:
        """Find the most similar failures by cosine similarity.

        Parameters
        ----------
        query_embedding:
            Query embedding vector.
        top_k:
            Number of results to return.

        Returns
        -------
        list[FailureRecord]
            Most similar failure records.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM failures WHERE embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding) + 1e-8
        )
        scored: list[tuple[float, Any]] = []

        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_norm, emb_norm))
            scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_record(row) for _, row in scored[:top_k]]

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FailureRecord:
        """Convert a database row to a FailureRecord."""
        embedding = None
        if row["embedding"]:
            embedding = np.frombuffer(
                row["embedding"], dtype=np.float32
            ).tolist()

        repair_trace = None
        if row["repair_trace"]:
            repair_trace = json.loads(row["repair_trace"])

        # Map cluster string to enum
        try:
            cluster = FailureCluster(row["failure_cluster"])
        except ValueError:
            cluster = FailureCluster.UNKNOWN

        return FailureRecord(
            record_id=row["record_id"],
            task=row["task"],
            code=row["code"],
            traceback=row["traceback"],
            failure_cluster=cluster,
            embedding=embedding,
            repair_trace=repair_trace,
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
