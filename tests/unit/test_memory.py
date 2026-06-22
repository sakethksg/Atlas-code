"""Unit tests for the FailureMemory persistence layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from afdad.failures.memory import FailureMemory
from afdad.utils.models import FailureCluster, FailureRecord


@pytest.fixture
def memory_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create({
        "db_path": str(tmp_path / "test_failures.db"),
    })


def test_failure_memory_store_and_query(memory_cfg: DictConfig) -> None:
    memory = FailureMemory(memory_cfg)
    
    # Store a failure record
    record = FailureRecord(
        task="Test task description",
        code="print('failed')",
        traceback="ZeroDivisionError",
        failure_cluster=FailureCluster.LOGIC,
        embedding=[0.1] * 10,
    )
    memory.store(record)
    
    # Retrieve
    records = memory.get_by_cluster("Logic")
    assert len(records) == 1
    assert records[0].task == "Test task description"
    assert records[0].failure_cluster == FailureCluster.LOGIC
    
    # Total count
    assert memory.total_count() == 1
    
    # Counts by cluster
    counts = memory.count_by_cluster()
    assert counts["Logic"] == 1
    
    # Clean up connection
    memory.close()


def test_failure_memory_similarity_search(memory_cfg: DictConfig) -> None:
    memory = FailureMemory(memory_cfg)
    
    # Store two records with distinct embeddings
    rec1 = FailureRecord(
        task="Task 1",
        code="",
        traceback="",
        failure_cluster=FailureCluster.SYNTAX,
        embedding=[1.0, 0.0, 0.0],
    )
    rec2 = FailureRecord(
        task="Task 2",
        code="",
        traceback="",
        failure_cluster=FailureCluster.RUNTIME,
        embedding=[0.0, 1.0, 0.0],
    )
    memory.store_batch([rec1, rec2])
    
    # Query similar to [0.9, 0.1, 0.0]
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    results = memory.search_similar(query, top_k=1)
    
    assert len(results) == 1
    assert results[0].task == "Task 1"
    
    memory.close()
