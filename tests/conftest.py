import sys

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging

# Monkeypatch logging.Logger.findCaller to prevent Python 3.13 access violations
# during frame walk tracing under pytest on Windows.
def _dummy_find_caller(self, stack_info=False, stacklevel=1):
    return ("unknown_file", 0, "unknown_func", None)

logging.Logger.findCaller = _dummy_find_caller
logging.logAsyncioTasks = False

import numpy as np
from unittest.mock import MagicMock
import openai
openai.AsyncOpenAI = MagicMock()

from afdad.failures.encoder import FailureEncoder

def _dummy_encode(self, text: str) -> np.ndarray:
    return np.zeros(self.embedding_dim, dtype=np.float32)

def _dummy_encode_batch(self, texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

FailureEncoder.encode = _dummy_encode
FailureEncoder.encode_batch = _dummy_encode_batch

import pytest

from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def mock_cfg() -> DictConfig:
    """Create a minimal mock Hydra DictConfig configuration."""
    raw_cfg = {
        "output_dir": "./outputs",
        "data_dir": "./data",
        "seed": 42,
        "mode": "train",
        "execution": {
            "timeout_seconds": 5,
            "temp_dir": "./outputs/exec_tmp",
            "max_concurrent_evals": 2,
        },
        "failure_memory": {
            "db_path": "./outputs/failure_memory_test.db",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "embedding_batch_size": 2,
            "similarity_top_k": 2,
            "text_truncation": {
                "task": 50,
                "traceback": 100,
                "stderr": 50,
                "code": 100,
            },
        },
        "clustering": {
            "n_clusters": 3,
            "cluster_names": ["Syntax", "Runtime", "Logic"],
            "centroids_path": "./outputs/centroids_test.npy",
            "min_samples_for_fit": 2,
            "ema_decay": 0.1,
            "seed": 42,
        },
        "trajectories": {
            "buffer_capacity": 10,
            "save_dir": "./outputs/trajectories_test",
            "curriculum_amplification": 5.0,
        },
        "distillation": {
            "dataset_dir": "./outputs/distillation_dataset_test",
            "max_seq_length": 512,
        },
        "evaluation": {
            "benchmarks": ["humaneval"],
            "num_samples_per_task": 1,
            "temperature": 0.0,
        },
        "pipeline": {
            "max_repair_attempts": 2,
            "max_tasks_per_round": 2,
            "num_training_rounds": 1,
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "test_project",
            "wandb_entity": None,
            "log_level": "INFO",
            "rich_markup": False,
        },
        "model": {
            "student": {
                "name": "mock-student",
                "base_url": "http://localhost:8001/v1",
                "api_key": "EMPTY",
                "generation": {
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "top_p": 0.95,
                    "stop_sequences": [],
                },
            },
            "expert": {
                "name": "mock-expert",
                "base_url": "http://localhost:8002/v1",
                "api_key": "EMPTY",
                "generation": {
                    "temperature": 0.0,
                    "max_tokens": 128,
                    "top_p": 0.95,
                    "stop_sequences": [],
                },
            },
        },
        "experiment": {
            "experiment": {
                "name": "test_run",
                "use_agentic_repair": True,
                "use_distillation": True,
                "use_failure_clustering": True,
                "use_adaptive_curriculum": True,
            }
        },
    }
    return OmegaConf.create(raw_cfg)

