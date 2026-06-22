"""
Reproducibility utilities — centralised seed management and experiment snapshots.

Ensures deterministic behavior across all random number generators used
in the AFDAD pipeline: Python's ``random``, NumPy, and PyTorch.

Research Significance:
    Reproducibility is a core requirement for publication-quality research.
    This module provides a single ``seed_everything()`` call that propagates
    a seed to all RNG sources, and ``save_experiment_snapshot()`` to persist
    the full configuration, dependency versions, and git commit hash for
    each experiment run.

Usage::

    from afdad.utils.reproducibility import seed_everything, save_experiment_snapshot

    seed_everything(cfg.seed)
    save_experiment_snapshot(cfg, cfg.output_dir)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int) -> None:
    """Set random seeds for all RNG sources used in the pipeline.

    Propagates the seed to:
    - ``random`` (Python stdlib)
    - ``numpy.random``
    - ``torch`` (if available)
    - ``PYTHONHASHSEED`` environment variable

    Args:
        seed: The global random seed.

    Side Effects:
        Sets ``os.environ["PYTHONHASHSEED"]`` and configures all
        random number generators for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Note: torch.backends.cudnn.deterministic = True can hurt performance.
            # We document this trade-off but do not set it by default.
    except ImportError:
        pass


def get_seeded_rng(seed: int) -> np.random.Generator:
    """Create an isolated NumPy random generator with the given seed.

    Use this for components that need independent, reproducible randomness
    without affecting the global RNG state.

    Args:
        seed: Seed for the generator.

    Returns:
        A NumPy ``Generator`` instance.
    """
    return np.random.default_rng(seed)


def save_experiment_snapshot(
    cfg: DictConfig,
    output_dir: str | Path,
) -> Path:
    """Save a reproducibility snapshot for the current experiment.

    Persists:
    - Full resolved Hydra configuration as YAML
    - Python version and platform info
    - Installed package versions (pip freeze equivalent)
    - Git commit hash and diff status (if in a git repo)
    - Timestamp

    Args:
        cfg: The full Hydra configuration.
        output_dir: Directory to save the snapshot.

    Returns:
        Path to the saved snapshot JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "config_yaml": OmegaConf.to_yaml(cfg, resolve=True),
    }

    # Git info
    snapshot["git"] = _get_git_info()

    # Installed packages
    snapshot["packages"] = _get_installed_packages()

    # Config hash for quick comparison
    config_str = OmegaConf.to_yaml(cfg, resolve=True)
    snapshot["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    snapshot_path = output_dir / "experiment_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, default=str),
        encoding="utf-8",
    )

    return snapshot_path


def _get_git_info() -> dict[str, str | bool]:
    """Retrieve git repository information if available."""
    info: dict[str, str | bool] = {"available": False}

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            info["available"] = True
            info["commit_hash"] = commit.stdout.strip()

            # Check for uncommitted changes
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            info["is_dirty"] = bool(status.stdout.strip())

            # Get branch name
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if branch.returncode == 0:
                info["branch"] = branch.stdout.strip()

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def _get_installed_packages() -> dict[str, str]:
    """Retrieve installed package versions for key dependencies."""
    packages: dict[str, str] = {}
    key_deps = [
        "langgraph", "langchain-core", "openai", "pydantic",
        "hydra-core", "omegaconf", "sentence-transformers",
        "scikit-learn", "numpy", "torch", "transformers",
        "peft", "trl", "datasets", "accelerate", "wandb",
    ]

    for pkg_name in key_deps:
        try:
            from importlib.metadata import version

            packages[pkg_name] = version(pkg_name)
        except Exception:
            packages[pkg_name] = "not installed"

    return packages
