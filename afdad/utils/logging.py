"""
Logging utilities — structured Python logging + Weights & Biases integration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler

console = Console()

_LOGGER_NAME = "afdad"
_logger: logging.Logger | None = None


def setup_logging(cfg: DictConfig) -> logging.Logger:
    """Initialise the global logger with rich console output and optional W&B.

    Parameters
    ----------
    cfg:
        The full Hydra config.  Expects ``cfg.logging.log_level``
        and ``cfg.logging.use_wandb``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _logger  # noqa: PLW0603

    level = getattr(logging, cfg.logging.log_level.upper(), logging.INFO)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on re-init
    logger.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
    )
    rich_handler.setLevel(level)
    fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(fmt)
    logger.addHandler(rich_handler)

    # Optional file handler
    file_handler = logging.FileHandler(
        f"{cfg.output_dir}/afdad.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the global AFDAD logger (creates a basic one if not initialised)."""
    global _logger  # noqa: PLW0603
    if _logger is None:
        _logger = logging.getLogger(_LOGGER_NAME)
        if not _logger.handlers:
            handler = RichHandler(console=console, show_path=False)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger


# ── Weights & Biases Helpers ──────────────────────────────────


def init_wandb(cfg: DictConfig) -> Any | None:
    """Initialise a W&B run from the Hydra config.

    Returns the ``wandb.Run`` object, or *None* if W&B is disabled.
    """
    if not cfg.logging.use_wandb:
        get_logger().info("W&B logging disabled.")
        return None

    try:
        import wandb  # noqa: WPS433

        run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment.experiment.name,
            config=dict(cfg),
            reinit=True,
        )
        get_logger().info(f"W&B run initialised: [bold]{run.name}[/bold]")
        return run
    except Exception as exc:
        get_logger().warning(f"Failed to initialise W&B: {exc}")
        return None


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to both the console logger and W&B (if active)."""
    logger = get_logger()
    logger.info(f"Metrics (step={step}): {metrics}")

    try:
        import wandb  # noqa: WPS433

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass
