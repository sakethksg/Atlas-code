"""
Logging utilities — structured Python logging + Weights & Biases integration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler

console = Console()

_LOGGER_NAME = "afdad"
_logger: logging.Logger | None = None
_JSON_LOG_PATH: str | None = None


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
    global _logger, _JSON_LOG_PATH  # noqa: PLW0603

    level = getattr(logging, cfg.logging.log_level.upper(), logging.INFO)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on re-init
    logger.handlers.clear()

    # Use standard StreamHandler under pytest or when stdout is not a TTY to prevent frame inspection and console crashes
    if "pytest" in sys.modules or sys.stdout is None or not sys.stdout.isatty():
        rich_handler = logging.StreamHandler(sys.stdout or sys.stderr)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    else:
        # Rich console handler (markup is configurable, useful for clean CI output)
        rich_markup = cfg.get("logging", {}).get("rich_markup", True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=rich_markup,
        )
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        
    rich_handler.setLevel(level)
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

    # Structured JSON log path
    _JSON_LOG_PATH = os.path.join(cfg.output_dir, "events.jsonl")

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the global AFDAD logger (creates a basic one if not initialised)."""
    global _logger  # noqa: PLW0603
    if _logger is None:
        _logger = logging.getLogger(_LOGGER_NAME)
        if not _logger.handlers:
            if "pytest" in sys.modules or sys.stdout is None or not sys.stdout.isatty():
                handler = logging.StreamHandler(sys.stdout or sys.stderr)
                fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
            else:
                handler = RichHandler(console=console, show_path=False)
                fmt = logging.Formatter("%(message)s")
            handler.setFormatter(fmt)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger



def log_event(event_name: str, **data: Any) -> None:
    """Log a structured JSON event to the events.jsonl file for machine parseability.

    Parameters
    ----------
    event_name:
        Category or identifier of the event.
    data:
        Key-value metadata associated with the event.
    """
    global _JSON_LOG_PATH
    logger = get_logger()

    event = {
        "event": event_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }

    if _JSON_LOG_PATH is not None:
        try:
            # Ensure folder exists
            os.makedirs(os.path.dirname(_JSON_LOG_PATH), exist_ok=True)
            with open(_JSON_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write event {event_name} to events.jsonl: {exc}")

    # Log structured message for debug tracking
    logger.debug(f"Event: {event_name} | {json.dumps(data)}")


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

        # Safely resolve name
        run_name = "afdad_run"
        exp = cfg.get("experiment", {})
        if hasattr(exp, "experiment"):
            run_name = exp.experiment.get("name", run_name)
        elif isinstance(exp, dict):
            run_name = exp.get("name", run_name)

        run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=run_name,
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
    except Exception as exc:
        logger.warning(f"Failed to log metrics to W&B: {exc}")

