"""Experiment tracker — logs and tracks meta-parameters across training rounds."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from afdad.utils.logging import get_logger


class ExperimentTracker:
    """Manages metadata and tracks metrics across iterative rounds.

    Ensures we have a local, persistent record of the entire distillation lifecycle.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_file = self.output_dir / "experiment_tracker.json"
        
        # Load existing data or initialize
        self._data = self._load_tracker()

    def log_round(
        self,
        round_id: int,
        metrics: dict[str, Any],
        dataset_stats: dict[str, Any] | None = None,
    ) -> None:
        """Log structured parameters and stats from a training round.

        Parameters
        ----------
        round_id:
            Active round index.
        metrics:
            Dictionary of metrics gathered during evaluation/training.
        dataset_stats:
            Statistics describing the current distillation dataset.
        """
        round_entry = {
            "round": round_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "dataset_stats": dataset_stats or {},
        }
        
        # Check if round already exists and update, or append
        rounds = self._data.setdefault("rounds", [])
        for i, r in enumerate(rounds):
            if r["round"] == round_id:
                rounds[i] = round_entry
                break
        else:
            rounds.append(round_entry)
            
        self._save_tracker()
        self.logger.info(f"Logged tracker metadata for Round {round_id}")

    def save_summary(self) -> Path:
        """Generate a structured summary report of the experiment run.

        Returns
        -------
        Path
            Path to the saved tracker file.
        """
        self._data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_tracker()
        
        # Generate summary markdown report
        summary_path = self.output_dir / "experiment_summary.md"
        
        exp_name = self.cfg.get("experiment", {}).get("experiment", {}).get("name", "unknown_run")
        
        md_lines = [
            f"# Experiment Summary — {exp_name}",
            "",
            f"- **Start Time**: {self._data.get('started_at', 'N/A')}",
            f"- **End Time**: {self._data.get('completed_at', 'N/A')}",
            f"- **Total Rounds**: {len(self._data.get('rounds', []))}",
            "",
            "## Iterative Progress",
            "",
            "| Round | Timestamp | Key Metrics |",
            "| :---: | :--- | :--- |",
        ]
        
        for r in self._data.get("rounds", []):
            round_num = r["round"]
            ts = r["timestamp"]
            
            # Extract key metrics like pass@1
            metric_strs = []
            for k, v in r.get("metrics", {}).items():
                if "pass@1" in k:
                    metric_strs.append(f"{k}: {v:.4f}")
            
            metrics_summary = ", ".join(metric_strs) if metric_strs else "None"
            md_lines.append(f"| {round_num} | {ts} | {metrics_summary} |")

        summary_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        self.logger.info(f"Saved experiment execution summary to {summary_path}")
        
        return self.tracker_file

    def _load_tracker(self) -> dict[str, Any]:
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                self.logger.warning(f"Failed to read existing tracker file: {exc}")
                
        # Initialize new tracker layout
        return {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "seed": self.cfg.seed,
                "num_training_rounds": self.cfg.pipeline.num_training_rounds,
                "max_tasks_per_round": self.cfg.pipeline.max_tasks_per_round,
            },
            "rounds": [],
        }

    def _save_tracker(self) -> None:
        try:
            with open(self.tracker_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as exc:
            self.logger.error(f"Failed to write experiment tracker: {exc}")
