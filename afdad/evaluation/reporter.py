"""Result reporter — generates structured report summaries from evaluations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afdad.utils.logging import get_logger


class ResultReporter:
    """Generates structured Markdown and JSON reports for experiment results.

    Enables tracking of performance improvements across different runs.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def generate_report(
        self,
        run_name: str,
        metrics: dict[str, Any],
        round_id: int | None = None,
    ) -> Path:
        """Create a Markdown and JSON summary report for a single evaluation.

        Parameters
        ----------
        run_name:
            Identifier of the active run/ablation.
        metrics:
            Dictionary of computed benchmark metrics.
        round_id:
            Optional iteration/training round index.

        Returns
        -------
        Path
            Path to the generated Markdown report.
        """
        round_suffix = f"_round_{round_id}" if round_id is not None else ""
        report_base = f"report_{run_name}{round_suffix}"
        
        # Save JSON data
        json_path = self.output_dir / f"{report_base}.json"
        report_data = {
            "run_name": run_name,
            "round_id": round_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        self.logger.info(f"Saved raw evaluation report to {json_path}")

        # Save Markdown report
        md_path = self.output_dir / f"{report_base}.md"
        
        md_lines = [
            f"# Evaluation Report Summary — {run_name}",
            "",
            f"- **Timestamp**: {report_data['timestamp']}",
            f"- **Training Round**: {round_id if round_id is not None else 'N/A'}",
            "",
            "## Benchmark Results",
            "",
            "| Metric | Value |",
            "| :--- | :---: |",
        ]
        
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                md_lines.append(f"| {k} | {v:.4f} |")
            else:
                md_lines.append(f"| {k} | {v} |")

        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        self.logger.info(f"Saved formatted Markdown report to {md_path}")
        
        return md_path

    def compare_runs(
        self,
        comparison_name: str,
        runs_data: list[dict[str, Any]],
    ) -> Path:
        """Generate a comparison report across multiple runs.

        Parameters
        ----------
        comparison_name:
            Identifier for this comparison (e.g. "ablation_study").
        runs_data:
            List of run dicts, each with keys: 'run_name', 'metrics'.
        """
        md_path = self.output_dir / f"compare_{comparison_name}.md"
        
        # Gather all distinct metric names
        all_metrics: set[str] = set()
        for run in runs_data:
            all_metrics.update(run.get("metrics", {}).keys())
        
        sorted_metrics = sorted(list(all_metrics))
        
        # Build headers
        headers = ["Metric"] + [run["run_name"] for run in runs_data]
        alignments = [":---"] + [":---:"] * len(runs_data)
        
        md_lines = [
            f"# Performance Comparison — {comparison_name}",
            "",
            f"- **Generated At**: {datetime.now(timezone.utc).isoformat()}",
            "",
            "|" + "|".join(headers) + "|",
            "|" + "|".join(alignments) + "|",
        ]
        
        for m in sorted_metrics:
            row = [m]
            for run in runs_data:
                val = run.get("metrics", {}).get(m, "N/A")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            md_lines.append("|" + "|".join(row) + "|")

        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        self.logger.info(f"Saved run comparison report to {md_path}")
        
        return md_path
