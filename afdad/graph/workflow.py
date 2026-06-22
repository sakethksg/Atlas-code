"""
LangGraph StateGraph assembly — wires all nodes and edges together.

Produces the compiled AFDAD graph that can process coding tasks
through the full pipeline: plan → generate → execute → (repair cycle) → distill.

The graph topology is dynamically configured based on experiment ablation
flags, enabling the five ablation configurations described in the paper:

- Run 0 — Baseline Student: all flags false
- Run 1 — Student + Agentic Repair: ``use_agentic_repair=true``
- Run 2 — Student + Agentic Repair + Distillation: ``use_agentic_repair=true``,
  ``use_distillation=true``
- Run 3 — Student + Distillation + Failure Clustering:
  ``use_distillation=true``, ``use_failure_clustering=true``
- Run 4 — Full AFDAD: all flags true

Research Significance:
    The ablation flags allow isolating the contribution of each AFDAD
    component. Without them, all runs would execute the full pipeline,
    making ablation studies impossible.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from afdad.graph.nodes import create_nodes
from afdad.graph.routing import (
    route_after_execution,
    route_after_re_execution,
    route_after_repair,
)
from afdad.graph.state import AFDADState
from afdad.utils.logging import get_logger


def _get_experiment_flags(cfg: DictConfig) -> dict[str, bool]:
    """Extract experiment ablation flags from the Hydra config.

    Args:
        cfg: Full Hydra configuration.

    Returns:
        Dictionary of ablation flags with safe defaults.
    """
    exp = cfg.get("experiment", {})
    if hasattr(exp, "experiment"):
        exp = exp.experiment

    return {
        "use_agentic_repair": exp.get("use_agentic_repair", True),
        "use_distillation": exp.get("use_distillation", True),
        "use_failure_clustering": exp.get("use_failure_clustering", True),
        "use_adaptive_curriculum": exp.get("use_adaptive_curriculum", True),
    }


def build_graph(cfg: DictConfig) -> Any:
    """Build and compile the AFDAD LangGraph StateGraph.

    The graph topology adapts to the experiment's ablation flags:

    - **Full AFDAD** (all flags true)::

        plan → student_generate → execute
            ↓ (success)        ↓ (failure)
        success_store      failure_analysis
                               ↓
                           failure_embed
                               ↓
                           failure_cluster
                               ↓
                           expert_repair
                        ↓ (verified)     ↓ (unverified & attempts left)
                      re_execute         expert_repair (retry)
                    ↓ (pass)  ↓ (fail)       ↓ (max attempts)
                distillation  failure_analysis  distillation
                    ↓
                   END

    - **No agentic repair** (``use_agentic_repair=false``):
      failures skip the repair team and go directly to distillation
      (if enabled) or END.

    - **No failure clustering** (``use_failure_clustering=false``):
      failure embedding and clustering nodes are skipped; failures go
      directly to expert repair (if enabled).

    - **No distillation** (``use_distillation=false``):
      repaired tasks go to END instead of the distillation node.

    Args:
        cfg: Full Hydra configuration.

    Returns:
        The compiled LangGraph ready for invocation.
    """
    logger = get_logger()
    flags = _get_experiment_flags(cfg)

    logger.info("Building AFDAD LangGraph workflow...")
    logger.info(f"  Ablation flags: {flags}")

    # Create node mapping with all dependencies bound
    nodes = create_nodes(cfg)

    # ── Build graph ──
    graph = StateGraph(AFDADState)

    # ── Always-present nodes ──
    graph.add_node("plan", nodes["plan_node"])
    graph.add_node("student_generate", nodes["student_generate_node"])
    graph.add_node("execute", nodes["execute_node"])
    graph.add_node("success_store", nodes["success_store_node"])

    # ── Set entry point ──
    graph.set_entry_point("plan")

    # ── Linear edges (always present) ──
    graph.add_edge("plan", "student_generate")
    graph.add_edge("student_generate", "execute")
    graph.add_edge("success_store", END)

    # ── Conditional topology based on ablation flags ──

    if not flags["use_agentic_repair"] and not flags["use_distillation"]:
        # ─── Baseline (Run 0): No repair, no distillation ───
        # Failures go directly to END (just log stats)
        graph.add_conditional_edges(
            "execute",
            route_after_execution,
            {
                "success_store": "success_store",
                "failure_analysis": "success_store",  # No repair, treat as terminal
            },
        )

    elif not flags["use_agentic_repair"] and flags["use_distillation"]:
        # ─── Run 3 variant: No repair but with distillation ───
        # Failures go through analysis for distillation without expert repair
        graph.add_node("failure_analysis", nodes["failure_analysis_node"])
        graph.add_node("distillation", nodes["distillation_node"])

        if flags["use_failure_clustering"]:
            graph.add_node("failure_embed", nodes["failure_embed_node"])
            graph.add_node("failure_cluster", nodes["failure_cluster_node"])
            graph.add_edge("failure_analysis", "failure_embed")
            graph.add_edge("failure_embed", "failure_cluster")
            graph.add_edge("failure_cluster", "distillation")
        else:
            graph.add_edge("failure_analysis", "distillation")

        graph.add_edge("distillation", END)

        graph.add_conditional_edges(
            "execute",
            route_after_execution,
            {
                "success_store": "success_store",
                "failure_analysis": "failure_analysis",
            },
        )

    else:
        # ─── Runs 1, 2, 4: With agentic repair ───
        graph.add_node("failure_analysis", nodes["failure_analysis_node"])
        graph.add_node("expert_repair", nodes["expert_repair_node"])

        if flags["use_failure_clustering"]:
            graph.add_node("failure_embed", nodes["failure_embed_node"])
            graph.add_node("failure_cluster", nodes["failure_cluster_node"])
            graph.add_edge("failure_analysis", "failure_embed")
            graph.add_edge("failure_embed", "failure_cluster")
            graph.add_edge("failure_cluster", "expert_repair")
        else:
            # Skip clustering — go directly from failure analysis to repair
            graph.add_edge("failure_analysis", "expert_repair")

        graph.add_conditional_edges(
            "execute",
            route_after_execution,
            {
                "success_store": "success_store",
                "failure_analysis": "failure_analysis",
            },
        )

        if flags["use_distillation"]:
            # ─── With distillation (Runs 2, 4) ───
            graph.add_node("re_execute", nodes["execute_node"])
            graph.add_node("distillation", nodes["distillation_node"])
            graph.add_edge("distillation", END)

            graph.add_conditional_edges(
                "expert_repair",
                route_after_repair,
                {
                    "re_execute": "re_execute",
                    "expert_repair": "expert_repair",
                    "distillation": "distillation",
                },
            )

            graph.add_conditional_edges(
                "re_execute",
                route_after_re_execution,
                {
                    "distillation": "distillation",
                    "failure_analysis": "failure_analysis",
                },
            )
        else:
            # ─── Without distillation (Run 1) ───
            # After repair, re-execute and go to END
            graph.add_node("re_execute", nodes["execute_node"])

            graph.add_conditional_edges(
                "expert_repair",
                route_after_repair,
                {
                    "re_execute": "re_execute",
                    "expert_repair": "expert_repair",
                    "distillation": "success_store",  # Route to END via success_store
                },
            )

            graph.add_conditional_edges(
                "re_execute",
                route_after_re_execution,
                {
                    "distillation": "success_store",  # Route to END via success_store
                    "failure_analysis": "failure_analysis",
                },
            )

    # ── Compile ──
    compiled = graph.compile()
    logger.info("AFDAD graph compiled successfully.")

    return compiled



def visualise_graph(cfg: DictConfig, output_path: str = "graph.png") -> None:
    """Render the AFDAD graph as a PNG image.

    Args:
        cfg: Full Hydra configuration.
        output_path: Path to save the rendered image.
    """
    compiled = build_graph(cfg)

    try:
        png_data = compiled.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        get_logger().info(f"Graph visualisation saved to {output_path}")
    except Exception as exc:
        get_logger().warning(
            f"Could not render graph (install graphviz?): {exc}"
        )
