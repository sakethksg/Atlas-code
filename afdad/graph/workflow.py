"""
LangGraph StateGraph assembly — wires all nodes and edges together.

Produces the compiled AFDAD graph that can process coding tasks
through the full pipeline: plan → generate → execute → (repair cycle) → distill.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from afdad.graph.nodes import NodeFactory
from afdad.graph.routing import (
    route_after_execution,
    route_after_re_execution,
    route_after_repair,
)
from afdad.graph.state import AFDADState
from afdad.utils.logging import get_logger


def build_graph(cfg: DictConfig) -> Any:
    """Build and compile the AFDAD LangGraph StateGraph.

    Parameters
    ----------
    cfg:
        Full Hydra configuration.

    Returns
    -------
    CompiledGraph
        The compiled LangGraph ready for invocation.

    Graph Topology
    --------------
    ```
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
    ```
    """
    logger = get_logger()
    logger.info("Building AFDAD LangGraph workflow...")

    # Create node factory with all dependencies
    factory = NodeFactory(cfg)

    # ── Build graph ──
    graph = StateGraph(AFDADState)

    # ── Add nodes ──
    graph.add_node("plan", factory.plan_node)
    graph.add_node("student_generate", factory.student_generate_node)
    graph.add_node("execute", factory.execute_node)
    graph.add_node("success_store", factory.success_store_node)
    graph.add_node("failure_analysis", factory.failure_analysis_node)
    graph.add_node("failure_embed", factory.failure_embed_node)
    graph.add_node("failure_cluster", factory.failure_cluster_node)
    graph.add_node("expert_repair", factory.expert_repair_node)
    graph.add_node("re_execute", factory.execute_node)  # reuse execute
    graph.add_node("distillation", factory.distillation_node)

    # ── Set entry point ──
    graph.set_entry_point("plan")

    # ── Linear edges ──
    graph.add_edge("plan", "student_generate")
    graph.add_edge("student_generate", "execute")
    graph.add_edge("success_store", END)
    graph.add_edge("failure_analysis", "failure_embed")
    graph.add_edge("failure_embed", "failure_cluster")
    graph.add_edge("failure_cluster", "expert_repair")
    graph.add_edge("distillation", END)

    # ── Conditional edges ──
    graph.add_conditional_edges(
        "execute",
        route_after_execution,
        {
            "success_store": "success_store",
            "failure_analysis": "failure_analysis",
        },
    )

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

    # ── Compile ──
    compiled = graph.compile()
    logger.info("AFDAD graph compiled successfully.")

    return compiled


def visualise_graph(cfg: DictConfig, output_path: str = "graph.png") -> None:
    """Render the AFDAD graph as a PNG image.

    Parameters
    ----------
    cfg:
        Full Hydra configuration.
    output_path:
        Path to save the rendered image.
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
