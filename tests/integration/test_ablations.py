"""Integration tests to verify ablation flag graph configurations."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from afdad.graph.workflow import build_graph


def test_ablation_run_0_baseline(mock_cfg: DictConfig) -> None:
    # Run 0: Baseline Student (all flags false)
    cfg = mock_cfg.copy()
    cfg.experiment.experiment.use_agentic_repair = False
    cfg.experiment.experiment.use_distillation = False
    cfg.experiment.experiment.use_failure_clustering = False
    cfg.experiment.experiment.use_adaptive_curriculum = False

    graph = build_graph(cfg)
    node_names = set(graph.get_graph().nodes.keys())

    # Baseline should only have plan, student_generate, execute, success_store, and __start__
    assert "plan" in node_names
    assert "student_generate" in node_names
    assert "execute" in node_names
    assert "success_store" in node_names

    # Should not have any repair, clustering, or distillation nodes
    assert "expert_repair" not in node_names
    assert "failure_cluster" not in node_names
    assert "distillation" not in node_names


def test_ablation_run_1_repair_only(mock_cfg: DictConfig) -> None:
    # Run 1: Student + Agentic Repair (repair=True, others false)
    cfg = mock_cfg.copy()
    cfg.experiment.experiment.use_agentic_repair = True
    cfg.experiment.experiment.use_distillation = False
    cfg.experiment.experiment.use_failure_clustering = False
    cfg.experiment.experiment.use_adaptive_curriculum = False

    graph = build_graph(cfg)
    node_names = set(graph.get_graph().nodes.keys())

    assert "plan" in node_names
    assert "student_generate" in node_names
    assert "execute" in node_names
    assert "failure_analysis" in node_names
    assert "expert_repair" in node_names
    assert "re_execute" in node_names

    # Should not have clustering or distillation nodes
    assert "failure_cluster" not in node_names
    assert "distillation" not in node_names


def test_ablation_run_2_repair_distill(mock_cfg: DictConfig) -> None:
    # Run 2: Student + Repair + Distill (repair=True, distill=True, others false)
    cfg = mock_cfg.copy()
    cfg.experiment.experiment.use_agentic_repair = True
    cfg.experiment.experiment.use_distillation = True
    cfg.experiment.experiment.use_failure_clustering = False
    cfg.experiment.experiment.use_adaptive_curriculum = False

    graph = build_graph(cfg)
    node_names = set(graph.get_graph().nodes.keys())

    assert "plan" in node_names
    assert "student_generate" in node_names
    assert "execute" in node_names
    assert "expert_repair" in node_names
    assert "distillation" in node_names

    # Should not have clustering nodes
    assert "failure_cluster" not in node_names


def test_ablation_run_4_full_afdad(mock_cfg: DictConfig) -> None:
    # Run 4: Full AFDAD (all flags true)
    cfg = mock_cfg.copy()
    cfg.experiment.experiment.use_agentic_repair = True
    cfg.experiment.experiment.use_distillation = True
    cfg.experiment.experiment.use_failure_clustering = True
    cfg.experiment.experiment.use_adaptive_curriculum = True

    graph = build_graph(cfg)
    node_names = set(graph.get_graph().nodes.keys())

    # All nodes must be present
    assert "plan" in node_names
    assert "student_generate" in node_names
    assert "execute" in node_names
    assert "failure_analysis" in node_names
    assert "failure_embed" in node_names
    assert "failure_cluster" in node_names
    assert "expert_repair" in node_names
    assert "re_execute" in node_names
    assert "distillation" in node_names
    assert "success_store" in node_names
