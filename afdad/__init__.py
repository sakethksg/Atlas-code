"""
AFDAD — Adaptive Failure-Driven Agentic Distillation.

A research framework that improves code-generation capabilities of Small
Language Models (SLMs) through execution-grounded agentic learning.

Modules:
    agents:        LLM-backed agents (Planner, Coder, Debugger, Critic).
    configs:       Hydra YAML configuration hierarchy.
    datasets:      Benchmark data loaders (HumanEval, MBPP).
    distillation:  Dataset building and QLoRA fine-tuning.
    evaluation:    Benchmark runners and metrics.
    execution:     Subprocess-based code execution and test evaluation.
    failures:      Failure embedding, clustering, and memory persistence.
    graph:         LangGraph state, nodes, routing, and workflow assembly.
    prompts:       Prompt templates for all agents.
    trajectories:  Repair trajectory collection and replay buffer.
    training:      End-to-end training loop orchestration.
    utils:         Logging, data models, reproducibility utilities.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
