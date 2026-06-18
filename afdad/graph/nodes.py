"""
LangGraph node functions — each node reads from and writes to AFDADState.

All nodes are async functions: (state: AFDADState) → partial state update.
They compose the full AFDAD pipeline from planning through distillation.
"""

from __future__ import annotations

import uuid
from typing import Any

from omegaconf import DictConfig

from afdad.agents.coder import CoderAgent
from afdad.agents.critic import CriticAgent
from afdad.agents.debugger import DebuggerAgent
from afdad.agents.planner import PlannerAgent
from afdad.execution.evaluator import Evaluator
from afdad.failures.clustering import FailureClustering
from afdad.failures.encoder import FailureEncoder
from afdad.failures.memory import FailureMemory
from afdad.graph.state import AFDADState
from afdad.trajectories.collector import TrajectoryCollector
from afdad.utils.logging import get_logger, log_metrics
from afdad.utils.models import (
    ExecutionResult,
    FailureCluster,
    FailureRecord,
)


# ══════════════════════════════════════════════════════════════
# Node Factory — creates node functions bound to components
# ══════════════════════════════════════════════════════════════


class NodeFactory:
    """Creates LangGraph node functions with injected dependencies.

    Parameters
    ----------
    cfg:
        Full Hydra configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger()

        # ── Agents ──
        self.planner = PlannerAgent(cfg.model.student)
        self.coder = CoderAgent(cfg.model.student)
        self.debugger = DebuggerAgent(cfg.model.expert)
        self.critic = CriticAgent(cfg.model.expert)

        # ── Execution ──
        self.evaluator = Evaluator(cfg.sandbox)

        # ── Failure Pipeline ──
        self.encoder = FailureEncoder(cfg.failure_memory)
        self.clustering = FailureClustering(cfg.clustering)
        self.memory = FailureMemory(cfg.failure_memory)

        # ── Trajectories ──
        self.collector = TrajectoryCollector(cfg.trajectories)

    # ──────────────────────────────────────────────────────────
    # Node 1: Planner
    # ──────────────────────────────────────────────────────────

    async def plan_node(self, state: AFDADState) -> dict[str, Any]:
        """Generate an algorithmic plan for the coding task."""
        self.logger.info("[bold cyan]▶ Planner Node[/bold cyan]")

        task = state["task"]
        plan_output = await self.planner.generate(task=task)

        plan_str = (
            f"Strategy: {plan_output.algorithm_strategy}\n"
            f"Complexity: {plan_output.complexity_estimate}\n"
            f"Edge Cases: {', '.join(plan_output.edge_cases)}\n"
            f"Observations: {plan_output.key_observations}"
        )

        self.logger.info(f"Plan generated ({len(plan_str)} chars)")
        return {"plan": plan_str}

    # ──────────────────────────────────────────────────────────
    # Node 2: Student Generation
    # ──────────────────────────────────────────────────────────

    async def student_generate_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Generate code using the student SLM."""
        self.logger.info("[bold green]▶ Student Generation Node[/bold green]")

        task = state["task"]
        plan = state.get("plan", "")

        code = await self.coder.generate(task=task, plan=plan)

        self.logger.info(f"Student code generated ({len(code)} chars)")
        return {
            "student_code": code,
            "trajectory_id": str(uuid.uuid4()),
            "expert_called": False,
        }

    # ──────────────────────────────────────────────────────────
    # Node 3: Execution
    # ──────────────────────────────────────────────────────────

    async def execute_node(self, state: AFDADState) -> dict[str, Any]:
        """Execute the generated code and run tests."""
        self.logger.info("[bold yellow]▶ Execution Node[/bold yellow]")

        # Use repaired code if available, otherwise student code
        code = state.get("repaired_code") or state["student_code"]
        test_cases = state.get("test_cases", [])
        entry_point = state.get("entry_point", "")

        # Build test code from test cases
        test_code = ""
        if test_cases:
            test_code = "\n".join(
                tc.get("test_code", "") for tc in test_cases
            )

        result = await self.evaluator.evaluate(
            code=code,
            test_code=test_code,
            entry_point=entry_point,
        )

        success = result.is_success

        self.logger.info(
            f"Execution: {'✅ PASS' if success else '❌ FAIL'} "
            f"({result.passed}/{result.total} tests)"
        )

        log_metrics({
            "tests_passed": result.passed,
            "tests_failed": result.failed,
            "tests_total": result.total,
            "execution_time_ms": result.execution_time_ms,
        })

        return {
            "execution_result": result.model_dump(),
            "success": success,
            "attempt": state.get("attempt", 0) + (0 if success else 1),
        }

    # ──────────────────────────────────────────────────────────
    # Node 4: Failure Analysis
    # ──────────────────────────────────────────────────────────

    async def failure_analysis_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Analyse the failure — construct structured failure description."""
        self.logger.info("[bold red]▶ Failure Analysis Node[/bold red]")

        task = state["task"]
        code = state.get("repaired_code") or state["student_code"]
        exec_result = ExecutionResult(**state["execution_result"])

        failure_text = FailureEncoder.build_failure_text(
            task=task,
            code=code,
            error_type=exec_result.error_type,
            stderr=exec_result.stderr,
            traceback=exec_result.traceback,
        )

        self.logger.info(
            f"Failure analysed: {exec_result.error_type or 'Unknown'}"
        )

        return {"failure_description": failure_text}

    # ──────────────────────────────────────────────────────────
    # Node 5: Failure Embedding
    # ──────────────────────────────────────────────────────────

    async def failure_embed_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Generate embedding for the failure description."""
        self.logger.info("[bold magenta]▶ Failure Embedding Node[/bold magenta]")

        failure_text = state["failure_description"]
        embedding = self.encoder.encode(failure_text)

        return {"failure_embedding": embedding.tolist()}

    # ──────────────────────────────────────────────────────────
    # Node 6: Failure Clustering
    # ──────────────────────────────────────────────────────────

    async def failure_cluster_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Assign the failure to a cluster."""
        self.logger.info("[bold blue]▶ Failure Clustering Node[/bold blue]")

        import numpy as np

        embedding = np.array(state["failure_embedding"], dtype=np.float32)
        cluster_name = self.clustering.predict(embedding)

        # Store in failure memory
        exec_result = ExecutionResult(**state["execution_result"])
        record = FailureRecord(
            task=state["task"],
            code=state.get("repaired_code") or state["student_code"],
            traceback=exec_result.traceback,
            failure_cluster=FailureCluster(cluster_name)
            if cluster_name in [e.value for e in FailureCluster]
            else FailureCluster.UNKNOWN,
            embedding=state["failure_embedding"],
        )
        self.memory.store(record)

        # Update clustering with new embedding
        self.clustering.partial_fit(embedding.reshape(1, -1))

        self.logger.info(f"Failure cluster: [bold]{cluster_name}[/bold]")

        log_metrics({"failure_cluster": cluster_name})

        return {"failure_cluster": cluster_name}

    # ──────────────────────────────────────────────────────────
    # Node 7: Expert Repair
    # ──────────────────────────────────────────────────────────

    async def expert_repair_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Run the agentic repair workflow: Debugger → Critic."""
        self.logger.info(
            "[bold red]▶ Expert Repair Node[/bold red] "
            "(Debugger → Critic)"
        )

        task = state["task"]
        plan = state.get("plan", "")
        failed_code = state.get("repaired_code") or state["student_code"]
        exec_result = ExecutionResult(**state["execution_result"])

        # Create trajectory
        trajectory = self.collector.create_trajectory()

        # ── Step 1: Debugger ──
        self.logger.info("  └─ Running Debugger agent...")
        debug_result = await self.debugger.generate(
            task=task,
            plan=plan,
            failed_code=failed_code,
            returncode=exec_result.returncode,
            passed=exec_result.passed,
            total=exec_result.total,
            error_type=exec_result.error_type or "Unknown",
            stderr=exec_result.stderr,
            stdout=exec_result.stdout,
            traceback=exec_result.traceback,
        )

        analysis = debug_result["analysis"]
        repaired_code = debug_result["repaired_code"]

        trajectory = self.collector.add_step(
            trajectory,
            agent="Debugger",
            action="analyse_and_repair",
            reasoning=analysis.detailed_explanation,
            output=repaired_code,
        )

        # ── Step 2: Critic ──
        self.logger.info("  └─ Running Critic agent...")
        critic_result = await self.critic.generate(
            task=task,
            failed_code=failed_code,
            error_type=exec_result.error_type or "Unknown",
            stderr=exec_result.stderr,
            repaired_code=repaired_code,
            repair_reasoning=analysis.detailed_explanation,
        )

        trajectory = self.collector.add_step(
            trajectory,
            agent="Critic",
            action="validate_repair",
            reasoning=critic_result.get("reasoning", ""),
            output=str(critic_result.get("is_valid", False)),
        )

        # Finalise trajectory
        repair_reasoning = (
            f"Root cause: {analysis.root_cause}\n"
            f"Category: {analysis.error_category}\n"
            f"Fix: {analysis.suggested_fix_strategy}\n"
            f"Critic verdict: {'Valid' if critic_result.get('is_valid') else 'Invalid'}"
        )

        trajectory = self.collector.finalise(
            trajectory,
            repaired_code=repaired_code,
            repair_reasoning=repair_reasoning,
            verified=critic_result.get("is_valid", False),
        )

        # Save trajectory
        self.collector.save_trajectory(trajectory)

        self.logger.info(
            f"Repair {'✅ verified' if critic_result.get('is_valid') else '⚠️ unverified'}"
        )

        log_metrics({"expert_called": 1, "repair_verified": int(critic_result.get("is_valid", False))})

        return {
            "repaired_code": repaired_code,
            "repair_trace": trajectory.model_dump(),
            "repair_verified": critic_result.get("is_valid", False),
            "expert_called": True,
            "trajectory_id": trajectory.trajectory_id,
        }

    # ──────────────────────────────────────────────────────────
    # Node 8: Distillation
    # ──────────────────────────────────────────────────────────

    async def distillation_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Generate a training example from the repair trajectory."""
        self.logger.info("[bold white]▶ Distillation Node[/bold white]")

        from afdad.utils.models import RepairTrajectory

        repair_trace = state.get("repair_trace", {})
        trajectory = RepairTrajectory(**repair_trace)

        training_example = self.collector.to_training_example(
            trajectory=trajectory,
            task=state["task"],
            plan=state.get("plan", ""),
            failed_code=state["student_code"],
            failure_cluster=state.get("failure_cluster", "Unknown"),
        )

        self.logger.info(
            f"Training example generated (cluster: {training_example.failure_cluster.value})"
        )

        return {"training_example": training_example.model_dump()}

    # ──────────────────────────────────────────────────────────
    # Node 9: Success Store (terminal)
    # ──────────────────────────────────────────────────────────

    async def success_store_node(
        self, state: AFDADState
    ) -> dict[str, Any]:
        """Store stats for a successful task (no repair needed)."""
        self.logger.info("[bold green]▶ Success — storing stats[/bold green]")

        log_metrics({
            "task_success": 1,
            "expert_called": int(state.get("expert_called", False)),
        })

        return {}
