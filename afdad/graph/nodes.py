"""
LangGraph node functions — each node reads from and writes to AFDADState.

All nodes are async functions: ``(state: AFDADState) → partial state update``.
They compose the full AFDAD pipeline from planning through distillation.

Design Rationale:
    Rather than a monolithic NodeFactory class, nodes are implemented as
    independent, standalone functions. The factory function ``create_nodes(cfg)``
    constructs all dependencies once and returns a mapping of node names
    to bound async callables. Individual nodes can be tested in isolation
    by passing mock dependencies directly.

Research Significance:
    Each node corresponds to a distinct stage in the AFDAD pipeline
    (Figure 1 in the paper): Plan → Generate → Execute → Analyse →
    Embed → Cluster → Repair → Distill.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Awaitable

import numpy as np
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
from afdad.utils.logging import get_logger, log_event, log_metrics
from afdad.utils.models import (
    ExecutionResult,
    FailureCluster,
    FailureRecord,
)

# Type alias for LangGraph node functions
NodeFn = Callable[[AFDADState], Awaitable[dict[str, Any]]]


# ──────────────────────────────────────────────────────────
# Standalone Node Functions (independently testable)
# ──────────────────────────────────────────────────────────

async def plan_node(
    state: AFDADState,
    planner: PlannerAgent,
    logger: Any = None,
) -> dict[str, Any]:
    """Generate an algorithmic plan for the coding task.

    Reads:
        ``state["task"]``

    Writes:
        ``plan`` — formatted algorithmic strategy string.
    """
    logger = logger or get_logger()
    logger.info("[bold cyan]▶ Planner Node[/bold cyan]")

    task = state["task"]
    plan_output = await planner.generate(task=task)

    plan_str = (
        f"Strategy: {plan_output.algorithm_strategy}\n"
        f"Complexity: {plan_output.complexity_estimate}\n"
        f"Edge Cases: {', '.join(plan_output.edge_cases)}\n"
        f"Observations: {plan_output.key_observations}"
    )

    logger.info(f"Plan generated ({len(plan_str)} chars)")
    log_event(
        "planner_executed",
        task_id=state.get("task_id", ""),
        plan_len=len(plan_str),
    )
    return {"plan": plan_str}


async def student_generate_node(
    state: AFDADState,
    coder: CoderAgent,
    logger: Any = None,
) -> dict[str, Any]:
    """Generate code using the student SLM.

    Reads:
        ``state["task"]``, ``state["plan"]``

    Writes:
        ``student_code``, ``trajectory_id``, ``expert_called``
    """
    logger = logger or get_logger()
    logger.info("[bold green]▶ Student Generation Node[/bold green]")

    task = state["task"]
    plan = state.get("plan", "")

    code = await coder.generate(task=task, plan=plan)

    logger.info(f"Student code generated ({len(code)} chars)")
    log_event(
        "student_generator_executed",
        task_id=state.get("task_id", ""),
        trajectory_id=state.get("trajectory_id", ""),
        code_len=len(code),
    )
    return {
        "student_code": code,
        "trajectory_id": str(uuid.uuid4()),
        "expert_called": False,
    }


async def execute_node(
    state: AFDADState,
    evaluator: Evaluator,
    logger: Any = None,
) -> dict[str, Any]:
    """Execute the generated code and run tests.

    Uses repaired code if available, otherwise student code.

    Reads:
        ``state["repaired_code"]`` or ``state["student_code"]``,
        ``state["test_cases"]``, ``state["entry_point"]``

    Writes:
        ``execution_result``, ``success``, ``attempt``
    """
    logger = logger or get_logger()
    logger.info("[bold yellow]▶ Execution Node[/bold yellow]")

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

    result = await evaluator.evaluate(
        code=code,
        test_code=test_code,
        entry_point=entry_point,
    )

    success = result.is_success

    logger.info(
        f"Execution: {'PASS' if success else 'FAIL'} "
        f"({result.passed}/{result.total} tests)"
    )

    log_metrics({
        "tests_passed": result.passed,
        "tests_failed": result.failed,
        "tests_total": result.total,
        "execution_time_ms": result.execution_time_ms,
    })

    log_event(
        "execution_completed",
        task_id=state.get("task_id", ""),
        success=success,
        attempt=state.get("attempt", 0) + (0 if success else 1),
        passed=result.passed,
        total=result.total,
    )

    return {
        "execution_result": result.model_dump(),
        "success": success,
        "attempt": state.get("attempt", 0) + (0 if success else 1),
    }


async def failure_analysis_node(
    state: AFDADState,
    truncation_limits: dict[str, int] | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Analyse the failure — construct structured failure description.

    Builds a text representation combining the task, code, error type,
    stderr, and traceback for downstream embedding and clustering.

    Reads:
        ``state["task"]``, ``state["student_code"]`` or
        ``state["repaired_code"]``, ``state["execution_result"]``

    Writes:
        ``failure_description``
    """
    logger = logger or get_logger()
    logger.info("[bold red]▶ Failure Analysis Node[/bold red]")

    task = state["task"]
    code = state.get("repaired_code") or state["student_code"]
    exec_result = ExecutionResult(**state["execution_result"])

    limits = truncation_limits or {}
    failure_text = FailureEncoder.build_failure_text(
        task=task,
        code=code,
        error_type=exec_result.error_type,
        stderr=exec_result.stderr,
        traceback=exec_result.traceback,
        task_limit=limits.get("task", 200),
        traceback_limit=limits.get("traceback", 500),
        stderr_limit=limits.get("stderr", 300),
        code_limit=limits.get("code", 300),
    )

    logger.info(
        f"Failure analysed: {exec_result.error_type or 'Unknown'}"
    )

    log_event(
        "failure_analysed",
        task_id=state.get("task_id", ""),
        error_type=exec_result.error_type or "Unknown",
    )

    return {"failure_description": failure_text}


async def failure_embed_node(
    state: AFDADState,
    encoder: FailureEncoder,
    logger: Any = None,
) -> dict[str, Any]:
    """Generate embedding for the failure description.

    Reads:
        ``state["failure_description"]``

    Writes:
        ``failure_embedding``
    """
    logger = logger or get_logger()
    logger.info("[bold magenta]▶ Failure Embedding Node[/bold magenta]")

    failure_text = state["failure_description"]
    embedding = encoder.encode(failure_text)

    log_event(
        "failure_embedded",
        task_id=state.get("task_id", ""),
        description_len=len(failure_text),
    )

    return {"failure_embedding": embedding.tolist()}


async def failure_cluster_node(
    state: AFDADState,
    clustering: FailureClustering,
    memory: FailureMemory,
    logger: Any = None,
) -> dict[str, Any]:
    """Assign the failure to a cluster and store in memory.

    Reads:
        ``state["failure_embedding"]``, ``state["execution_result"]``,
        ``state["task"]``, ``state["student_code"]``

    Writes:
        ``failure_cluster``

    Side Effects:
        Stores the failure record in the failure memory database.
        Updates clustering centroids via partial_fit.
    """
    logger = logger or get_logger()
    logger.info("[bold blue]▶ Failure Clustering Node[/bold blue]")

    embedding = np.array(state["failure_embedding"], dtype=np.float32)
    cluster_name = clustering.predict(embedding)

    # Store in failure memory
    exec_result = ExecutionResult(**state["execution_result"])
    record = FailureRecord(
        task=state["task"],
        code=state.get("repaired_code") or state["student_code"],
        traceback=exec_result.traceback,
        failure_cluster=FailureCluster.from_string(cluster_name),
        embedding=state["failure_embedding"],
    )
    memory.store(record)

    # Update clustering with new embedding
    clustering.partial_fit(embedding.reshape(1, -1))

    logger.info(f"Failure cluster: [bold]{cluster_name}[/bold]")

    log_metrics({"failure_cluster": cluster_name})

    log_event(
        "failure_clustered",
        task_id=state.get("task_id", ""),
        cluster=cluster_name,
    )

    return {"failure_cluster": cluster_name}


async def expert_repair_node(
    state: AFDADState,
    debugger: DebuggerAgent,
    critic: CriticAgent,
    collector: TrajectoryCollector,
    logger: Any = None,
) -> dict[str, Any]:
    """Run the agentic repair workflow: Debugger → Critic.

    The expert repair team analyses the failure, generates a fix,
    and validates it. The full trajectory is captured for distillation.

    Reads:
        ``state["task"]``, ``state["plan"]``,
        ``state["student_code"]`` or ``state["repaired_code"]``,
        ``state["execution_result"]``

    Writes:
        ``repaired_code``, ``repair_trace``, ``repair_verified``,
        ``expert_called``, ``trajectory_id``

    Side Effects:
        Saves the repair trajectory to disk.
    """
    logger = logger or get_logger()
    logger.info(
        "[bold red]▶ Expert Repair Node[/bold red] "
        "(Debugger → Critic)"
    )

    task = state["task"]
    plan = state.get("plan", "")
    failed_code = state.get("repaired_code") or state["student_code"]
    exec_result = ExecutionResult(**state["execution_result"])

    # Create trajectory
    trajectory = collector.create_trajectory()

    # ── Step 1: Debugger ──
    logger.info("  └─ Running Debugger agent...")
    debug_result = await debugger.generate(
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

    trajectory = collector.add_step(
        trajectory,
        agent="Debugger",
        action="analyse_and_repair",
        reasoning=analysis.detailed_explanation,
        output=repaired_code,
    )

    # ── Step 2: Critic ──
    logger.info("  └─ Running Critic agent...")
    critic_result = await critic.generate(
        task=task,
        failed_code=failed_code,
        error_type=exec_result.error_type or "Unknown",
        stderr=exec_result.stderr,
        repaired_code=repaired_code,
        repair_reasoning=analysis.detailed_explanation,
    )

    trajectory = collector.add_step(
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

    trajectory = collector.finalise(
        trajectory,
        repaired_code=repaired_code,
        repair_reasoning=repair_reasoning,
        verified=critic_result.get("is_valid", False),
    )

    # Save trajectory
    collector.save_trajectory(trajectory)

    logger.info(
        f"Repair {'verified' if critic_result.get('is_valid') else 'unverified'}"
    )

    log_metrics({
        "expert_called": 1,
        "repair_verified": int(critic_result.get("is_valid", False)),
    })

    log_event(
        "expert_repair_completed",
        task_id=state.get("task_id", ""),
        trajectory_id=trajectory.trajectory_id,
        verified=critic_result.get("is_valid", False),
    )

    return {
        "repaired_code": repaired_code,
        "repair_trace": trajectory.model_dump(),
        "repair_verified": critic_result.get("is_valid", False),
        "expert_called": True,
        "trajectory_id": trajectory.trajectory_id,
    }


async def distillation_node(
    state: AFDADState,
    collector: TrajectoryCollector,
    logger: Any = None,
) -> dict[str, Any]:
    """Generate a training example from the repair trajectory.

    Reads:
        ``state["repair_trace"]``, ``state["task"]``,
        ``state["plan"]``, ``state["student_code"]``,
        ``state["failure_cluster"]``

    Writes:
        ``training_example``
    """
    logger = logger or get_logger()
    logger.info("[bold white]▶ Distillation Node[/bold white]")

    from afdad.utils.models import RepairTrajectory

    repair_trace = state.get("repair_trace", {})
    trajectory = RepairTrajectory(**repair_trace)

    training_example = collector.to_training_example(
        trajectory=trajectory,
        task=state["task"],
        plan=state.get("plan", ""),
        failed_code=state["student_code"],
        failure_cluster=state.get("failure_cluster", "Unknown"),
        source_task_id=state.get("task_id"),
    )

    logger.info(
        f"Training example generated (cluster: {training_example.failure_cluster.value})"
    )

    log_event(
        "distillation_example_generated",
        task_id=state.get("task_id", ""),
        cluster=training_example.failure_cluster.value,
    )

    return {"training_example": training_example.model_dump()}


async def success_store_node(
    state: AFDADState,
    logger: Any = None,
) -> dict[str, Any]:
    """Store stats for a successful task (no repair needed).

    Reads:
        ``state["expert_called"]``

    Writes:
        Nothing — terminal node.
    """
    logger = logger or get_logger()
    logger.info("[bold green]▶ Success — storing stats[/bold green]")

    log_metrics({
        "task_success": 1,
        "expert_called": int(state.get("expert_called", False)),
    })

    log_event(
        "task_success_stored",
        task_id=state.get("task_id", ""),
        expert_called=state.get("expert_called", False),
    )

    return {}


# ──────────────────────────────────────────────────────────
# Dependency Constructor / Factory
# ──────────────────────────────────────────────────────────

def create_nodes(cfg: DictConfig) -> dict[str, NodeFn]:
    """Construct all dependencies and return a dictionary of bound node functions.

    Args:
        cfg: Full Hydra configuration.

    Returns:
        Mapping of node names to async callable functions.
    """
    logger = get_logger()

    # ── Dependency Initialization ──
    planner = PlannerAgent(cfg.model.student)
    coder = CoderAgent(cfg.model.student)
    debugger = DebuggerAgent(cfg.model.expert)
    critic = CriticAgent(cfg.model.expert)
    evaluator = Evaluator(cfg.execution)
    encoder = FailureEncoder(cfg.failure_memory)
    clustering = FailureClustering(cfg.clustering, encoder)
    memory = FailureMemory(cfg.failure_memory)
    collector = TrajectoryCollector(cfg.trajectories)

    # ── Return Bound Functions ──
    return {
        "plan_node": lambda state: plan_node(state, planner, logger),
        "student_generate_node": lambda state: student_generate_node(state, coder, logger),
        "execute_node": lambda state: execute_node(state, evaluator, logger),
        "failure_analysis_node": lambda state: failure_analysis_node(
            state,
            cfg.get("failure_memory", {}).get("text_truncation", {}),
            logger,
        ),
        "failure_embed_node": lambda state: failure_embed_node(state, encoder, logger),
        "failure_cluster_node": lambda state: failure_cluster_node(state, clustering, memory, logger),
        "expert_repair_node": lambda state: expert_repair_node(state, debugger, critic, collector, logger),
        "distillation_node": lambda state: distillation_node(state, collector, logger),
        "success_store_node": lambda state: success_store_node(state, logger),
    }


# Deprecated class wrapper for backward compatibility if needed
class NodeFactory:
    """Wrapper class for backward compatibility with NodeFactory calls.

    Instantiates create_nodes internally and delegates attributes.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._nodes = create_nodes(cfg)

    @property
    def plan_node(self) -> NodeFn:
        return self._nodes["plan_node"]

    @property
    def student_generate_node(self) -> NodeFn:
        return self._nodes["student_generate_node"]

    @property
    def execute_node(self) -> NodeFn:
        return self._nodes["execute_node"]

    @property
    def failure_analysis_node(self) -> NodeFn:
        return self._nodes["failure_analysis_node"]

    @property
    def failure_embed_node(self) -> NodeFn:
        return self._nodes["failure_embed_node"]

    @property
    def failure_cluster_node(self) -> NodeFn:
        return self._nodes["failure_cluster_node"]

    @property
    def expert_repair_node(self) -> NodeFn:
        return self._nodes["expert_repair_node"]

    @property
    def distillation_node(self) -> NodeFn:
        return self._nodes["distillation_node"]

    @property
    def success_store_node(self) -> NodeFn:
        return self._nodes["success_store_node"]
