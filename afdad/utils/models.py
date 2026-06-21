"""
Pydantic data models used across the AFDAD framework.

All structured data flowing through the pipeline is validated
via these models. They enforce strong typing as per the spec.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────


class FailureCluster(str, Enum):
    """Predefined failure cluster categories."""

    SYNTAX = "Syntax"
    RUNTIME = "Runtime"
    LOGIC = "Logic"
    EDGE_CASES = "EdgeCases"
    ALGORITHM_DESIGN = "AlgorithmDesign"
    EFFICIENCY = "Efficiency"
    UNKNOWN = "Unknown"


class TaskStatus(str, Enum):
    """Status of a task in the pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REPAIRED = "repaired"


# ── Execution ─────────────────────────────────────────────────


class ExecutionResult(BaseModel):
    """Result from code execution."""

    passed: int = Field(default=0, description="Number of tests passed")
    failed: int = Field(default=0, description="Number of tests failed")
    total: int = Field(default=0, description="Total number of tests")
    stderr: str = Field(default="", description="Standard error output")
    stdout: str = Field(default="", description="Standard output")
    returncode: int = Field(default=-1, description="Process return code")
    timeout: bool = Field(default=False, description="Whether execution timed out")
    error_type: str | None = Field(
        default=None, description="Type of error (e.g., SyntaxError, RuntimeError)"
    )
    traceback: str = Field(default="", description="Full traceback if available")
    execution_time_ms: float = Field(
        default=0.0, description="Execution time in milliseconds"
    )

    @property
    def is_success(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.passed > 0 and not self.timeout


class TestCase(BaseModel):
    """A single test case for code evaluation."""

    input: str = Field(description="Test input")
    expected_output: str = Field(description="Expected output")
    test_code: str = Field(default="", description="Executable test code")


# ── Planning ──────────────────────────────────────────────────


class PlanOutput(BaseModel):
    """Structured output from the Planner agent."""

    algorithm_strategy: str = Field(
        description="High-level algorithm approach"
    )
    complexity_estimate: str = Field(
        description="Time and space complexity estimate"
    )
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Identified edge cases",
    )
    key_observations: str = Field(
        default="", description="Important observations about the problem"
    )


# ── Failure Analysis ─────────────────────────────────────────


class FailureAnalysis(BaseModel):
    """Structured failure description produced by the analysis node."""

    root_cause: str = Field(description="Identified root cause of the failure")
    error_category: str = Field(description="Broad error category")
    detailed_explanation: str = Field(
        description="Detailed explanation of what went wrong"
    )
    suggested_fix_strategy: str = Field(
        description="Suggested approach to fix the issue"
    )
    relevant_code_section: str = Field(
        default="", description="The specific code section causing the issue"
    )


# ── Repair ────────────────────────────────────────────────────


class RepairStep(BaseModel):
    """A single step in the repair trajectory."""

    agent: str = Field(description="Agent that performed this step")
    action: str = Field(description="Action taken")
    reasoning: str = Field(description="Reasoning behind the action")
    output: str = Field(description="Output of the step")


class RepairTrajectory(BaseModel):
    """Full repair trajectory from the agentic repair team."""

    trajectory_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trajectory identifier",
    )
    steps: list[RepairStep] = Field(
        default_factory=list, description="Sequence of repair steps"
    )
    repair_reasoning: str = Field(
        default="", description="Overall repair reasoning summary"
    )
    repaired_code: str = Field(default="", description="Final repaired code")
    verified: bool = Field(
        default=False, description="Whether the repair was verified by the critic"
    )
    num_attempts: int = Field(default=0, description="Number of repair attempts")


# ── Failure Memory ────────────────────────────────────────────


class FailureRecord(BaseModel):
    """A persisted failure record in the failure memory database."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier",
    )
    task: str = Field(description="Original coding task")
    code: str = Field(description="Failed code")
    traceback: str = Field(default="", description="Error traceback")
    failure_cluster: FailureCluster = Field(
        default=FailureCluster.UNKNOWN,
        description="Assigned failure cluster",
    )
    embedding: list[float] | None = Field(
        default=None, description="Failure embedding vector"
    )
    repair_trace: dict[str, Any] | None = Field(
        default=None, description="Associated repair trajectory"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of failure record creation",
    )


# ── Training ──────────────────────────────────────────────────


class TrainingExample(BaseModel):
    """A single training example for distillation.

    Format follows the spec:
    problem + failure + repair reasoning + corrected solution.
    """

    problem: str = Field(description="Original coding problem")
    plan: str = Field(default="", description="Algorithmic plan")
    failed_code: str = Field(description="Student's failed code")
    repair_reasoning: str = Field(
        description="Expert's reasoning about the failure and fix"
    )
    repaired_code: str = Field(description="Corrected solution")
    failure_cluster: FailureCluster = Field(
        default=FailureCluster.UNKNOWN,
        description="Failure category for curriculum weighting",
    )
    weight: float = Field(
        default=1.0, description="Sampling weight for adaptive curriculum"
    )


# ── Cluster Info ──────────────────────────────────────────────


class ClusterInfo(BaseModel):
    """Statistics about a failure cluster."""

    cluster_name: str = Field(description="Name of the cluster")
    total_failures: int = Field(default=0, description="Total failures in cluster")
    failure_rate: float = Field(
        default=0.0, description="Failure rate for this cluster"
    )
    recent_trend: str = Field(
        default="stable",
        description="Trend: improving, worsening, stable",
    )


# ── Task Result ───────────────────────────────────────────────


class TaskResult(BaseModel):
    """Final result of processing a single task through the pipeline."""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier",
    )
    task: str = Field(description="Original coding task")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current task status"
    )
    student_code: str = Field(default="", description="Student-generated code")
    repaired_code: str = Field(default="", description="Expert-repaired code")
    execution_result: ExecutionResult | None = Field(
        default=None, description="Execution result"
    )
    failure_cluster: FailureCluster | None = Field(
        default=None, description="Assigned failure cluster"
    )
    repair_trajectory: RepairTrajectory | None = Field(
        default=None, description="Repair trajectory if failure occurred"
    )
    expert_called: bool = Field(
        default=False, description="Whether the expert was invoked"
    )
    training_example: TrainingExample | None = Field(
        default=None, description="Generated training example"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp",
    )
