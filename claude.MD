# CLAUDE.md

# Adaptive Failure-Driven Agentic Distillation (AFDAD)

## Research Objective

Build a complete research framework that improves the code-generation capabilities of Small Language Models (SLMs) through execution-grounded agentic learning.

The framework should enable an SLM to:

* Attempt coding tasks independently.
* Learn from execution failures.
* Receive agent-generated repair trajectories.
* Distill debugging knowledge.
* Gradually reduce dependence on expert models.

The goal is not merely solving coding tasks but creating a self-improving learning ecosystem.

---

# Core Research Hypothesis

Current distillation pipelines are passive.

Teacher → Student

The student receives solutions but does not actively influence future learning.

AFDAD introduces:

Student → Failure → Agentic Repair → Distillation → Improved Student

The student learns directly from its own mistakes.

---

# High-Level Architecture

```text
                ┌─────────────────┐
                │ Coding Problem  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Student SLM     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Execution Node  │
                └────────┬────────┘
                         │
               ┌─────────┴─────────┐
               │                   │
            Success             Failure
               │                   │
               ▼                   ▼
          Store Stats     Failure Analysis
                                   │
                                   ▼
                         Failure Clustering
                                   │
                                   ▼
                          Agentic Repair Team
                                   │
                                   ▼
                           Repair Trajectory
                                   │
                                   ▼
                              Distillation
                                   │
                                   ▼
                            Student Update
```

---

# Technology Stack

## Agent Orchestration

Mandatory:

LangGraph

Use StateGraph.

All components must be implemented as graph nodes.

Do NOT use linear LangChain chains.

The system is cyclic and stateful.

---

## Inference Backend

Use vLLM OpenAI-compatible API.

Expert server:

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

python -m vllm.entrypoints.openai.api_server \
  --model MODEL_PATH \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 32768 \
  --dtype auto
```

Client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
```

---

# Recommended Models

## Student

Primary:

Qwen2.5-Coder-1.5B-Instruct

Secondary:

Qwen2.5-Coder-7B-Instruct

Optional:

DeepSeek-Coder-1.3B

---

## Expert

Primary:

Qwen2.5-Coder-32B-Instruct

Alternative:

DeepSeek-V3

All expert interactions occur through vLLM API.

---

# Repository Structure

```text
afdad/

├── configs/

├── agents/
│   ├── planner.py
│   ├── coder.py
│   ├── debugger.py
│   ├── critic.py

├── graph/
│   ├── state.py
│   ├── workflow.py
│   ├── nodes.py
│   ├── routing.py

├── execution/
│   ├── sandbox.py
│   ├── evaluator.py

├── failures/
│   ├── encoder.py
│   ├── clustering.py
│   ├── memory.py

├── trajectories/
│   ├── collector.py
│   ├── replay_buffer.py

├── distillation/
│   ├── dataset_builder.py
│   ├── trainer.py
│   ├── lora_train.py

├── training/
│   ├── train_student.py

├── evaluation/
│   ├── humaneval.py
│   ├── mbpp.py

├── prompts/

└── main.py
```

---

# LangGraph State

Create a single shared graph state.

```python
class AFDADState(TypedDict):

    task: str

    plan: str

    student_code: str

    repaired_code: str

    execution_result: dict

    failure_cluster: str

    repair_trace: dict

    success: bool

    expert_called: bool

    trajectory_id: str
```

---

# LangGraph Nodes

## Planner Node

Purpose:

Understand problem.

Generate algorithmic plan.

Output:

* algorithm strategy
* complexity estimate
* edge cases

---

## Student Generation Node

Input:

task
plan

Output:

student_code

Student always attempts first.

No expert involvement.

---

## Execution Node

Execute generated code.

Run official tests.

Return:

```python
passed
failed
stderr
stdout
```

---

## Failure Analysis Node

Only triggered when execution fails.

Construct:

problem
+
student code
+
compiler log
+
test differences

Generate structured failure description.

---

## Failure Embedding Node

Generate embedding using:

jina-embeddings-v3

or

bge-large-en

Store vector representation.

---

## Failure Clustering Node

Cluster failures.

Recommended clusters:

* Syntax
* Runtime
* Logic
* Edge Cases
* Algorithm Design
* Efficiency

Use:

Spherical K-Means

Persist centroids.

---

## Expert Repair Node

Agentic repair workflow.

Sub-agents:

Planner

Debugger

Critic

Workflow:

Failure
→ Analysis
→ Repair
→ Verification

Output:

repair trajectory

---

## Distillation Node

Generate training samples.

Format:

{
problem,
plan,
failed_code,
repair_reasoning,
repaired_code
}

Add to replay buffer.

---

# Agent Design

## Planner Agent

Responsibilities:

* task decomposition
* edge case discovery
* algorithm selection

---

## Debugger Agent

Responsibilities:

* identify root cause
* explain failure
* propose fixes

---

## Critic Agent

Responsibilities:

* validate repair
* detect hallucinated changes
* ensure test consistency

---

# Failure Memory

Persist every failure.

Store:

```python
{
    task,
    code,
    traceback,
    failure_cluster,
    embedding,
    repair_trace
}
```

Use SQLite initially.

Upgrade later to PostgreSQL.

---

# Adaptive Curriculum

Core novelty.

Calculate:

cluster_failure_rate

Example:

Syntax = 12%

Logic = 45%

Edge Cases = 30%

Training sampler prioritizes:

highest failure clusters

Student focuses on weaknesses.

---

# Distillation Strategy

Distill:

NOT

problem → solution

Instead:

problem
+
failure
+
repair reasoning
+
corrected solution

This teaches debugging skills.

---

# LoRA Training

Use:

PEFT

QLoRA

Target modules:

q_proj
k_proj
v_proj
o_proj

Rank:

16

Alpha:

32

---

# Datasets

Phase 1

HumanEval

MBPP

Phase 2

HumanEval-ET

MBPP-ET

Phase 3

LiveCodeBench

---

# Metrics

Primary:

Pass@1

Pass@5

Repair Success Rate

Failure Reduction Rate

Student Improvement Rate

---

Secondary:

Expert Calls

Average Cost

Latency

Cluster Accuracy

---

# Ablation Studies

Run 0

Baseline Student

---

Run 1

Student + Agentic Repair

---

Run 2

Student + Agentic Repair + Distillation

---

Run 3

Student + Distillation + Failure Clustering

---

Run 4

Full AFDAD

---

# Success Criteria

The project is successful if:

1. Student Pass@1 improves significantly.

2. Failure clusters become more balanced over time.

3. Repair trajectories outperform direct solution distillation.

4. Expert dependence decreases as training progresses.

5. Full AFDAD outperforms standard fine-tuning and standard distillation baselines.

---

# Engineering Principles

* Strong typing.
* Pydantic models everywhere.
* Async API calls.
* Reproducible experiments.
* Hydra configs.
* Weights & Biases logging.
* Modular design.
* Every LangGraph node independently testable.

The codebase should prioritize research reproducibility and publication-quality experimentation.
