# Adaptive Failure-Driven Agentic Distillation (AFDAD)

Official implementation of **"The Student Becomes the Master: A Closed-Loop, Failure-Aware Framework for Cost-Efficient Code Generation"** (Adaptive Failure-Driven Agentic Distillation).

AFDAD is a framework designed to elevate the code-generation performance of Small Language Models (SLMs) to match or exceed frontier models by leveraging an execution-grounded, closed-loop training pipeline.

---

## 🚀 Key Features

- **Student-First Code Generation**: Fast, cost-efficient initial generation using a student model.
- **Agentic Execution & Validation**: Automated subprocess execution of candidate solutions against unit tests.
- **Failure-Aware Memory**: Persists failures and traces in a SQLite database to capture learning history.
- **Failure Clustering**: Groups failures into semantic categories (Syntax, Runtime, Logic, Edge Cases, Algorithm Design, Efficiency) using spherical K-means on failure text embeddings.
- **Expert-Guided Repair**: Invokes a collaborative Debugger and Critic multi-agent team (backed by a frontier expert model) to identify root causes, produce verified corrections, and log structured repair steps.
- **Adaptive Curriculum Distillation**: Builds a priority replay buffer that oversamples training data from high-failure clusters, ensuring the student model focuses training epochs on its weakest areas.
- **Ablation Controls**: Fully configurable workflow topology for rigorous empirical ablation studies.

---

## 🛠️ Architecture Overview

The AFDAD framework is built as a cyclic, stateful graph using **LangGraph**:

```
                       [ plan_node ]
                             │
                             ▼
                 [ student_generate_node ]
                             │
                             ▼
                       [ execute_node ]
                        /            \
             (Success) /              \ (Failure)
                      v                v
             [ success_store ]   [ failure_analysis ]
                      │                │
                      ▼                ▼
                    (END)       [ failure_embed ]
                                       │
                                       ▼
                                [ failure_cluster ]
                                       │
                                       ▼
                                [ expert_repair ]
                                /               \
                    (Verified) /                 \ (Max attempts)
                              v                   v
                        [ re_execute ]     [ distillation ]
                         /          \             │
                  (Pass)/            \(Fail)      ▼
                       v              v         (END)
                [ distillation ]  (To Repair Loop)
                       │
                       ▼
                     (END)
```

---

## 📦 Installation

This repository uses Python 3.12+ and is packaged using `pyproject.toml` (PEP 517). We recommend using `uv` or `pip` to manage dependencies.

### Installation via pip

```bash
# Clone the repository
git clone https://github.com/sakethksg/Atlas-code.git
cd Atlas-code

# Install package with dependencies
pip install -e .
```

### Installation for Developers (includes pytest, mypy, and ruff)

```bash
pip install -e ".[dev,eval]"
```

---

## ⚙️ Configuration & Ablation Presets

AFDAD utilizes **Hydra** for configuration management. Central configurations reside in `afdad/configs/config.yaml`.

Ablations are controlled via flags in the experiment configuration:
- `use_agentic_repair`: Activates Debugger & Critic team repair.
- `use_distillation`: Generates training data from trajectories.
- `use_failure_clustering`: Embeds and groups failures for cluster tracking.
- `use_adaptive_curriculum`: Weight replay buffer sampling based on failure rates.

### Preset Ablation Configurations

| Run Presets | Configuration Name | `use_agentic_repair` | `use_distillation` | `use_failure_clustering` | `use_adaptive_curriculum` | Description |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **Run 0** | Baseline Student | `false` | `false` | `false` | `false` | Baseline SLM without any enhancement |
| **Run 1** | Repair Only | `true` | `false` | `false` | `false` | Student with expert repair team, no self-training |
| **Run 2** | Uniform Distill | `true` | `true` | `false` | `false` | Distillation using uniform trajectory sampling |
| **Run 3** | Cluster Distill | `false` | `true` | `true` | `true` | Distillation without repair, using curriculum weighting |
| **Run 4** | Full AFDAD | `true` | `true` | `true` | `true` | Full closed-loop adaptive pipeline (ours) |

---

## 🏃 Running the Pipeline

### Running the Full Training Loop
To start the end-to-end distillation loop:
```bash
python -m afdad.main mode=train
```

### Running a Standalone Single Task
Run a single prompt through the active compiled graph and inspect results:
```bash
python -m afdad.main mode=single_task task="Write a Python function to check if a number is prime"
```

### Standalone Benchmark Evaluation
Evaluate the current student model on benchmark tests:
```bash
python -m afdad.main mode=evaluate evaluation.benchmarks=\[humaneval,mbpp\]
```

### Visualizing the LangGraph Topology
To render the active graph layout matching the loaded config:
```bash
python -m afdad.main mode=visualise output_path=afdad_graph.png
```

---

## 🧪 Testing

To run unit and integration tests:
```bash
pytest tests/ -v
```

---

## 📄 Citation

If you use this repository or work in your research, please cite our paper:

```bibtex
@inproceedings{ksg2026afdad,
  title={The Student Becomes the Master: A Closed-Loop, Failure-Aware Framework for Cost-Efficient Code Generation},
  author={KSG, Saketh and others},
  booktitle={arXiv preprint arXiv:2606.xxxxx},
  year={2026}
}
```
