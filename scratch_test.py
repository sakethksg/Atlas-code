from omegaconf import OmegaConf
import sys

# Disable AsyncOpenAI.__del__ to prevent C-level access violations during process shutdown on Windows
try:
    from openai import AsyncOpenAI
    AsyncOpenAI.__del__ = lambda self: None
except ImportError:
    pass

from afdad.graph.workflow import _get_experiment_flags
from afdad.graph.nodes import create_nodes
from langgraph.graph import END, StateGraph
from afdad.graph.state import AFDADState
from afdad.utils.logging import get_logger
from afdad.graph.routing import (
    route_after_execution,
    route_after_re_execution,
    route_after_repair,
)

cfg = OmegaConf.create({
    "seed": 42,
    "output_dir": "./outputs",
    "data_dir": "./data",
    "execution": {
        "timeout_seconds": 30,
        "temp_dir": "./outputs/exec_tmp",
        "max_concurrent_evals": 8,
    },
    "failure_memory": {
        "db_path": "./outputs/failure_memory.db",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
        "embedding_batch_size": 32,
        "similarity_top_k": 5,
        "text_truncation": {
            "task": 200,
            "traceback": 500,
            "stderr": 300,
            "code": 300,
        },
    },
    "clustering": {
        "n_clusters": 6,
        "cluster_names": ["Syntax", "Runtime", "Logic", "EdgeCases", "AlgorithmDesign", "Efficiency"],
        "centroids_path": "./outputs/centroids.npy",
        "min_samples_for_fit": 20,
        "ema_decay": 0.1,
        "seed": 42,
    },
    "trajectories": {
        "buffer_capacity": 10000,
        "save_dir": "./outputs/trajectories",
        "curriculum_amplification": 5.0,
    },
    "distillation": {
        "dataset_dir": "./outputs/distillation_dataset",
        "max_seq_length": 4096,
    },
    "evaluation": {
        "benchmarks": ["humaneval", "mbpp"],
        "num_samples_per_task": 5,
        "temperature": 0.8,
    },
    "pipeline": {
        "max_repair_attempts": 3,
        "max_tasks_per_round": 164,
        "num_training_rounds": 5,
    },
    "logging": {
        "use_wandb": False,
        "wandb_project": "afdad",
        "wandb_entity": None,
        "log_level": "INFO",
    },
    "model": {
        "student": {
            "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "base_url": "http://localhost:8001/v1",
            "api_key": "EMPTY",
            "generation": {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.95,
                "stop_sequences": ["\n```\n", "\nclass ", "\ndef "],
            },
        },
        "expert": {
            "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "generation": {
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.95,
                "stop_sequences": [],
            },
        },
    },
    "experiment": {
        "experiment": {
            "name": "full_afdad",
            "use_agentic_repair": True,
            "use_distillation": True,
            "use_failure_clustering": True,
            "use_adaptive_curriculum": True,
        }
    },
})

print("1. Getting logger...")
logger = get_logger()

print("2. Getting flags...")
flags = _get_experiment_flags(cfg)
print("   Flags:", flags)

print("3. Creating nodes...")
nodes = create_nodes(cfg)
print("   Nodes created successfully!")

print("4. Instantiating StateGraph...")
graph = StateGraph(AFDADState)
print("   StateGraph instantiated!")

print("5. Adding nodes...")
graph.add_node("plan", nodes["plan_node"])
graph.add_node("student_generate", nodes["student_generate_node"])
graph.add_node("execute", nodes["execute_node"])
graph.add_node("success_store", nodes["success_store_node"])

print("6. Setting entry point...")
graph.set_entry_point("plan")

print("7. Adding linear edges...")
graph.add_edge("plan", "student_generate")
graph.add_edge("student_generate", "execute")
graph.add_edge("success_store", END)

print("8. Adding failure nodes...")
graph.add_node("failure_analysis", nodes["failure_analysis_node"])
graph.add_node("expert_repair", nodes["expert_repair_node"])

if flags["use_failure_clustering"]:
    print("   Adding failure_embed & failure_cluster nodes...")
    graph.add_node("failure_embed", nodes["failure_embed_node"])
    graph.add_node("failure_cluster", nodes["failure_cluster_node"])
    graph.add_edge("failure_analysis", "failure_embed")
    graph.add_edge("failure_embed", "failure_cluster")
    graph.add_edge("failure_cluster", "expert_repair")
else:
    graph.add_edge("failure_analysis", "expert_repair")

print("9. Adding conditional edges...")
graph.add_conditional_edges(
    "execute",
    route_after_execution,
    {
        "success_store": "success_store",
        "failure_analysis": "failure_analysis",
    },
)

if flags["use_distillation"]:
    print("   Adding distillation edges...")
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

print("10. Compiling graph...")
compiled = graph.compile()
print("11. Compiled successfully!")
