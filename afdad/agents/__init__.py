"""Agent modules for AFDAD — Planner, Coder, Debugger, Critic.

Each agent inherits from :class:`BaseAgent` and implements a specific role
in the AFDAD pipeline. The Planner and Coder use the student SLM, while
the Debugger and Critic use the expert model for agentic repair.
"""

from afdad.agents.base import BaseAgent
from afdad.agents.coder import CoderAgent
from afdad.agents.critic import CriticAgent
from afdad.agents.debugger import DebuggerAgent
from afdad.agents.planner import PlannerAgent

__all__ = [
    "BaseAgent",
    "CoderAgent",
    "CriticAgent",
    "DebuggerAgent",
    "PlannerAgent",
]
