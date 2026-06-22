"""Failure analysis — embedding, clustering, and memory persistence.

Implements the failure pipeline core: failures are encoded into dense
embeddings, clustered into predefined categories (Syntax, Runtime, Logic,
EdgeCases, AlgorithmDesign, Efficiency), and persisted in a SQLite database
for retrieval and adaptive curriculum construction.
"""

from afdad.failures.clustering import FailureClustering
from afdad.failures.encoder import FailureEncoder
from afdad.failures.memory import FailureMemory

__all__ = ["FailureClustering", "FailureEncoder", "FailureMemory"]
