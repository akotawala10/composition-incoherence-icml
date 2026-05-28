"""JCD QP module: L2 projection onto coherent-marginal polytopes."""

from .closed_form import (
    project_and,
    project_equal,
    project_implies,
    project_mutex,
    project_neg,
    project_or,
    project_partition,
)
from .constraints import compile_constraints, compile_equalities
from .solver import kkt_residual, project

__all__ = [
    "compile_constraints",
    "compile_equalities",
    "kkt_residual",
    "project",
    "project_and",
    "project_equal",
    "project_implies",
    "project_mutex",
    "project_neg",
    "project_or",
    "project_partition",
]
