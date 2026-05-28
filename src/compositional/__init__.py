"""Compositional incoherence: certificate, repair, and aggregator types.

Public API
----------
- compositional_residual:    compute ε* from a composed quote and joint clique.
- hierarchical_project:      hierarchical Boyle--Dykstra repair.
- OwnerSelectedAggregator:   owner-selected coordinate aggregator.
- make_joint_clique:         build a joint Clique from local cliques + coupling set.
"""

from .certificate import compositional_residual
from .projection import hierarchical_project
from .aggregator import OwnerSelectedAggregator
from .relations import make_joint_clique

__all__ = [
    "compositional_residual",
    "hierarchical_project",
    "OwnerSelectedAggregator",
    "make_joint_clique",
]
