"""The compositional residual ε* — output-only runtime certificate.

ε* is the L2 distance from a composed quote q to the joint coherent polytope
M^*. It is computable from agent outputs and the cross-component coupling set
alone, without component internals.
"""

from __future__ import annotations

import numpy as np

from jcd.qp.solver import project as l2_project_onto_polytope
from jcd.types import Clique


def compositional_residual(composed_quote: np.ndarray, joint_clique: Clique) -> float:
    """Compute ε*(q) = ||q - Π*(q)||_2.

    Parameters
    ----------
    composed_quote : np.ndarray of shape (m_star,)
        The agent's joint quote, assembled by the aggregator.
    joint_clique : Clique
        Joint clique encoding local relations + cross-component coupling set.

    Returns
    -------
    float
        The compositional residual ε*. Zero iff the composed quote lies in the
        joint polytope; otherwise certifies a Dutch-book exposure on the
        assembled belief.
    """
    q = np.asarray(composed_quote, dtype=float).ravel()
    if q.shape != (joint_clique.m,):
        raise ValueError(
            f"composed_quote shape {q.shape} does not match joint clique m={joint_clique.m}"
        )
    projected = l2_project_onto_polytope(joint_clique, q)
    return float(np.linalg.norm(q - projected))
