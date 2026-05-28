"""Aggregator types for assembling component outputs into a joint quote.

The dichotomy theorem of the paper is stated under owner-selected coordinate
aggregation: each joint coordinate is the unique output of one component.
This file provides the owner-selected aggregator and helpers for the routing
patterns evaluated in the paper (planner-to-specialist, disjoint-tool
composition, sharded retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OwnerSelectedAggregator:
    """Owner-selected coordinate aggregator.

    Each joint coordinate is owned by exactly one component. The aggregator
    selects ``component_outputs[owners[j]][local_index_in_owner(j)]`` as the
    quote for joint coordinate ``j``.

    Attributes
    ----------
    owners : list[int]
        ``owners[j]`` is the component index that owns joint coordinate ``j``.
    local_index : list[int]
        ``local_index[j]`` is the within-component coordinate that joint
        coordinate ``j`` corresponds to in the owner's local marginal.
    """

    owners: tuple[int, ...]
    local_index: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.owners) != len(self.local_index):
            raise ValueError("owners and local_index must have the same length.")

    @property
    def m_star(self) -> int:
        return len(self.owners)

    def assemble(self, component_outputs: list[np.ndarray]) -> np.ndarray:
        """Assemble per-component outputs into a joint quote.

        Parameters
        ----------
        component_outputs : list of arrays
            ``component_outputs[a]`` is the local marginal for component ``a``.

        Returns
        -------
        np.ndarray of shape (m_star,)
            The joint quote ``q`` with ``q[j] = component_outputs[owners[j]][local_index[j]]``.
        """
        out = np.empty(self.m_star, dtype=float)
        for j, (a, k) in enumerate(zip(self.owners, self.local_index)):
            out[j] = float(component_outputs[a][k])
        return out
