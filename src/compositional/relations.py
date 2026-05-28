"""Construct joint cliques from local cliques and cross-component coupling sets.

Each component owns a local question set with its own logical relations. The
joint clique additionally respects coupling constraints from C: shared-question
identifications (equal), partitions spanning components, and cross-component
logical relations (e.g., conjunctions across components).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from jcd.types import Clique, Relation


def make_joint_clique(
    local_cliques: Iterable[Clique],
    coupling_relations: Iterable[Relation] | None = None,
) -> Clique:
    """Build a joint Clique by lifting local relations to a shared coordinate space.

    Parameters
    ----------
    local_cliques : iterable of Clique
        Per-component local cliques. Each ``local_cliques[a].m`` is the local
        coordinate count.
    coupling_relations : iterable of Relation, optional
        Cross-component coupling relations whose indices reference the joint
        coordinate space (i.e., already lifted by the caller). If None, the
        joint clique reduces to the Cartesian product of local cliques.

    Returns
    -------
    Clique
        Joint clique whose feasible set is M^* ⊆ [0,1]^{m^*} with m^* = sum of
        local m_a.

    Notes
    -----
    The local index ``j`` of component ``a`` lifts to joint index
    ``offsets[a] + j`` where ``offsets[a]`` is the cumulative sum of prior
    component sizes.
    """
    locals_list = list(local_cliques)
    offsets: list[int] = []
    cum = 0
    for c in locals_list:
        offsets.append(cum)
        cum += c.m
    m_star = cum

    rels: list[Relation] = []
    for off, c in zip(offsets, locals_list):
        for r in c.relations:
            rels.append(Relation(type=r.type, indices=tuple(i + off for i in r.indices)))

    if coupling_relations is not None:
        rels.extend(coupling_relations)

    return Clique(m=m_star, relations=rels)


def lift_indices(local_indices: Iterable[int], component_offset: int) -> tuple[int, ...]:
    """Lift component-local indices into the joint coordinate space."""
    return tuple(i + component_offset for i in local_indices)
