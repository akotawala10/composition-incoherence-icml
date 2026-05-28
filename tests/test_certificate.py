"""Tests for compositional.certificate.compositional_residual."""

from __future__ import annotations

import numpy as np

from compositional import compositional_residual, make_joint_clique
from jcd.types import Clique, Relation


def test_residual_zero_on_coherent_partition() -> None:
    """A composed quote already on the partition simplex has ε* ≈ 0."""
    joint = Clique(m=3, relations=[Relation(type="partition", indices=(0, 1, 2))])
    q = np.array([0.2, 0.3, 0.5])
    eps = compositional_residual(q, joint)
    assert eps < 1e-9, f"expected ε* ≈ 0 on coherent partition, got {eps}"


def test_residual_positive_on_incoherent_partition() -> None:
    """A composed quote with sum > 1 has ε* > 0."""
    joint = Clique(m=4, relations=[Relation(type="partition", indices=(0, 1, 2, 3))])
    q = np.array([0.6, 0.6, 0.6, 0.7])  # sums to 2.5
    eps = compositional_residual(q, joint)
    assert eps > 0.5, f"expected ε* > 0.5 for sum-2.5 partition, got {eps}"


def test_residual_zero_on_coherent_negation() -> None:
    """A composed (p, 1-p) pair has ε* = 0 on a negation clique."""
    joint = Clique(m=2, relations=[Relation(type="neg", indices=(0, 1))])
    q = np.array([0.3, 0.7])
    eps = compositional_residual(q, joint)
    assert eps < 1e-9


def test_residual_positive_on_incoherent_negation() -> None:
    """A composed (P(A), P(¬A)) summing past 1 has ε* > 0."""
    joint = Clique(m=2, relations=[Relation(type="neg", indices=(0, 1))])
    q = np.array([0.7, 0.6])  # sums to 1.3
    eps = compositional_residual(q, joint)
    assert eps > 0.0, f"expected ε* > 0 for sum-1.3 negation, got {eps}"


def test_residual_invariant_to_uniform_shift() -> None:
    """ε* depends only on the residual to the polytope, not the absolute value."""
    joint = Clique(m=2, relations=[Relation(type="neg", indices=(0, 1))])
    eps1 = compositional_residual(np.array([0.4, 0.4]), joint)
    eps2 = compositional_residual(np.array([0.5, 0.5]), joint)
    # Both have sum 0.8 and 1.0; first has nonzero ε*, second has zero.
    assert eps1 > 0.0
    assert eps2 < 1e-9


def test_make_joint_clique_partition_across_components() -> None:
    """Joint clique correctly lifts a partition spanning two components."""
    # Component 0 owns coordinates 0, 1; component 1 owns coordinates 2, 3.
    local0 = Clique(m=2, relations=[])
    local1 = Clique(m=2, relations=[])
    coupling = [Relation(type="partition", indices=(0, 1, 2, 3))]
    joint = make_joint_clique([local0, local1], coupling)
    assert joint.m == 4
    assert joint.relations[0].type == "partition"
    assert joint.relations[0].indices == (0, 1, 2, 3)
