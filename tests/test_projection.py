"""Tests for compositional.projection.hierarchical_project."""

from __future__ import annotations

import numpy as np

from compositional import compositional_residual, hierarchical_project
from jcd.types import Clique, Relation


def test_projection_drives_residual_to_zero() -> None:
    """After hierarchical projection, ε* should be at numerical floor."""
    joint = Clique(m=4, relations=[Relation(type="partition", indices=(0, 1, 2, 3))])
    q = np.array([0.6, 0.6, 0.6, 0.7])
    repaired = hierarchical_project(q, joint)
    eps_after = compositional_residual(repaired, joint)
    assert eps_after < 1e-9, f"residual should be at QP floor, got {eps_after}"


def test_projection_idempotent() -> None:
    """Applying the projection twice returns the same result."""
    joint = Clique(m=4, relations=[Relation(type="partition", indices=(0, 1, 2, 3))])
    q = np.array([0.6, 0.6, 0.6, 0.7])
    once = hierarchical_project(q, joint)
    twice = hierarchical_project(once, joint)
    assert np.allclose(once, twice, atol=1e-9)


def test_projection_preserves_already_coherent() -> None:
    """Projecting an already-coherent vector returns the input."""
    joint = Clique(m=3, relations=[Relation(type="partition", indices=(0, 1, 2))])
    q = np.array([0.2, 0.3, 0.5])
    projected = hierarchical_project(q, joint)
    assert np.allclose(projected, q, atol=1e-9)


def test_projection_partition_sums_to_one() -> None:
    """Projection onto a partition simplex always yields sum-to-one."""
    joint = Clique(m=5, relations=[Relation(type="partition", indices=(0, 1, 2, 3, 4))])
    q = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # sum = 3.5
    projected = hierarchical_project(q, joint)
    assert abs(projected.sum() - 1.0) < 1e-9
    assert (projected >= -1e-12).all()


def test_projection_negation_sums_to_one() -> None:
    """Projection onto a negation clique sets p1 + p2 = 1."""
    joint = Clique(m=2, relations=[Relation(type="neg", indices=(0, 1))])
    q = np.array([0.7, 0.6])
    projected = hierarchical_project(q, joint)
    assert abs(projected.sum() - 1.0) < 1e-9
