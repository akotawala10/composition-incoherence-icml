"""Tests for compositional.aggregator.OwnerSelectedAggregator."""

from __future__ import annotations

import numpy as np
import pytest

from compositional import OwnerSelectedAggregator


def test_owner_selected_assembles_correctly() -> None:
    """Each joint coordinate pulls from the assigned component's local index."""
    agg = OwnerSelectedAggregator(owners=(0, 0, 1, 1), local_index=(0, 1, 0, 1))
    component_outputs = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    assembled = agg.assemble(component_outputs)
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(assembled, expected)


def test_owner_selected_handles_disjoint_specialists() -> None:
    """A planner-routed partition: 4 coordinates, 4 different specialists."""
    agg = OwnerSelectedAggregator(owners=(0, 1, 2, 3), local_index=(0, 0, 0, 0))
    component_outputs = [
        np.array([0.7]), np.array([0.6]), np.array([0.6]), np.array([0.6]),
    ]
    assembled = agg.assemble(component_outputs)
    assert np.allclose(assembled, [0.7, 0.6, 0.6, 0.6])


def test_owner_selected_validates_lengths() -> None:
    """owners and local_index must agree in length."""
    with pytest.raises(ValueError):
        OwnerSelectedAggregator(owners=(0, 1), local_index=(0,))


def test_m_star_matches_owners_length() -> None:
    agg = OwnerSelectedAggregator(owners=(0, 1, 2), local_index=(0, 0, 0))
    assert agg.m_star == 3
