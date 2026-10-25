# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Whole-water proposals, ambiguous proton ownership, and stable identity."""

import dataclasses

import numpy as np
import pytest

from qmdpa.core import Geometry
from qmdpa.selection import WaterRegionSelector


def waters(oxygen_positions=((0, 0, 0), (3, 0, 0), (0, 4, 0)), counts=None):
    if counts is None:
        counts = [2] * len(oxygen_positions)
    positions, symbols = [], []
    offsets = np.array([[0.75, 0.55, 0], [-0.75, 0.55, 0], [0, -0.9, 0], [0, 0, 0.9]])
    for oxygen, count in zip(oxygen_positions, counts):
        positions.append(oxygen)
        symbols.append("O")
        for offset in offsets[:count]:
            positions.append(np.asarray(oxygen) + offset)
            symbols.append("H")
    return Geometry(tuple(range(1, len(positions) + 1)), tuple(symbols), positions)


def test_neutral_proposal_contains_complete_groups_and_preserves_geometry():
    geometry = waters()
    original = geometry.positions.copy()
    result = WaterRegionSelector((1,), 2).select(geometry)
    assert result.status == "selected"
    assert result.selected_oxygen_ids == (1, 4)
    assert result.selected_atom_ids == (1, 2, 3, 4, 5, 6)
    assert tuple(group.kind for group in result.groups) == ("water-like",) * 3
    assert [(score.oxygen_id, score.reason) for score in result.scores] == [
        (1, "center"),
        (4, "within-oxygen-budget"),
        (7, "outside-oxygen-budget"),
    ]
    assert result.issues == ()
    np.testing.assert_array_equal(geometry.positions, original)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "changed"
    with pytest.raises(ValueError):
        geometry.positions[0, 0] = 1


def test_two_centers_include_both_ions_and_choose_ellipsoid_interior():
    geometry = waters(((0, 0, 0), (8, 0, 0), (4, 0, 0), (0, 4, 0)), [3, 1, 2, 2])
    result = WaterRegionSelector((5, 1), 3).select(geometry)
    assert result.center_oxygen_ids == (1, 5)
    assert result.selected_oxygen_ids == (1, 5, 7)
    assert result.selected_atom_ids == tuple(range(1, 10))
    assert [(group.oxygen_id, group.kind) for group in result.groups[:2]] == [
        (1, "hydronium-like"),
        (5, "hydroxide-like"),
    ]
    assert result.scores[2].score_angstrom == pytest.approx(8)
    assert result.scores[3].score_angstrom > 8


def test_membership_depends_on_current_geometry_without_history_fallback():
    selector = WaterRegionSelector((1,), 2)
    first = selector.select(waters())
    second = selector.select(waters(((0, 0, 0), (6, 0, 0), (0, 3, 0))))
    assert first.selected_oxygen_ids == (1, 4)
    assert second.selected_oxygen_ids == (1, 7)
    assert selector.select(waters()) == first


def test_shared_proton_withholds_proposal_instead_of_hard_assignment():
    geometry = waters(((0, 0, 0), (2.4, 0, 0)))
    positions = geometry.positions.copy()
    positions[1] = [1.2, 0, 0]
    shared = Geometry(geometry.atom_ids, geometry.symbols, positions)
    selector = WaterRegionSelector((1,), 1)
    assert selector.select(geometry).status == "selected"
    result = selector.select(shared)
    assert result.status == "unresolved"
    assert result.selected_atom_ids == ()
    assert result.selected_oxygen_ids == ()
    issue = next(issue for issue in result.issues if issue.code == "shared-hydrogen")
    assert issue.atom_ids == (2, 1, 4)
    assert not result.hydrogen_assignments[0].accepted
    assert result.hydrogen_assignments[0].distance_angstrom == pytest.approx(1.2)
    assert selector.select(geometry).status == "selected"


def test_proton_ownership_changes_without_changing_atom_ids():
    before = waters(((0, 0, 0), (2.5, 0, 0)))
    positions = before.positions.copy()
    positions[1] = [1.55, 0, 0]
    after = Geometry(before.atom_ids, before.symbols, positions)
    selector = WaterRegionSelector((1, 4), 2)
    first, second = selector.select(before), selector.select(after)
    assert first.status == second.status == "selected"
    assert [group.kind for group in first.groups] == ["water-like", "water-like"]
    assert [group.kind for group in second.groups] == ["hydroxide-like", "hydronium-like"]
    assert second.groups[1].hydrogen_ids == (2, 5, 6)
    assert first.selected_atom_ids == second.selected_atom_ids


def test_reordered_input_and_large_ids_preserve_proposals():
    original = waters()
    large = Geometry(
        tuple(i + 2**40 for i in original.atom_ids), original.symbols, original.positions
    )
    selector = WaterRegionSelector((2**40 + 1,), 2)
    result = selector.select(large)
    reversed_geometry = Geometry(large.atom_ids[::-1], large.symbols[::-1], large.positions[::-1])
    assert selector.select(reversed_geometry) == result
    assert result.selected_atom_ids == tuple(i + 2**40 for i in range(1, 7))


def test_rigid_motion_preserves_membership_and_scores():
    original = waters()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = Geometry(
        original.atom_ids, original.symbols, original.positions @ rotation + [7, -3, 2]
    )
    selector = WaterRegionSelector((1,), 2)
    first, second = selector.select(original), selector.select(moved)
    assert first.selected_atom_ids == second.selected_atom_ids
    np.testing.assert_allclose(
        [item.score_angstrom for item in first.scores],
        [item.score_angstrom for item in second.scores],
        atol=1e-12,
    )


def test_ties_at_budget_boundary_are_explicit_and_deterministic():
    geometry = waters(((0, 0, 0), (3, 0, 0), (-3, 0, 0)))
    result = WaterRegionSelector((1,), 2).select(geometry)
    assert result.selected_oxygen_ids == (1, 4)
    assert result.boundary_ties == (4, 7)
    assert WaterRegionSelector((1,), 3).select(geometry).boundary_ties == ()


def test_unbound_or_coincident_hydrogen_and_unsupported_group_are_flagged():
    geometry = waters(((0, 0, 0),), [4])
    result = WaterRegionSelector((1,), 1).select(geometry)
    assert result.status == "unresolved"
    assert result.issues[0].code == "unsupported-oh-group"
    for position in ([9, 0, 0], [0, 0, 0]):
        original = waters(((0, 0, 0),))
        positions = original.positions.copy()
        positions[1] = position
        result = WaterRegionSelector((1,), 1).select(
            Geometry(original.atom_ids, original.symbols, positions)
        )
        assert result.status == "unresolved"
        assert result.issues[0].code == "invalid-oh-distance"
        assert result.selected_atom_ids == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"center_oxygen_ids": ()},
        {"center_oxygen_ids": (1, 1)},
        {"center_oxygen_ids": (1, 4, 7)},
        {"center_oxygen_ids": (True,)},
        {"oxygen_count": True},
        {"oxygen_count": 1.5},
        {"oxygen_count": 0},
        {"center_oxygen_ids": (1, 4), "oxygen_count": 1},
        {"max_oh_distance": float("nan")},
        {"max_oh_distance": True},
        {"shared_distance_gap": 0},
        {"shared_distance_gap": 2},
        {"score_tolerance": -1},
    ],
)
def test_invalid_selector_configuration_fails(changes):
    config = {"center_oxygen_ids": (1,), "oxygen_count": 2, **changes}
    with pytest.raises((ValueError, TypeError)):
        WaterRegionSelector(**config)


def test_invalid_geometry_and_periodicity_are_rejected():
    geometry = waters()
    selector = WaterRegionSelector((1,), 2)
    with pytest.raises(ValueError, match="nonperiodic"):
        selector.select(
            Geometry(geometry.atom_ids, geometry.symbols, geometry.positions, np.eye(3))
        )
    with pytest.raises(ValueError, match="only oxygen"):
        selector.select(
            Geometry(geometry.atom_ids, ("C", *geometry.symbols[1:]), geometry.positions)
        )
    with pytest.raises(ValueError, match="present oxygen"):
        WaterRegionSelector((2,), 2).select(geometry)
    with pytest.raises(ValueError, match="exceeds"):
        WaterRegionSelector((1,), 4).select(geometry)
    with pytest.raises(TypeError, match="Geometry"):
        selector.select(None)
    with pytest.raises(ValueError, match="both oxygen"):
        selector.select(Geometry((1,), ("O",), [[0, 0, 0]]))
