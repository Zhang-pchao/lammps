# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Strict offline policy input and explicit group identity contracts."""

import copy
import dataclasses

import numpy as np
import pytest

from qmdpa.core import Geometry
from qmdpa.selection import WaterRegionSelector
from qmdpa.selection_config import read_json, selection_from_settings


def water_settings():
    return {
        "schema_version": 1,
        "mode": "offline",
        "method": "water-nearest",
        "centers": [1],
        "oxygen_count": 1,
    }


def group_settings():
    return {
        "schema_version": 1,
        "mode": "offline",
        "method": "fixed-groups",
        "groups": [
            {"group_id": "oxide", "atom_ids": [20, 11], "symbols": ["O", "Ti"]},
            {"group_id": "adsorbate", "atom_ids": [105, 101], "symbols": ["H", "C"]},
            {"group_id": "environment", "atom_ids": [2**53 + 1], "symbols": ["Ar"]},
        ],
        "selected_group_ids": ["oxide", "adsorbate"],
    }


def nonwater_geometry():
    return Geometry(
        (11, 20, 101, 105, 2**53 + 1),
        ("Ti", "O", "C", "H", "Ar"),
        [[0, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [9, 0, 0]],
    )


def test_water_defaults_and_explicit_parameters_preserve_existing_selector():
    settings = water_settings()
    selector, resolved = selection_from_settings(settings)
    assert selector == WaterRegionSelector((1,), 1)
    assert resolved["max_oh_distance"] == 1.35
    assert resolved["shared_distance_gap"] == 0.10
    assert resolved["score_tolerance"] == 1e-10
    geometry = Geometry((1, 2, 3), ("O", "H", "H"), [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert selector.select(geometry) == WaterRegionSelector((1,), 1).select(geometry)
    settings.update(max_oh_distance=1.2, shared_distance_gap=0.2, score_tolerance=1e-8)
    configured, _ = selection_from_settings(settings)
    assert configured == WaterRegionSelector((1,), 1, 1.2, 0.2, 1e-8)


def test_generic_whole_groups_preserve_nonwater_identity_and_input_order():
    settings = group_settings()
    original = copy.deepcopy(settings)
    selector, resolved = selection_from_settings(settings)
    geometry = nonwater_geometry()
    proposal = selector.select(geometry)
    assert proposal.status == "selected"
    assert proposal.selected_atom_ids == (11, 20, 101, 105)
    assert proposal.selected_group_ids == ("adsorbate", "oxide")
    assert settings == original
    assert geometry.atom_ids == (11, 20, 101, 105, 2**53 + 1)
    reordered = Geometry(geometry.atom_ids[::-1], geometry.symbols[::-1], geometry.positions[::-1])
    assert proposal == selector.select(reordered)
    settings["groups"].reverse()
    settings["selected_group_ids"].reverse()
    for group in settings["groups"]:
        group["atom_ids"].reverse()
        group["symbols"].reverse()
    other, other_resolved = selection_from_settings(settings)
    assert resolved == other_resolved
    assert dataclasses.asdict(proposal) == dataclasses.asdict(other.select(reordered))


@pytest.mark.parametrize("change", ["missing", "extra", "element"])
def test_groups_require_complete_identity_coverage(change):
    settings = group_settings()
    if change == "missing":
        settings["groups"].pop()
    elif change == "extra":
        settings["groups"].append({"group_id": "extra", "atom_ids": [7], "symbols": ["He"]})
    else:
        settings["groups"][0]["symbols"][0] = "N"
    selector, _ = selection_from_settings(settings)
    with pytest.raises(ValueError, match="cover every input atom"):
        selector.select(nonwater_geometry())


def test_fixed_groups_reject_periodic_geometry():
    selector, _ = selection_from_settings(group_settings())
    geometry = nonwater_geometry()
    periodic = Geometry(geometry.atom_ids, geometry.symbols, geometry.positions, np.eye(3) * 20)
    with pytest.raises(ValueError, match="nonperiodic"):
        selector.select(periodic)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"mode": "adaptive"},
        {"mode": "dynamics"},
        {"mode": True},
        {"method": "unknown"},
        {"method": True},
        {"method": []},
        {"update_interval": 10},
        {"weights": [1.0]},
        {"centers": []},
        {"centers": [True]},
        {"centers": [1.0]},
        {"oxygen_count": True},
        {"oxygen_count": 1.0},
        {"max_oh_distance": True},
        {"max_oh_distance": float("nan")},
        {"max_oh_distance": float("inf")},
        {"max_oh_distance": "1.35"},
        {"shared_distance_gap": False},
        {"shared_distance_gap": 1.4},
        {"score_tolerance": 0.0},
        {"score_tolerance": [1e-10]},
    ],
)
def test_invalid_water_settings(changes):
    with pytest.raises((TypeError, ValueError)):
        selection_from_settings({**water_settings(), **changes})


@pytest.mark.parametrize("missing", ["schema_version", "mode", "method", "centers", "oxygen_count"])
def test_required_water_fields(missing):
    settings = water_settings()
    del settings[missing]
    with pytest.raises(ValueError):
        selection_from_settings(settings)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate-group",
        "overlap",
        "duplicate-atom",
        "unknown-selected",
        "duplicate-selected",
        "empty-selected",
        "bool-selected",
        "bool-atom",
        "bool-symbol",
        "missing-symbol",
        "unknown-group-field",
        "water-option",
        "empty-id",
        "non-list-groups",
    ],
)
def test_invalid_group_declarations(change):
    settings = group_settings()
    if change == "duplicate-group":
        settings["groups"][1]["group_id"] = "oxide"
    elif change == "overlap":
        settings["groups"][1]["atom_ids"][0] = 20
    elif change == "duplicate-atom":
        settings["groups"][0]["atom_ids"] = [11, 11]
    elif change == "unknown-selected":
        settings["selected_group_ids"] = ["absent"]
    elif change == "duplicate-selected":
        settings["selected_group_ids"] = ["oxide", "oxide"]
    elif change == "empty-selected":
        settings["selected_group_ids"] = []
    elif change == "bool-selected":
        settings["selected_group_ids"] = [True]
    elif change == "bool-atom":
        settings["groups"][0]["atom_ids"][0] = True
    elif change == "bool-symbol":
        settings["groups"][0]["symbols"][0] = True
    elif change == "missing-symbol":
        settings["groups"][0]["symbols"].pop()
    elif change == "unknown-group-field":
        settings["groups"][0]["charge"] = 0
    elif change == "water-option":
        settings["centers"] = [11]
    elif change == "empty-id":
        settings["groups"][0]["group_id"] = ""
    else:
        settings["groups"] = {}
    with pytest.raises((TypeError, ValueError)):
        selection_from_settings(settings)


@pytest.mark.parametrize(
    "raw",
    [
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{"x": 1, "x": 2}',
        '{"group": {"x": 1, "x": 2}}',
    ],
)
def test_json_rejects_nonfinite_constants_and_duplicate_fields(raw):
    with pytest.raises(ValueError):
        read_json(raw)
