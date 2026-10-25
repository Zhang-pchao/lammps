# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Versioned offline region settings and explicit, chemistry-independent groups."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .core import Geometry, _atom_ids
from .selection import WaterRegionSelector

WATER_DEFAULTS = {
    "max_oh_distance": 1.35,
    "shared_distance_gap": 0.10,
    "score_tolerance": 1e-10,
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON value is unsupported: {value}")


def read_json(data):
    """Decode JSON without silently accepting duplicate fields or NaN constants."""
    return json.loads(data, object_pairs_hook=_object, parse_constant=_constant)


def _fields(document, required, optional=(), *, name="Settings"):
    if not isinstance(document, dict):
        raise TypeError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in document):
        raise ValueError(f"{name} field names must be strings")
    missing = set(required) - set(document)
    unknown = set(document) - set(required) - set(optional)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _list(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON list")
    return value


def _group_id(value):
    if not isinstance(value, str) or not value or value.strip() != value or not value.isprintable():
        raise ValueError("Each group ID must be a nonempty printable string without outer spaces")
    return value


@dataclass(frozen=True)
class FixedGroup:
    """Caller-declared atom identities; no chemical bonds or charge are inferred."""

    group_id: str
    atom_ids: tuple[int, ...]
    symbols: tuple[str, ...]

    def __post_init__(self):
        group_id = _group_id(self.group_id)
        ids = _atom_ids(self.atom_ids)
        symbols = tuple(self.symbols)
        if len(ids) != len(symbols) or any(not isinstance(s, str) or not s for s in symbols):
            raise ValueError("Each group atom ID needs a nonempty element symbol")
        ordered = sorted(zip(ids, symbols))
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "atom_ids", tuple(i for i, _ in ordered))
        object.__setattr__(self, "symbols", tuple(s for _, s in ordered))


@dataclass(frozen=True)
class FixedGroupResult:
    """A whole-group offline proposal in stable atom and group ID order."""

    status: str
    selected_atom_ids: tuple[int, ...]
    selected_group_ids: tuple[str, ...]
    groups: tuple[FixedGroup, ...]


@dataclass(frozen=True)
class FixedGroupSelector:
    """Select declared groups after checking complete atom/element identity."""

    groups: tuple[FixedGroup, ...]
    selected_group_ids: tuple[str, ...]

    def __post_init__(self):
        groups = tuple(self.groups)
        if not groups or any(not isinstance(group, FixedGroup) for group in groups):
            raise ValueError("At least one explicit FixedGroup is required")
        group_ids = tuple(group.group_id for group in groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("Group IDs must be distinct")
        atom_ids = tuple(atom_id for group in groups for atom_id in group.atom_ids)
        if len(set(atom_ids)) != len(atom_ids):
            raise ValueError("Declared atom groups must be disjoint")
        selected = tuple(_group_id(value) for value in self.selected_group_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Selected group IDs must be nonempty and distinct")
        if not set(selected) <= set(group_ids):
            raise ValueError("Every selected group ID must be declared")
        object.__setattr__(self, "groups", tuple(sorted(groups, key=lambda group: group.group_id)))
        object.__setattr__(self, "selected_group_ids", tuple(sorted(selected)))

    def select(self, geometry: Geometry) -> FixedGroupResult:
        if not isinstance(geometry, Geometry):
            raise TypeError("Selection requires a Geometry")
        if geometry.periodic:
            raise ValueError("Offline group selection requires nonperiodic coordinates")
        expected = sorted(
            (atom_id, symbol)
            for group in self.groups
            for atom_id, symbol in zip(group.atom_ids, group.symbols)
        )
        actual = sorted(zip(geometry.atom_ids, geometry.symbols))
        if expected != actual:
            raise ValueError("Declared groups must cover every input atom ID and element exactly")
        selected = tuple(
            sorted(
                atom_id
                for group in self.groups
                if group.group_id in self.selected_group_ids
                for atom_id in group.atom_ids
            )
        )
        return FixedGroupResult("selected", selected, self.selected_group_ids, self.groups)


def selection_from_settings(document):
    """Return a validated selector and its complete, ordered version-1 settings.

    Only offline finite geometries are supported. Unknown fields and methods
    are errors, so an adaptive/dynamics request cannot silently become offline.
    """
    if not isinstance(document, dict):
        raise TypeError("Settings must be a JSON object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if document.get("mode") != "offline":
        raise ValueError("mode must be offline; adaptive dynamics is unsupported")
    method = document.get("method")
    common = ("schema_version", "mode", "method")
    if method == "water-nearest":
        _fields(document, (*common, "centers", "oxygen_count"), WATER_DEFAULTS)
        centers = _list(document["centers"], "centers")
        if type(document["oxygen_count"]) is not int:
            raise ValueError("oxygen_count must be an integer")
        selector = WaterRegionSelector(
            center_oxygen_ids=tuple(centers),
            oxygen_count=document["oxygen_count"],
            **{key: document.get(key, default) for key, default in WATER_DEFAULTS.items()},
        )
        resolved = {
            "schema_version": 1,
            "mode": "offline",
            "method": method,
            "centers": list(selector.center_oxygen_ids),
            "oxygen_count": selector.oxygen_count,
            **{key: getattr(selector, key) for key in WATER_DEFAULTS},
        }
    elif method == "fixed-groups":
        _fields(document, (*common, "groups", "selected_group_ids"))
        groups = []
        for group in _list(document["groups"], "groups"):
            _fields(group, ("group_id", "atom_ids", "symbols"), name="Group")
            groups.append(
                FixedGroup(
                    group["group_id"],
                    tuple(_list(group["atom_ids"], "Group atom_ids")),
                    tuple(_list(group["symbols"], "Group symbols")),
                )
            )
        selector = FixedGroupSelector(
            tuple(groups), tuple(_list(document["selected_group_ids"], "selected_group_ids"))
        )
        resolved = {
            "schema_version": 1,
            "mode": "offline",
            "method": method,
            "groups": [
                {
                    "group_id": group.group_id,
                    "atom_ids": list(group.atom_ids),
                    "symbols": list(group.symbols),
                }
                for group in selector.groups
            ],
            "selected_group_ids": list(selector.selected_group_ids),
        }
    else:
        raise ValueError("method must be water-nearest or fixed-groups")
    return selector, resolved
