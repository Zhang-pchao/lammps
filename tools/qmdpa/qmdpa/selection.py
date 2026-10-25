# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Offline, finite-water region proposals; no potential or dynamics is changed.

Nearest-oxygen ownership is a structural diagnostic, not a bond order or charge
model. Ambiguous hydrogen ownership withholds a proposal. This module supplies
neither continuous partition weights nor their derivatives and must not be
used as a force-time adaptive partition policy.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np

from .core import Geometry, _atom_ids


@dataclass(frozen=True)
class SelectionIssue:
    """A reason why a frame cannot produce an unambiguous proposal."""

    code: str
    atom_ids: tuple[int, ...]
    message: str


@dataclass(frozen=True)
class HydrogenAssignment:
    """Nearest two oxygen distances, before any ownership is accepted."""

    hydrogen_id: int
    nearest_oxygen_id: int
    distance_angstrom: float
    second_oxygen_id: int | None
    second_distance_angstrom: float | None
    accepted: bool


@dataclass(frozen=True)
class WaterGroup:
    """An oxygen and its accepted hydrogens; kind is structural only."""

    oxygen_id: int
    hydrogen_ids: tuple[int, ...]
    kind: str

    @property
    def atom_ids(self):
        return (self.oxygen_id, *self.hydrogen_ids)


@dataclass(frozen=True)
class OxygenScore:
    """A distance score used only for geometric ranking."""

    oxygen_id: int
    score_angstrom: float
    selected: bool
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    """Immutable proposal and diagnostics in stable atom-ID order.

    Unresolved frames have no selected IDs. Ties spanning the budget boundary
    are reported even though stable IDs deterministically resolve their order.
    """

    status: str
    selected_atom_ids: tuple[int, ...]
    selected_oxygen_ids: tuple[int, ...]
    center_oxygen_ids: tuple[int, ...]
    groups: tuple[WaterGroup, ...]
    hydrogen_assignments: tuple[HydrogenAssignment, ...]
    scores: tuple[OxygenScore, ...]
    issues: tuple[SelectionIssue, ...]
    boundary_ties: tuple[int, ...]


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real number")
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a positive finite real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return value


@dataclass(frozen=True)
class WaterRegionSelector:
    """Propose whole OH groups around one or two explicit oxygen centers.

    For one center, rank by distance to it. For two, rank by the sum of the
    distances to both centers (ellipsoidal level sets). Centers are always
    included. The oxygen budget includes the centers. Equal scores are ordered
    by stable atom ID and ties across the selection boundary are reported.
    Only nonperiodic O/H geometries are accepted. Every frame is independent.
    """

    center_oxygen_ids: tuple[int, ...]
    oxygen_count: int
    max_oh_distance: float = 1.35
    shared_distance_gap: float = 0.10
    score_tolerance: float = 1e-10

    def __post_init__(self):
        centers = _atom_ids(self.center_oxygen_ids)
        if len(centers) not in (1, 2):
            raise ValueError("Exactly one or two distinct oxygen centers are required")
        if isinstance(self.oxygen_count, (bool, np.bool_)):
            raise TypeError("The oxygen count must be an integer")
        try:
            count = operator.index(self.oxygen_count)
        except TypeError as exc:
            raise ValueError("The oxygen count must be an integer") from exc
        if count < len(centers):
            raise ValueError("The oxygen count must include every center")
        object.__setattr__(self, "center_oxygen_ids", tuple(sorted(centers)))
        object.__setattr__(self, "oxygen_count", count)
        for name in ("max_oh_distance", "shared_distance_gap", "score_tolerance"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if self.shared_distance_gap >= self.max_oh_distance:
            raise ValueError("The shared-distance gap must be smaller than the bond limit")

    def select(self, geometry: Geometry) -> SelectionResult:
        """Analyze one frame without changing it or retaining previous members."""
        if not isinstance(geometry, Geometry):
            raise TypeError("Selection requires a Geometry")
        if geometry.periodic:
            raise ValueError("Offline water selection requires nonperiodic coordinates")
        if any(symbol not in ("O", "H") for symbol in geometry.symbols):
            raise ValueError("Offline water selection supports only oxygen and hydrogen")
        rows = {atom_id: row for row, atom_id in enumerate(geometry.atom_ids)}
        oxygens = tuple(sorted(i for i, s in zip(geometry.atom_ids, geometry.symbols) if s == "O"))
        hydrogens = tuple(
            sorted(i for i, s in zip(geometry.atom_ids, geometry.symbols) if s == "H")
        )
        if not oxygens or not hydrogens:
            raise ValueError("Water selection requires both oxygen and hydrogen atoms")
        if not set(self.center_oxygen_ids) <= set(oxygens):
            raise ValueError("Every center must be a present oxygen atom ID")
        if self.oxygen_count > len(oxygens):
            raise ValueError("The oxygen count exceeds the number of oxygen atoms")
        oxygen_positions = geometry.positions[[rows[i] for i in oxygens]]
        members = {oxygen: [] for oxygen in oxygens}
        assignments = []
        issues = []
        for hydrogen in hydrogens:
            distances = np.linalg.norm(
                oxygen_positions - geometry.positions[rows[hydrogen]], axis=1
            )
            if not np.all(np.isfinite(distances)):
                raise ValueError("Interatomic distances must be finite")
            order = sorted(range(len(oxygens)), key=lambda i: (distances[i], oxygens[i]))
            nearest = order[0]
            distance = float(distances[nearest])
            second = order[1] if len(order) > 1 else None
            second_distance = None if second is None else float(distances[second])
            accepted = True
            if distance <= 0 or distance > self.max_oh_distance:
                accepted = False
                issues.append(
                    SelectionIssue(
                        "invalid-oh-distance",
                        (hydrogen, oxygens[nearest]),
                        "The nearest O-H distance is zero or exceeds the configured limit.",
                    )
                )
            elif (
                second_distance is not None
                and second_distance - distance <= self.shared_distance_gap
            ):
                accepted = False
                issues.append(
                    SelectionIssue(
                        "shared-hydrogen",
                        (hydrogen, oxygens[nearest], oxygens[second]),
                        "Two oxygens are too close in distance to assign this hydrogen.",
                    )
                )
            if accepted:
                members[oxygens[nearest]].append(hydrogen)
            assignments.append(
                HydrogenAssignment(
                    hydrogen,
                    oxygens[nearest],
                    distance,
                    None if second is None else oxygens[second],
                    second_distance,
                    accepted,
                )
            )
        labels = {1: "hydroxide-like", 2: "water-like", 3: "hydronium-like"}
        groups = []
        for oxygen in oxygens:
            hydrogen_ids = tuple(members[oxygen])
            kind = labels.get(len(hydrogen_ids), "unresolved")
            groups.append(WaterGroup(oxygen, hydrogen_ids, kind))
            if kind == "unresolved":
                issues.append(
                    SelectionIssue(
                        "unsupported-oh-group",
                        (oxygen, *hydrogen_ids),
                        "An oxygen needs one, two or three unambiguously assigned hydrogens.",
                    )
                )
        if issues:
            return SelectionResult(
                "unresolved",
                (),
                (),
                self.center_oxygen_ids,
                tuple(groups),
                tuple(assignments),
                (),
                tuple(issues),
                (),
            )
        scores = {
            oxygen: float(
                sum(
                    np.linalg.norm(
                        geometry.positions[rows[oxygen]] - geometry.positions[rows[center]]
                    )
                    for center in self.center_oxygen_ids
                )
            )
            for oxygen in oxygens
        }
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError("Region distance scores must be finite")
        ranked = sorted(
            (oxygen for oxygen in oxygens if oxygen not in self.center_oxygen_ids),
            key=lambda oxygen: (scores[oxygen], oxygen),
        )
        capacity = self.oxygen_count - len(self.center_oxygen_ids)
        chosen = set(self.center_oxygen_ids).union(ranked[:capacity])
        ties = ()
        if 0 < capacity < len(ranked):
            last_score = scores[ranked[capacity - 1]]
            if abs(scores[ranked[capacity]] - last_score) <= self.score_tolerance:
                ties = tuple(
                    sorted(i for i in ranked if abs(scores[i] - last_score) <= self.score_tolerance)
                )
        atom_ids = tuple(
            sorted(atom for group in groups if group.oxygen_id in chosen for atom in group.atom_ids)
        )
        return SelectionResult(
            "selected",
            atom_ids,
            tuple(sorted(chosen)),
            self.center_oxygen_ids,
            tuple(groups),
            tuple(assignments),
            tuple(
                OxygenScore(
                    oxygen,
                    scores[oxygen],
                    oxygen in chosen,
                    "center"
                    if oxygen in self.center_oxygen_ids
                    else "within-oxygen-budget"
                    if oxygen in chosen
                    else "outside-oxygen-budget",
                )
                for oxygen in oxygens
            ),
            (),
            ties,
        )
