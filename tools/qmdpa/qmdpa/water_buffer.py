#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Bounded whole-water buffers around a protected finite aqueous QM core."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .adaptive import MechanicalPartition, _integer, _real, _text
from .core import EvaluationError, Geometry
from .molecular_adaptive import (
    MolecularMechanicalMixer,
    MolecularWeights,
    _configure_family,
    _radial_preferences,
)
from .selection_config import FixedGroup, _group_id

HAMILTONIAN = "finite-water-buffer-mechanical"
SWITCH = "quintic-water-buffer"


@dataclass(frozen=True)
class QuinticWaterBufferWeights:
    """All subsets of one to six declared waters, including empty occupancy."""

    center_atom_id: int
    candidate_anchors: tuple[tuple[str, int], ...]
    inner_radius_angstrom: float
    outer_radius_angstrom: float
    max_partitions: int = 64

    def __post_init__(self):
        center = _integer(self.center_atom_id, "Center atom ID", 1)
        anchors = tuple(sorted((_group_id(g), _integer(a, "Water anchor ID", 1))
                               for g, a in self.candidate_anchors))
        if (not 1 <= len(anchors) <= 6 or len({g for g, _ in anchors}) != len(anchors)
                or len({a for _, a in anchors}) != len(anchors)
                or center in {a for _, a in anchors}):
            raise ValueError("Declare one to six distinct water anchors outside the core center")
        limit = _integer(self.max_partitions, "Maximum partition count", 1)
        if limit > 64 or 2 ** len(anchors) > limit:
            raise ValueError("The complete water-buffer family must fit within 64 partitions")
        inner = _real(self.inner_radius_angstrom, "Inner radius")
        outer = _real(self.outer_radius_angstrom, "Outer radius")
        if not 0 < inner < outer:
            raise ValueError("Radii require 0 < inner_radius < outer_radius")
        for name, value in (("center_atom_id", center), ("candidate_anchors", anchors),
                            ("inner_radius_angstrom", inner), ("outer_radius_angstrom", outer),
                            ("max_partitions", limit)):
            object.__setattr__(self, name, value)

    @property
    def group_subsets(self):
        groups = tuple(g for g, _ in self.candidate_anchors)
        return tuple(subset for n in range(len(groups) + 1) for subset in combinations(groups, n))

    def evaluate(self, geometry):
        preferences, derivatives, radii = _radial_preferences(
            geometry, self.center_atom_id, self.candidate_anchors,
            self.inner_radius_angstrom, self.outer_radius_angstrom)
        groups = tuple(g for g, _ in self.candidate_anchors)
        weights = []
        gradients = []
        for subset in self.group_subsets:
            included = np.array([g in subset for g in groups])
            factors = np.where(included, preferences, 1 - preferences)
            dfactors = np.where(included[:, None, None], derivatives, -derivatives)
            weights.append(np.prod(factors))
            gradient = np.zeros_like(geometry.positions)
            # No division by a vanishing preference at a pure endpoint.
            for i in range(len(groups)):
                others = [j for j in range(len(groups)) if j != i]
                gradient += np.prod(factors[others]) * dfactors[i]
            gradients.append(gradient)
        return MolecularWeights(self.group_subsets, weights, gradients, preferences, radii)


@dataclass(frozen=True)
class WaterBoundaryGuard:
    """Reject lost molecular identity; never reconstruct bonds from distances."""

    min_oh_angstrom: float
    max_oh_angstrom: float
    min_foreign_oh_angstrom: float

    def __post_init__(self):
        values = [_real(getattr(self, name), name) for name in
                  ("min_oh_angstrom", "max_oh_angstrom", "min_foreign_oh_angstrom")]
        if not 0 < values[0] < values[1] < values[2]:
            raise ValueError("Water guards require 0 < min_OH < max_OH < min_foreign_OH")
        for name, value in zip(("min_oh_angstrom", "max_oh_angstrom", "min_foreign_oh_angstrom"),
                               values, strict=True):
            object.__setattr__(self, name, value)

    def validate(self, geometry, groups, core_group_ids, switch):
        rows = {a: i for i, a in enumerate(geometry.atom_ids)}
        core = {a for g in groups if g.group_id in core_group_ids for a in g.atom_ids}
        oxygen = {a for a, s in zip(geometry.atom_ids, geometry.symbols, strict=True) if s == "O"}
        hydrogen = {a for a, s in zip(geometry.atom_ids, geometry.symbols, strict=True) if s == "H"}
        candidates = {g for g, _ in switch.candidate_anchors}

        def distance(a, b):
            return np.linalg.norm(geometry.positions[rows[a]] - geometry.positions[rows[b]])

        for group in groups:
            if group.group_id in core_group_ids:
                continue
            own_o = next(a for a, s in zip(group.atom_ids, group.symbols, strict=True) if s == "O")
            own_h = [a for a, s in zip(group.atom_ids, group.symbols, strict=True) if s == "H"]
            if (group.group_id not in candidates
                    and distance(own_o, switch.center_atom_id) < switch.outer_radius_angstrom):
                raise EvaluationError("An undeclared buffer water entered the switching support")
            for h in own_h:
                if not self.min_oh_angstrom <= distance(h, own_o) <= self.max_oh_angstrom:
                    raise EvaluationError("A buffer/environment water lost its declared O-H identity")
                if any(distance(h, other) < self.min_foreign_oh_angstrom for other in oxygen-{own_o}):
                    raise EvaluationError("A water hydrogen approached a foreign oxygen across the protected boundary")
            if any(distance(h, own_o) < self.min_foreign_oh_angstrom for h in hydrogen.intersection(core)):
                raise EvaluationError("A protected-core hydrogen approached an outer water oxygen")


class WaterBufferMechanicalMixer(MolecularMechanicalMixer):
    """Reuse the complete molecular mixer with declared water reference offsets."""

    hamiltonian = HAMILTONIAN

    def __init__(self, full_low, *, groups, core_group_ids, partitions, switch,
                 guard, reference_delta_ev, reference_source):
        _configure_family(self, groups, core_group_ids, partitions, switch,
                          MechanicalPartition, "high", switch_type=QuinticWaterBufferWeights)
        if not isinstance(guard, WaterBoundaryGuard):
            raise TypeError("A water buffer requires an explicit WaterBoundaryGuard")
        self.guard = guard
        self.full_low = full_low
        self.reference_delta_ev = _real(reference_delta_ev, "Water reference difference")
        self.reference_source = _text(reference_source, "Water reference source")
        core_symbols = []
        symbols = {}
        for group in self.groups:
            symbols.update(zip(group.atom_ids, group.symbols, strict=True))
            if group.group_id in self.core_group_ids:
                core_symbols.extend(group.symbols)
            elif sorted(group.symbols) != ["H", "H", "O"]:
                raise ValueError("Every outer group must be one declared complete H2O molecule")
        if (set(core_symbols) != {"H", "O"}
                or core_symbols.count("H") != 2*core_symbols.count("O")):
            raise ValueError("The protected neutral aqueous core requires H:O composition 2:1")
        if any(symbols[a] != "O" for a in [switch.center_atom_id]+[a for _, a in switch.candidate_anchors]):
            raise ValueError("The protected center and water anchors must be oxygen atoms")
        for partition, subset in zip(self.partitions, self.switch.group_subsets, strict=True):
            expected = -len(subset)*self.reference_delta_ev
            if (partition.energy_offset_ev != expected
                    or partition.offset_source != self.reference_source):
                raise ValueError("Every partition offset must match its fixed water reference and occupancy")
            if (partition.qm_charge, partition.qm_multiplicity) != (0, 1):
                raise ValueError("The aqueous buffer family requires neutral singlet partitions")

    def _validate_geometry(self, geometry):
        super()._validate_geometry(geometry)
        self.guard.validate(geometry, self.groups, self.core_group_ids, self.switch)


def frame_record(frame, config):
    """Reconstruct deterministic occupancy evidence from an accepted geometry."""
    from .water_reference import reference_identity

    symbols = dict(zip(config["full_region"]["atom_ids"], config["full_region"]["symbols"], strict=True))
    ids = tuple(a["id"] for a in frame["atoms"])
    if set(ids) != set(symbols):
        raise ValueError("Water-buffer frame does not cover the declared system")
    geometry = Geometry(ids, tuple(symbols[a] for a in ids), [a["x"] for a in frame["atoms"]])
    switch = QuinticWaterBufferWeights(**{k: v for k, v in config["switch"].items() if k != "method"})
    groups = tuple(FixedGroup(**g) for g in config["groups"])
    WaterBoundaryGuard(**config["water_guard"]).validate(geometry, groups, config["core_group_ids"], switch)
    weights = switch.evaluate(geometry)
    partitions = [p["partition_id"] for p in config["partitions"]]
    counts = [len(s) for s in switch.group_subsets]
    return {
        "reference_sha256": reference_identity(config["buffer_reference"]),
        "partition_ids": partitions, "buffer_water_counts": counts,
        "weights": weights.weights.tolist(),
        "candidate_preferences": weights.candidate_weights.tolist(),
        "expected_buffer_waters": float(np.dot(counts, weights.weights)),
        "active_partition_ids": [partitions[i] for i in range(len(partitions))
                                 if weights.weights[i] != 0 or np.any(weights.gradients_per_angstrom[i])],
    }


def validate_buffer_frame(frame, config):
    """Occupancy is derived state, not an independently adjustable history."""
    if config.get("hamiltonian") != HAMILTONIAN:
        if "water_buffer" in frame:
            raise ValueError("Unexpected water-buffer frame record")
        return
    expected = frame_record(frame, config)
    actual = frame.get("water_buffer")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("Missing or invalid water-buffer frame record")
    numeric = {"weights", "candidate_preferences", "expected_buffer_waters"}
    for key, value in expected.items():
        if key in numeric:
            observed = np.asarray(actual[key])
            target = np.asarray(value)
            if (observed.shape != target.shape or observed.dtype.kind not in "fiu"
                    or not np.all(np.isfinite(observed))
                    or not np.allclose(observed, target, atol=1e-12, rtol=0)):
                raise ValueError("Water-buffer frame weights or occupancy differ from its geometry")
        elif actual[key] != value:
            raise ValueError("Water-buffer frame identity or partition family differs")
