# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Conservative finite exchange among multiple complete molecular groups.

A fixed core retains exactly k candidate molecules. Every k-subset belongs to
the declared family. Normalized compact radial weights change with coordinates;
their full derivatives enter the force. Chemical grouping is never inferred.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .adaptive import MechanicalPartition, _integer, _real
from .core import Evaluation, EvaluationError, Geometry, _evaluate, _readonly_array
from .selection_config import FixedGroup, FixedGroupSelector, _group_id


@dataclass(frozen=True, eq=False)
class MolecularWeights:
    """Normalized family weights and derivatives in the supplied atom order."""

    group_subsets: tuple[tuple[str, ...], ...]
    weights: np.ndarray
    gradients_per_angstrom: np.ndarray
    candidate_weights: np.ndarray
    radii_angstrom: np.ndarray

    def __post_init__(self):
        weights = _readonly_array(self.weights, "Molecular partition weights")
        gradients = _readonly_array(self.gradients_per_angstrom, "Molecular weight derivatives")
        candidates = _readonly_array(self.candidate_weights, "Candidate weights")
        radii = _readonly_array(self.radii_angstrom, "Candidate radii")
        if (weights.shape != (len(self.group_subsets),)
                or gradients.ndim != 3 or gradients.shape[0] != len(weights)
                or gradients.shape[2] != 3 or gradients.shape[1] == 0):
            raise ValueError("Partition weight and derivative counts must match the family")
        if (np.any(weights < 0) or np.any(weights > 1)
                or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
                or not np.allclose(gradients.sum(axis=0), 0, rtol=0, atol=1e-10)):
            raise EvaluationError("Molecular weights or their derivatives are not normalized")
        if candidates.shape != radii.shape or candidates.ndim != 1:
            raise ValueError("Candidate weight and radius counts differ")
        for name, value in (("weights", weights), ("gradients_per_angstrom", gradients),
                            ("candidate_weights", candidates), ("radii_angstrom", radii)):
            object.__setattr__(self, name, value)


def _radial_preferences(geometry, center_atom_id, candidate_anchors,
                        inner_radius_angstrom, outer_radius_angstrom):
    """Shared compact quintic preferences and complete center reaction."""
    if not isinstance(geometry, Geometry) or geometry.periodic:
        raise ValueError("Molecular weights require a nonperiodic geometry")
    lookup = {atom: i for i, atom in enumerate(geometry.atom_ids)}
    try:
        center = lookup[center_atom_id]
        rows = [lookup[atom] for _, atom in candidate_anchors]
    except KeyError as exc:
        raise ValueError("Every molecular anchor must be present") from exc
    delta = geometry.positions[rows] - geometry.positions[center]
    radii = np.linalg.norm(delta, axis=1)
    if not np.isfinite(radii).all() or np.any(radii == 0):
        raise ValueError("Molecular anchors must have finite, nonzero center distances")
    preference = np.zeros(len(rows))
    derivatives = np.zeros((len(rows), len(geometry.atom_ids), 3))
    width = outer_radius_angstrom - inner_radius_angstrom
    for i, (row, radius) in enumerate(zip(rows, radii)):
        if radius <= inner_radius_angstrom:
            preference[i] = 1
        elif radius < outer_radius_angstrom:
            t = (radius - inner_radius_angstrom) / width
            u = t if t <= 0.5 else 1 - t
            tail = u ** 3 * (10 + u * (-15 + 6 * u))
            preference[i] = 1 - tail if t <= 0.5 else tail
            radial = -30 * t ** 2 * (1 - t) ** 2 / width
            derivatives[i, row] = radial * delta[i] / radius
            derivatives[i, center] = -derivatives[i, row]
    return preference, derivatives, radii


@dataclass(frozen=True)
class QuinticMolecularWeights:
    """Select k of the declared candidate groups with smooth finite weights.

    Each candidate has an explicit anchor ID. A radial preference is one below
    inner_radius, zero beyond outer_radius, and a quintic between them. Each
    k-subset receives the product of its preferences, normalized over all such
    subsets. At least k candidates must have positive preference. Loss of this
    coverage is an error, never an abrupt change to k or to another model.
    """

    center_atom_id: int
    candidate_anchors: tuple[tuple[str, int], ...]
    occupancy: int
    inner_radius_angstrom: float
    outer_radius_angstrom: float
    max_partitions: int = 64

    def __post_init__(self):
        center = _integer(self.center_atom_id, "Center atom ID", 1)
        anchors = tuple(sorted((_group_id(group), _integer(atom, "Anchor atom ID", 1))
                               for group, atom in self.candidate_anchors))
        if (len(anchors) < 2 or len({g for g, _ in anchors}) != len(anchors)
                or len({a for _, a in anchors}) != len(anchors)
                or center in {a for _, a in anchors}):
            raise ValueError("Distinct candidate groups and anchors outside the center are required")
        occupancy = _integer(self.occupancy, "Candidate occupancy", 1)
        if occupancy >= len(anchors):
            raise ValueError("Candidate occupancy must be smaller than the candidate count")
        limit = _integer(self.max_partitions, "Maximum partition count", 1)
        if math.comb(len(anchors), occupancy) > limit:
            raise ValueError("The complete partition family exceeds max_partitions")
        inner = _real(self.inner_radius_angstrom, "Inner radius")
        outer = _real(self.outer_radius_angstrom, "Outer radius")
        if not 0 < inner < outer:
            raise ValueError("Radii require 0 < inner_radius < outer_radius")
        for name, value in (("center_atom_id", center), ("candidate_anchors", anchors),
                            ("occupancy", occupancy), ("max_partitions", limit),
                            ("inner_radius_angstrom", inner), ("outer_radius_angstrom", outer)):
            object.__setattr__(self, name, value)

    @property
    def group_subsets(self):
        return tuple(combinations(tuple(g for g, _ in self.candidate_anchors), self.occupancy))

    def evaluate(self, geometry: Geometry):
        preference, derivatives, radii = _radial_preferences(
            geometry, self.center_atom_id, self.candidate_anchors,
            self.inner_radius_angstrom, self.outer_radius_angstrom)
        if np.count_nonzero(preference > 0) < self.occupancy:
            raise EvaluationError("Too few candidate molecules inside the outer radius for occupancy")
        # All subsets have the same degree. A common scale cancels in the
        # normalized quotient, including its derivative, and avoids tiny products.
        scale = np.max(preference)
        factors, dfactors = preference / scale, derivatives / scale
        subsets = tuple(combinations(range(len(preference)), self.occupancy))
        values = np.array([np.prod(factors[list(subset)]) for subset in subsets])
        gradients = np.zeros((len(subsets), len(geometry.atom_ids), 3))
        for index, subset in enumerate(subsets):
            # Product rule without division remains defined at pure endpoints.
            for member in subset:
                other = [j for j in subset if j != member]
                gradients[index] += np.prod(factors[other]) * dfactors[member]
        normalization = values.sum()
        if not np.isfinite(normalization) or normalization <= 0:
            raise EvaluationError("Molecular weight normalization is outside finite range")
        weights = values / normalization
        gradients = (gradients - weights[:, None, None] * gradients.sum(axis=0)) / normalization
        return MolecularWeights(self.group_subsets, weights, gradients, preference, radii)


@dataclass(frozen=True, eq=False)
class MolecularAdaptiveTerms:
    """Full ledger; an inactive partition has no calculated correction (None)."""

    atom_ids: tuple[int, ...]
    partition_ids: tuple[str, ...]
    qm_atom_ids: tuple[tuple[int, ...], ...]
    weights: MolecularWeights
    active_partition_ids: tuple[str, ...]
    baseline: Evaluation
    corrections: tuple[Evaluation | None, ...]
    switching_forces: np.ndarray
    correction: Evaluation
    total: Evaluation


def _configure_family(self, groups, core_group_ids, partitions, switch,
                      partition_type, high_attribute, *, switch_type=QuinticMolecularWeights):
    """Validate the shared complete molecular family without evaluating a provider."""
    if not isinstance(switch, switch_type):
        raise TypeError(f"Molecular exchange requires {switch_type.__name__}")
    groups = tuple(groups)
    if not groups or any(not isinstance(g, FixedGroup) for g in groups):
        raise ValueError("Explicit complete FixedGroup declarations are required")
    core = tuple(sorted(_group_id(g) for g in core_group_ids))
    self._coverage = FixedGroupSelector(groups, core)
    self.groups = self._coverage.groups
    by_group = {g.group_id: g for g in self.groups}
    candidates = tuple(g for g, _ in switch.candidate_anchors)
    if not set(candidates) <= set(by_group) or set(core).intersection(candidates):
        raise ValueError("Candidate groups must be declared and outside the fixed core")
    if switch.center_atom_id not in {a for g in core for a in by_group[g].atom_ids}:
        raise ValueError("The center anchor must belong to the fixed core")
    for group, atom in switch.candidate_anchors:
        if atom not in by_group[group].atom_ids:
            raise ValueError("Each candidate anchor must belong to its declared complete group")
    compositions = [Counter(by_group[g].symbols) for g in candidates]
    if any(x != compositions[0] for x in compositions[1:]):
        raise ValueError("Exchange candidates must have identical elemental composition")
    partitions = tuple(partitions)
    if not partitions or any(not isinstance(p, partition_type) for p in partitions):
        raise ValueError(f"Explicit {partition_type.__name__} declarations are required")
    if len({p.partition_id for p in partitions}) != len(partitions):
        raise ValueError("Partition IDs must be distinct")
    by_subset = {p.qm_group_ids: p for p in partitions}
    expected = tuple(tuple(sorted(core + subset)) for subset in switch.group_subsets)
    if len(by_subset) != len(partitions) or set(by_subset) != set(expected):
        raise ValueError("Partitions must contain the complete fixed-occupancy family exactly")
    self.partitions = tuple(by_subset[subset] for subset in expected)
    first = self.partitions[0]
    for partition in self.partitions:
        for name in ("qm_charge", "qm_multiplicity", "method_identity",
                     "energy_reference", "electronic_state"):
            if getattr(partition, name) != getattr(first, name):
                raise ValueError(f"Every partition must share {name}")
        for name in ("xc", "basis"):
            if getattr(getattr(partition.calculator, high_attribute), name, None) != getattr(getattr(first.calculator, high_attribute), name, None):
                raise ValueError(f"Every high provider must share {name}")
    self.qm_atom_ids = tuple(tuple(sorted(a for group in p.qm_group_ids
                                         for a in by_group[group].atom_ids))
                             for p in self.partitions)
    self._atom_ids = tuple(sorted(a for g in self.groups for a in g.atom_ids))
    self.core_group_ids, self.switch = core, switch


def _mix_values(weights, values):
    """Blend complete same-order quantities, retaining every weight derivative."""
    active = tuple(i for i, value in enumerate(values) if value is not None)
    origin = values[active[0]].energy
    energy = origin
    force = np.zeros_like(values[active[0]].forces)
    switching = np.zeros_like(force)
    for i in active:
        value = values[i]
        energy += weights.weights[i] * (value.energy - origin)
        force += weights.weights[i] * value.forces
        switching -= (value.energy - origin) * weights.gradients_per_angstrom[i]
    return Evaluation(energy, force + switching), _readonly_array(switching, "Switching forces")


class MolecularMechanicalMixer:
    """Ground ME exchange with fixed composition and a complete k-subset family.

    The core is retained. All candidates must have identical composition and
    every partition must have the same declared method, energy reference and
    charge/spin. Offsets remain explicit. A varying QM molecule count, chemical
    regrouping, excited-state correspondence and periodicity are separate models.
    """

    hamiltonian = "finite-molecular-mechanical"

    def __init__(self, full_low, *, groups, core_group_ids, partitions, switch):
        partitions = tuple(partitions)
        if (self.hamiltonian == "finite-molecular-mechanical"
                and any(isinstance(p, MechanicalPartition) and p.electronic_state != "ground"
                        for p in partitions)):
            raise ValueError("The ground molecular model cannot admit selected-state partitions")
        _configure_family(self, groups, core_group_ids, partitions, switch,
                          MechanicalPartition, "high")
        self.full_low = full_low

    def _validate_geometry(self, geometry):
        self._coverage.select(geometry)

    def evaluate_terms(self, geometry):
        self._validate_geometry(geometry)
        weights = self.switch.evaluate(geometry)
        for partition in self.partitions:
            partition.validate_provider_state()
        ordered = geometry.select(self._atom_ids)
        result = _evaluate(self.full_low, ordered)
        lookup = {a: i for i, a in enumerate(geometry.atom_ids)}
        rows = {a: i for i, a in enumerate(self._atom_ids)}
        baseline = Evaluation(result.energy, result.forces[[rows[a] for a in geometry.atom_ids]])
        corrections = []
        active = []
        for i, (partition, ids) in enumerate(zip(self.partitions, self.qm_atom_ids)):
            if weights.weights[i] == 0 and not np.any(weights.gradients_per_angstrom[i]):
                corrections.append(None)
                continue
            value = partition.calculator.correction(geometry.select(ids))
            force = np.zeros_like(geometry.positions)
            force[[lookup[a] for a in ids]] = value.forces
            corrections.append(Evaluation(value.energy + partition.energy_offset_ev, force))
            active.append(i)
        correction, switching = _mix_values(weights, corrections)
        total = Evaluation(baseline.energy + correction.energy, baseline.forces + correction.forces)
        return MolecularAdaptiveTerms(
            geometry.atom_ids, tuple(p.partition_id for p in self.partitions), self.qm_atom_ids,
            weights, tuple(self.partitions[i].partition_id for i in active), baseline,
            tuple(corrections), _readonly_array(switching, "Switching forces"), correction, total,
        )

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).total
