# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Conservative two-partition mechanical diagnostics with complete fixed groups.

A common full-system baseline is evaluated once. Both isolated QM corrections
are evaluated on every call, including at pure endpoints. This module neither
changes an MD partition nor infers bonds, electronic states, or energy offsets.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .core import (
    Evaluation,
    EvaluationError,
    Geometry,
    SubtractiveCalculator,
    _evaluate,
    _readonly_array,
)
from .selection_config import FixedGroup, FixedGroupSelector


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicitly declared")
    return value


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, eq=False)
class MechanicalPartition:
    """One declared ground-state QM correction and its energy-reference contract.

    The high provider must expose charge and multiplicity, as PySCFBackend does.
    A low provider's non-None charge_spin, when available, must match them.
    method_identity and energy_reference are caller-owned scientific identities;
    a label alone does not qualify an arbitrary provider as a ground-state one.
    """

    partition_id: str
    qm_group_ids: tuple[str, ...]
    calculator: SubtractiveCalculator
    qm_charge: int
    qm_multiplicity: int
    method_identity: str
    energy_reference: str
    energy_offset_ev: float = 0.0
    offset_source: str | None = None
    electronic_state: str = "ground"

    def __post_init__(self):
        _text(self.partition_id, "partition_id")
        if isinstance(self.qm_group_ids, str):
            raise TypeError("qm_group_ids must list distinct complete groups")
        groups = tuple(_text(value, "QM group ID") for value in self.qm_group_ids)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("qm_group_ids must be nonempty and distinct")
        if not isinstance(self.calculator, SubtractiveCalculator):
            raise TypeError("A mechanical partition requires a SubtractiveCalculator")
        charge = _integer(self.qm_charge, "QM charge")
        multiplicity = _integer(self.qm_multiplicity, "QM multiplicity", 1)
        _text(self.method_identity, "method_identity")
        _text(self.energy_reference, "energy_reference")
        offset = _real(self.energy_offset_ev, "energy_offset_ev")
        if offset != 0.0 or self.offset_source is not None:
            _text(self.offset_source, "A declared energy offset source")
        self._validate_electronic_state()
        object.__setattr__(self, "qm_group_ids", tuple(sorted(groups)))
        object.__setattr__(self, "qm_charge", charge)
        object.__setattr__(self, "qm_multiplicity", multiplicity)
        object.__setattr__(self, "energy_offset_ev", offset)
        self.validate_provider_state()

    def _validate_electronic_state(self):
        if self.electronic_state != "ground":
            raise ValueError("Only declared ground-state partitions are supported")

    def validate_provider_state(self):
        """Check declared states without importing or selecting a scientific engine."""
        high = self.calculator.high
        if (
            _integer(getattr(high, "charge", None), "High-provider charge") != self.qm_charge
            or _integer(getattr(high, "multiplicity", None), "High-provider multiplicity", 1)
            != self.qm_multiplicity
        ):
            raise ValueError("High-provider charge/multiplicity differs from its partition")
        condition = getattr(self.calculator.low, "charge_spin", None)
        if condition is not None:
            values = np.asarray(condition)
            if (
                values.shape != (2,)
                or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
                or not np.array_equal(values, [self.qm_charge, self.qm_multiplicity])
            ):
                raise ValueError("Low-region charge_spin differs from its QM partition state")


@dataclass(frozen=True, eq=False)
class PartitionWeights:
    """Two normalized weights and all Cartesian derivatives, in input atom order."""

    coordinate_angstrom: float
    weights: np.ndarray
    gradients_per_angstrom: np.ndarray

    def __post_init__(self):
        coordinate = _real(self.coordinate_angstrom, "Exchange coordinate")
        weights = _readonly_array(self.weights, "Partition weights")
        gradients = _readonly_array(self.gradients_per_angstrom, "Weight gradients")
        if weights.shape != (2,) or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("Two partition weights in [0, 1] are required")
        if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Partition weights must sum to one")
        if (
            gradients.ndim != 3
            or gradients.shape[0] != 2
            or gradients.shape[1] == 0
            or gradients.shape[2] != 3
        ):
            raise ValueError("Weight gradients must have shape (2, atoms, 3)")
        if not np.allclose(gradients.sum(axis=0), 0.0, rtol=0, atol=1e-12):
            raise ValueError("Partition weight gradients must sum to zero")
        object.__setattr__(self, "coordinate_angstrom", coordinate)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "gradients_per_angstrom", gradients)


@dataclass(frozen=True)
class QuinticDistanceExchange:
    """Exchange two complete candidate groups around an explicit center atom.

    s = distance(center, first) - distance(center, second). Partition A is pure
    when s <= -half_width and partition B when s >= half_width. The intervening
    quintic has zero first and second derivatives at both endpoints.
    """

    center_atom_id: int
    first_atom_id: int
    second_atom_id: int
    half_width_angstrom: float

    def __post_init__(self):
        anchors = tuple(
            _integer(getattr(self, name), name, 1)
            for name in ("center_atom_id", "first_atom_id", "second_atom_id")
        )
        if len(set(anchors)) != 3:
            raise ValueError("Center and candidate anchor IDs must be distinct")
        width = _real(self.half_width_angstrom, "half_width_angstrom")
        if width <= 0:
            raise ValueError("half_width_angstrom must be positive")
        for name, value in zip(("center_atom_id", "first_atom_id", "second_atom_id"), anchors):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "half_width_angstrom", width)

    def evaluate(self, geometry: Geometry) -> PartitionWeights:
        if not isinstance(geometry, Geometry):
            raise TypeError("The exchange policy requires a Geometry")
        if geometry.periodic:
            raise ValueError("Exchange weights require a nonperiodic geometry")
        rows = {atom: index for index, atom in enumerate(geometry.atom_ids)}
        try:
            center, first, second = (
                rows[atom]
                for atom in (self.center_atom_id, self.first_atom_id, self.second_atom_id)
            )
        except KeyError as exc:
            raise ValueError("Every exchange anchor must be present") from exc
        positions = geometry.positions
        if np.array_equal(positions[first], positions[second]):
            raise ValueError("Candidate anchor positions must not coincide")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                delta_a = positions[first] - positions[center]
                delta_b = positions[second] - positions[center]
                radius_a, radius_b = np.linalg.norm(delta_a), np.linalg.norm(delta_b)
                if radius_a == 0 or radius_b == 0:
                    raise ValueError("Candidate and center anchor positions must not coincide")
                coordinate = float(radius_a - radius_b)
                width = self.half_width_angstrom
                if coordinate <= -width:
                    weight_a, derivative = 1.0, 0.0
                elif coordinate >= width:
                    weight_a, derivative = 0.0, 0.0
                else:
                    t = 0.5 * (coordinate / width + 1.0)
                    # Evaluate the smaller tail to avoid cancellation near a pure endpoint.
                    u = t if t <= 0.5 else 1.0 - t
                    tail = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
                    weight_a = 1.0 - tail if t <= 0.5 else tail
                    derivative = -15.0 * t**2 * (1.0 - t) ** 2 / width
                gradients = np.zeros((2, len(geometry.atom_ids), 3))
                gradients[0, first] = derivative * delta_a / radius_a
                gradients[0, second] = -derivative * delta_b / radius_b
                gradients[0, center] = -gradients[0, first] - gradients[0, second]
                gradients[1] = -gradients[0]
        except FloatingPointError as exc:
            raise EvaluationError("Exchange geometry is outside finite numerical range") from exc
        return PartitionWeights(coordinate, [weight_a, 1.0 - weight_a], gradients)


@dataclass(frozen=True, eq=False)
class AdaptiveTerms:
    """Explicit full-system energy ledger; force arrays follow atom_ids.

    corrections contain both partition offsets. correction contains the full
    blended correction including switching forces; total adds baseline once.
    No virial, dynamics update, or electronic-state history is supplied.
    """

    model: ClassVar[str] = "diagnostic-conservative-two-partition-mechanical"
    atom_ids: tuple[int, ...]
    partition_ids: tuple[str, str]
    qm_atom_ids: tuple[tuple[int, ...], tuple[int, ...]]
    weights: np.ndarray
    weight_gradients_per_angstrom: np.ndarray
    coordinate_angstrom: float
    partition_energies_ev: tuple[float, float]
    energy_offsets_ev: tuple[float, float]
    offset_sources: tuple[str | None, str | None]
    baseline: Evaluation
    corrections: tuple[Evaluation, Evaluation]
    weighted_correction_forces: np.ndarray
    switching_forces: np.ndarray
    switching_force_max_ev_per_angstrom: float
    correction: Evaluation
    total: Evaluation


class ConservativeMechanicalMixer:
    """Blend exactly two same-composition, same-state complete-group corrections.

    The shared full_low receives sorted global atom IDs; each isolated provider
    receives sorted QM IDs. The caller must bind per-atom backend parameters to
    those orders. The calculator objects' full_low members are not used.
    """

    def __init__(self, full_low, *, groups, partitions, switch):
        groups = tuple(groups)
        if not groups or any(not isinstance(group, FixedGroup) for group in groups):
            raise ValueError("Explicit FixedGroup declarations are required")
        self._coverage = FixedGroupSelector(groups, tuple(group.group_id for group in groups))
        self.groups = self._coverage.groups
        self.partitions = tuple(partitions)
        if len(self.partitions) != 2 or any(
            not isinstance(p, MechanicalPartition) for p in self.partitions
        ):
            raise ValueError("Exactly two MechanicalPartition declarations are required")
        first, second = self.partitions
        if first.partition_id == second.partition_id:
            raise ValueError("Partition IDs must be distinct")
        if not isinstance(switch, QuinticDistanceExchange):
            raise TypeError("The mixer requires an explicit QuinticDistanceExchange")
        self.switch = switch
        self.full_low = full_low
        by_group = {group.group_id: group for group in self.groups}
        by_atom = {atom: group.group_id for group in self.groups for atom in group.atom_ids}
        try:
            center_group, first_group, second_group = (
                by_atom[atom]
                for atom in (switch.center_atom_id, switch.first_atom_id, switch.second_atom_id)
            )
        except KeyError as exc:
            raise ValueError("Every exchange anchor must belong to a declared group") from exc
        if len({center_group, first_group, second_group}) != 3:
            raise ValueError("Exchange anchors must belong to three distinct complete groups")
        for partition in self.partitions:
            if not set(partition.qm_group_ids) <= set(by_group):
                raise ValueError("Every QM group must be declared")
        groups_a, groups_b = set(first.qm_group_ids), set(second.qm_group_ids)
        if (
            center_group not in groups_a & groups_b
            or groups_a - groups_b != {first_group}
            or groups_b - groups_a != {second_group}
        ):
            raise ValueError(
                "Partitions must retain the center group and exchange only their candidate groups"
            )
        self.qm_atom_ids = tuple(
            tuple(sorted(atom for group in p.qm_group_ids for atom in by_group[group].atom_ids))
            for p in self.partitions
        )
        compositions = [
            Counter(symbol for group in p.qm_group_ids for symbol in by_group[group].symbols)
            for p in self.partitions
        ]
        if compositions[0] != compositions[1]:
            raise ValueError("Both QM partitions must have identical elemental composition")
        for name in (
            "qm_charge",
            "qm_multiplicity",
            "method_identity",
            "energy_reference",
            "electronic_state",
        ):
            if getattr(first, name) != getattr(second, name):
                raise ValueError(f"Both QM partitions must share {name}")
        for name in ("xc", "basis"):
            if getattr(first.calculator.high, name, None) != getattr(
                second.calculator.high, name, None
            ):
                raise ValueError(f"High-provider {name} differs between partitions")
        self._atom_ids = tuple(sorted(by_atom))

    def evaluate_terms(self, geometry: Geometry) -> AdaptiveTerms:
        """Evaluate both corrections and include every derivative of the weights."""
        self._coverage.select(geometry)
        weights = self.switch.evaluate(geometry)
        ordered = geometry.select(self._atom_ids)
        rows = {atom: index for index, atom in enumerate(self._atom_ids)}
        baseline_ordered = _evaluate(self.full_low, ordered)
        baseline = Evaluation(
            baseline_ordered.energy,
            baseline_ordered.forces[[rows[atom] for atom in geometry.atom_ids]],
        )
        lookup = {atom: row for row, atom in enumerate(geometry.atom_ids)}
        corrections = []
        for partition, ids in zip(self.partitions, self.qm_atom_ids):
            partition.validate_provider_state()
            isolated = partition.calculator.correction(geometry.select(ids))
            forces = np.zeros_like(geometry.positions)
            forces[[lookup[atom] for atom in ids]] = isolated.forces
            corrections.append(Evaluation(isolated.energy + partition.energy_offset_ev, forces))
        first, second = corrections
        energy_difference = second.energy - first.energy
        energy = first.energy + weights.weights[1] * energy_difference
        weighted_forces = weights.weights[0] * first.forces + weights.weights[1] * second.forces
        # sum(grad(w)) = 0 removes every shared energy zero, including the full baseline.
        switching_forces = -energy_difference * weights.gradients_per_angstrom[1]
        switching_max = max(math.hypot(*row) for row in switching_forces)
        if not math.isfinite(switching_max):
            raise EvaluationError("Switching force magnitude is outside finite numerical range")
        correction = Evaluation(energy, weighted_forces + switching_forces)
        total = Evaluation(baseline.energy + correction.energy, baseline.forces + correction.forces)
        return AdaptiveTerms(
            geometry.atom_ids,
            tuple(p.partition_id for p in self.partitions),
            self.qm_atom_ids,
            weights.weights,
            weights.gradients_per_angstrom,
            weights.coordinate_angstrom,
            tuple(baseline.energy + item.energy for item in corrections),
            tuple(p.energy_offset_ev for p in self.partitions),
            tuple(p.offset_source for p in self.partitions),
            baseline,
            tuple(corrections),
            _readonly_array(weighted_forces, "Weighted correction forces"),
            _readonly_array(switching_forces, "Switching forces"),
            switching_max,
            correction,
            total,
        )

    def evaluate(self, geometry: Geometry) -> Evaluation:
        return self.evaluate_terms(geometry).total
