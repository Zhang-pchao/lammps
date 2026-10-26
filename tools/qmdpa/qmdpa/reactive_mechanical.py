#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Conservative neutral-water corrections across declared proton topologies."""

from dataclasses import dataclass

import numpy as np

from .adaptive import MechanicalPartition, _real, _text
from .core import Evaluation, EvaluationError, _evaluate
from .molecular_adaptive import _mix_values
from .reactive_water import HAMILTONIAN, ReactiveWaterFamily


@dataclass(frozen=True)
class ReactiveWaterGuard:
    min_oh_angstrom: float
    max_oh_angstrom: float
    min_foreign_oh_angstrom: float

    def __post_init__(self):
        values = [_real(getattr(self, name), name) for name in
                  ("min_oh_angstrom", "max_oh_angstrom", "min_foreign_oh_angstrom")]
        if not 0 < values[0] < values[1] <= values[2]:
            raise ValueError("Reactive guards require 0 < min_OH < max_OH <= min_foreign_OH")
        for name, value in zip(("min_oh_angstrom", "max_oh_angstrom", "min_foreign_oh_angstrom"), values, strict=True):
            object.__setattr__(self, name, value)

    def validate(self, geometry, family):
        family.coverage.select(geometry)
        lookup = {a: i for i, a in enumerate(geometry.atom_ids)}
        channel_by_h = {c.hydrogen_id: c for c in family.channels}
        oxygens = set(family.oxygen.values())

        def distance(a, b):
            return np.linalg.norm(geometry.positions[lookup[a]]-geometry.positions[lookup[b]])

        for group, hydrogens in family.initial_hydrogens.items():
            oxygen = family.oxygen[group]
            if (group not in family.pool
                    and distance(oxygen, family.oxygen[family.seed_group_id]) < family.outer_radius_angstrom):
                raise EvaluationError("An undeclared reactive-pool water entered radial support")
            for h in hydrogens:
                channel = channel_by_h.get(h)
                allowed = ({family.oxygen[channel.donor_group_id], family.oxygen[channel.acceptor_group_id]}
                           if channel is not None else {oxygen})
                if not self.min_oh_angstrom <= min(distance(h, o) for o in allowed) <= self.max_oh_angstrom:
                    raise EvaluationError("A water hydrogen left its declared reactive ownership domain")
                if any(distance(h, o) < self.min_foreign_oh_angstrom for o in oxygens-allowed):
                    raise EvaluationError("A hydrogen approached an undeclared proton-transfer oxygen")


@dataclass(frozen=True)
class ReactiveMechanicalTerms:
    atom_ids: tuple[int, ...]
    partition_ids: tuple[str, ...]
    qm_atom_ids: tuple[tuple[int, ...], ...]
    weights: object
    baseline: Evaluation
    correction: Evaluation
    switching_forces: np.ndarray
    total: Evaluation
    evaluated_fragments: tuple[tuple[int, ...], ...]


class ReactiveWaterMechanicalMixer:
    """Full low-level baseline plus every conservative ownership/subset correction."""

    hamiltonian = HAMILTONIAN

    def __init__(self, full_low, correction_calculator, *, family, guard,
                 reference_delta_ev, reference_source, method_identity, energy_reference):
        if not isinstance(family, ReactiveWaterFamily):
            raise TypeError("Reactive ME requires a complete ReactiveWaterFamily")
        if not isinstance(guard, ReactiveWaterGuard):
            raise TypeError("Reactive ME requires an explicit ReactiveWaterGuard")
        self.family, self.guard, self.full_low = family, guard, full_low
        self.reference_delta_ev = _real(reference_delta_ev, "Water reference difference")
        self.reference_source = _text(reference_source, "Water reference source")
        self.partitions = tuple(MechanicalPartition(
            p.partition_id, p.qm_group_ids, correction_calculator, 0, 1,
            method_identity, energy_reference,
            energy_offset_ev=-(len(p.qm_group_ids)-1)*self.reference_delta_ev,
            offset_source=self.reference_source) for p in family.partitions)

    def evaluate_terms(self, geometry):
        self.guard.validate(geometry, self.family)
        weights = self.family.evaluate(geometry)
        for partition in self.partitions:
            partition.validate_provider_state()
        ordered = geometry.select(self.family.atom_ids)
        baseline_value = _evaluate(self.full_low, ordered)
        ordered_rows = {a: i for i, a in enumerate(self.family.atom_ids)}
        baseline = Evaluation(baseline_value.energy,
                              baseline_value.forces[[ordered_rows[a] for a in geometry.atom_ids]])
        rows = {a: i for i, a in enumerate(geometry.atom_ids)}
        cache, values = {}, []
        for index, (spec, partition) in enumerate(zip(self.family.partitions, self.partitions, strict=True)):
            if weights.weights[index] == 0 and not np.any(weights.gradients_per_angstrom[index]):
                values.append(None)
                continue
            ids = spec.qm_atom_ids
            if ids not in cache:
                fragment = geometry.select(ids)
                if fragment.symbols.count("H") != 2*fragment.symbols.count("O"):
                    raise EvaluationError("A reactive QM fragment lost neutral water stoichiometry")
                value = partition.calculator.correction(fragment)
                embedded = np.zeros_like(geometry.positions)
                embedded[[rows[a] for a in ids]] = value.forces
                cache[ids] = Evaluation(value.energy + partition.energy_offset_ev, embedded)
            values.append(cache[ids])
        correction, switching = _mix_values(weights, values)
        return ReactiveMechanicalTerms(
            geometry.atom_ids, tuple(p.partition_id for p in self.family.partitions),
            tuple(p.qm_atom_ids for p in self.family.partitions), weights, baseline,
            correction, switching, Evaluation(baseline.energy+correction.energy,
                                              baseline.forces+correction.forces), tuple(cache))

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).total
