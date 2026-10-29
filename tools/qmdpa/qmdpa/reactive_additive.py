#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Complete additive EE totals on a declared conservative proton family."""

import math

import numpy as np

from .adaptive import _real, _text
from .additive import AdditiveCalculator
from .core import Evaluation
from .embedding import FixedEmbeddingCharges
from .molecular_adaptive import _mix_values
from .molecular_additive import (
    AdditivePartition, MolecularAdditiveTerms, subset_charges, validate_partition_charges,
)
from .reactive_mechanical import ReactiveWaterGuard
from .reactive_water import ADDITIVE_HAMILTONIAN, ReactiveWaterFamily


class ReactiveWaterAdditiveMixer:
    """Mix environment DPA, embedded QM and cross terms, with full derivatives.

    An immutable atom-ID charge table supplies each environment subset. Every
    DPA-only group is a complete neutral water, including after reassignment.
    At least one permanent environment water is required. No full-DPA baseline
    is added. Identical QM/environment domains share a calculator and are
    evaluated once per call, with no reuse across geometries.
    """

    hamiltonian = ADDITIVE_HAMILTONIAN

    def __init__(self, low, embedded_factory, cross, *, family, guard, charges,
                 reference_delta_ev, reference_source, method_identity, energy_reference):
        if not isinstance(family, ReactiveWaterFamily):
            raise TypeError("Reactive EE requires a complete ReactiveWaterFamily")
        if not isinstance(guard, ReactiveWaterGuard):
            raise TypeError("Reactive EE requires an explicit ReactiveWaterGuard")
        if not isinstance(charges, FixedEmbeddingCharges) or set(charges.atom_ids) != set(family.atom_ids):
            raise ValueError("The fixed charge table must cover every declared atom exactly")
        if not callable(embedded_factory):
            raise TypeError("Reactive EE requires an embedded provider factory")
        self.family, self.guard = family, guard
        self.charges = subset_charges(charges, family.atom_ids)
        self.reference_delta_ev = _real(reference_delta_ev, "Water reference difference")
        self.reference_source = _text(reference_source, "Water reference source")
        lookup = dict(zip(self.charges.atom_ids, self.charges.charges_e, strict=True))
        calculators, partitions = {}, []
        self.qm_atom_ids = tuple(p.qm_atom_ids for p in family.partitions)
        self.environment_atom_ids = tuple(
            tuple(a for a in family.atom_ids if a not in ids) for ids in self.qm_atom_ids)
        # Validate the entire charge/domain contract before constructing engines.
        for spec, env_ids in zip(family.partitions, self.environment_atom_ids, strict=True):
            if not env_ids:
                raise ValueError("Reactive additive EE requires a nonempty permanent environment")
            for group, hydrogens in spec.group_hydrogens:
                if group not in spec.qm_group_ids:
                    ids = (family.oxygen[group], *hydrogens)
                    if abs(math.fsum(lookup[a] for a in ids)) > 1e-8:
                        raise ValueError("Each reassigned environment water must have zero fixed charge")
        for spec, env_ids in zip(family.partitions, self.environment_atom_ids, strict=True):
            ids = spec.qm_atom_ids
            if ids not in calculators:
                field = subset_charges(self.charges, env_ids)
                calculators[ids] = AdditiveCalculator(
                    low, embedded_factory(field), cross,
                    qm_atom_ids=ids, environment_atom_ids=env_ids)
            partitions.append(AdditivePartition(
                spec.partition_id, spec.qm_group_ids, calculators[ids], 0, 1,
                method_identity, energy_reference,
                energy_offset_ev=-(len(spec.qm_group_ids)-1)*self.reference_delta_ev,
                offset_source=self.reference_source))
        self.partitions = tuple(partitions)
        self._validate_partitions()

    def _validate_partitions(self):
        for partition, qm_ids, env_ids in zip(
                self.partitions, self.qm_atom_ids, self.environment_atom_ids, strict=True):
            validate_partition_charges(partition, self.charges, qm_ids, env_ids)

    def evaluate_terms(self, geometry):
        self.guard.validate(geometry, self.family)
        weights = self.family.evaluate(geometry)
        self._validate_partitions()
        cache, totals = {}, []
        for i, partition in enumerate(self.partitions):
            if weights.weights[i] == 0 and not np.any(weights.gradients_per_angstrom[i]):
                totals.append(None)
                continue
            ids = self.qm_atom_ids[i]
            if ids not in cache:
                value = partition.calculator.evaluate(geometry)
                cache[ids] = Evaluation(value.energy + partition.energy_offset_ev, value.forces)
            totals.append(cache[ids])
        total, switching = _mix_values(weights, totals)
        return MolecularAdditiveTerms(
            geometry.atom_ids, tuple(p.partition_id for p in self.partitions),
            self.qm_atom_ids, self.environment_atom_ids, weights,
            tuple(p.partition_id for p, value in zip(self.partitions, totals, strict=True)
                  if value is not None), tuple(totals), switching, total)

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).total
