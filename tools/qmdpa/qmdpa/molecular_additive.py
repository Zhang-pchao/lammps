# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite molecular additive EE with a single immutable atom-ID charge table."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .adaptive import _integer, _real, _text
from .additive import AdditiveCalculator
from .core import Evaluation
from .embedding import FixedEmbeddingCharges
from .molecular_adaptive import MolecularWeights, _configure_family, _mix_values


def subset_charges(charges, atom_ids):
    """Select exact ordered fixed charges and optional Gaussian radii by atom ID."""
    if not isinstance(charges, FixedEmbeddingCharges):
        raise TypeError("A full-system FixedEmbeddingCharges table is required")
    ids = tuple(atom_ids)
    lookup = {atom: i for i, atom in enumerate(charges.atom_ids)}
    try:
        rows = [lookup[atom] for atom in ids]
    except KeyError as exc:
        raise ValueError("Every environment atom must occur in the full charge table") from exc
    return FixedEmbeddingCharges(
        ids, charges.charges_e[rows], charges.source,
        None if charges.radii_angstrom is None else charges.radii_angstrom[rows],
    )


@dataclass(frozen=True, eq=False)
class AdditivePartition:
    """One complete fixed-partition additive Hamiltonian and its declared reference."""

    partition_id: str
    qm_group_ids: tuple[str, ...]
    calculator: AdditiveCalculator
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
        if not isinstance(self.calculator, AdditiveCalculator):
            raise TypeError("An additive partition requires an AdditiveCalculator")
        _text(self.method_identity, "method_identity")
        _text(self.energy_reference, "energy_reference")
        offset = _real(self.energy_offset_ev, "energy_offset_ev")
        if offset != 0 or self.offset_source is not None:
            _text(self.offset_source, "A declared energy offset source")
        if self.electronic_state != "ground":
            raise ValueError("Only declared ground-state partitions are supported")
        object.__setattr__(self, "qm_group_ids", tuple(sorted(groups)))
        object.__setattr__(self, "qm_charge", _integer(self.qm_charge, "QM charge"))
        object.__setattr__(self, "qm_multiplicity", _integer(self.qm_multiplicity, "QM multiplicity", 1))
        object.__setattr__(self, "energy_offset_ev", offset)
        self.validate_provider_state()

    def validate_provider_state(self):
        high = self.calculator.embedded_backend
        if (_integer(getattr(high, "charge", None), "High-provider charge") != self.qm_charge
                or _integer(getattr(high, "multiplicity", None), "High-provider multiplicity", 1)
                != self.qm_multiplicity):
            raise ValueError("High-provider charge/multiplicity differs from its partition")
        charges = getattr(high, "charges", None)
        if not isinstance(charges, FixedEmbeddingCharges):
            raise ValueError("Each embedded provider must expose its fixed charges")
        if charges.atom_ids != self.calculator.environment_atom_ids:
            raise ValueError("Provider charge IDs must match its environment in order")
        state = getattr(self.calculator.environment_backend, "charge_spin", None)
        if state is not None:
            values = np.asarray(state)
            if (values.shape != (2,) or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all()
                    or not math.isclose(math.fsum(charges.charges_e), float(values[0]),
                                        rel_tol=0, abs_tol=1e-8)):
                raise ValueError("Total fixed environment charge must match the DPA charge")


@dataclass(frozen=True, eq=False)
class MolecularAdditiveTerms:
    """Per-partition complete totals and their full conservative weighted sum."""

    atom_ids: tuple[int, ...]
    partition_ids: tuple[str, ...]
    qm_atom_ids: tuple[tuple[int, ...], ...]
    environment_atom_ids: tuple[tuple[int, ...], ...]
    weights: MolecularWeights
    active_partition_ids: tuple[str, ...]
    partition_totals: tuple[Evaluation | None, ...]
    switching_forces: np.ndarray
    total: Evaluation


def validate_partition_charges(partition, charges, qm_ids, env_ids):
    """Validate an exact fixed partition before calling any physical provider."""
    partition.validate_provider_state()
    calculator = partition.calculator
    if calculator.qm_atom_ids != qm_ids or calculator.environment_atom_ids != env_ids:
        raise ValueError("Each additive partition must use its exact ordered complete regions")
    expected = subset_charges(charges, env_ids)
    actual = calculator.embedded_backend.charges
    if (actual.source != expected.source
            or not np.array_equal(actual.charges_e, expected.charges_e)
            or not np.array_equal(actual.radii_angstrom, expected.radii_angstrom)):
        raise ValueError("Every partition must use the same fixed atom-ID charge table")


class MolecularAdditiveMixer:
    """Blend full additive energies; no full-DPA baseline or extra Coulomb term.

    Each partition uses the exact environment subset of one fixed charge table.
    The complete same-composition family and ground reference are fixed. This
    is an explicitly weighted potential, not a self-consistent fractional QM
    region, variable occupancy or a correspondence between excited states.
    """

    hamiltonian = "finite-molecular-additive"

    def __init__(self, *, groups, core_group_ids, partitions, switch, charges):
        _configure_family(self, groups, core_group_ids, partitions, switch,
                          AdditivePartition, "embedded_backend")
        if not isinstance(charges, FixedEmbeddingCharges) or set(charges.atom_ids) != set(self._atom_ids):
            raise ValueError("The fixed charge table must cover every declared atom exactly")
        self.charges = subset_charges(charges, self._atom_ids)
        self.environment_atom_ids = tuple(
            tuple(a for a in self._atom_ids if a not in ids) for ids in self.qm_atom_ids)
        self._validate_partitions()

    def _validate_partitions(self):
        for partition, qm_ids, env_ids in zip(self.partitions, self.qm_atom_ids,
                                               self.environment_atom_ids, strict=True):
            validate_partition_charges(partition, self.charges, qm_ids, env_ids)

    def evaluate_terms(self, geometry):
        self._coverage.select(geometry)
        weights = self.switch.evaluate(geometry)
        # Validate all states and charge subsets before any scientific work.
        self._validate_partitions()
        totals = []
        for i, partition in enumerate(self.partitions):
            if weights.weights[i] == 0 and not np.any(weights.gradients_per_angstrom[i]):
                totals.append(None)
                continue
            value = partition.calculator.evaluate(geometry)
            totals.append(Evaluation(value.energy + partition.energy_offset_ev, value.forces))
        total, switching = _mix_values(weights, totals)
        return MolecularAdditiveTerms(
            geometry.atom_ids, tuple(p.partition_id for p in self.partitions),
            self.qm_atom_ids, self.environment_atom_ids, weights,
            tuple(p.partition_id for p, value in zip(self.partitions, totals) if value is not None),
            tuple(totals), switching, total,
        )

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).total
