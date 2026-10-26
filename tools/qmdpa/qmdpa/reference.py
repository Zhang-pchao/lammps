# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Experimental fixed-region embedding with an explicit Coulomb reference.

E = E_low(full) + E_high_emb(Q; qM) - E_low(Q) - V_ref(Q, M).

The reference is a declared point-charge model, not a decomposition of a
pretrained potential. Consistent forces do not establish absence of physical
double counting. No additional LJ or environment Coulomb energy is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .additive import _embedded_evaluate
from .core import Evaluation, EvaluationError, Geometry, _atom_ids, _evaluate
from .embedding import EmbeddedEvaluation, FixedEmbeddingCharges

# e / (4*pi*epsilon_0) * 1e10: charges in e, distances in Angstrom, energy in eV.
COULOMB_EV_ANGSTROM = 14.39964547842567
COULOMB_SOURCE = (
    "SI elementary charge 1.602176634e-19 C (exact); "
    "CODATA 2018 vacuum permittivity 8.8541878128e-12 F/m; "
    "e/(4*pi*epsilon_0)*1e10, in eV Angstrom"
)


@dataclass(frozen=True, eq=False)
class FixedChargeCrossCoulomb:
    """Untruncated Q--M Coulomb energy and forces for explicit fixed charges.

    Both charge sets are immutable and carry their own provenance. Geometry
    orders must match their charge orders. Gaussian widths, periodic images,
    coordinate-dependent charges, and within-region interactions are absent.
    Coincident sites are rejected even if their charge product is zero.
    """

    qm_charges: FixedEmbeddingCharges
    environment_charges: FixedEmbeddingCharges

    def __post_init__(self):
        for name, charges in (
            ("QM", self.qm_charges),
            ("Environment", self.environment_charges),
        ):
            if not isinstance(charges, FixedEmbeddingCharges):
                raise TypeError(f"{name} charges must be explicit FixedEmbeddingCharges")
            if charges.radii_angstrom is not None:
                raise ValueError("The Coulomb reference supports point charges only")
        if set(self.qm_charges.atom_ids).intersection(self.environment_charges.atom_ids):
            raise ValueError("QM and environment charge IDs must be disjoint")

    def evaluate(self, qm_geometry: Geometry, environment_geometry: Geometry) -> EmbeddedEvaluation:
        """Return the positive reference term, before the calculator subtracts it."""
        if qm_geometry.periodic or environment_geometry.periodic:
            raise ValueError("The Coulomb reference requires nonperiodic geometries")
        if (
            qm_geometry.atom_ids != self.qm_charges.atom_ids
            or environment_geometry.atom_ids != self.environment_charges.atom_ids
        ):
            raise ValueError("Geometry IDs must match the fixed reference charges in order")
        qm_forces = np.zeros_like(qm_geometry.positions)
        environment_forces = np.zeros_like(environment_geometry.positions)
        energy = 0.0
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                for i, q_charge in enumerate(self.qm_charges.charges_e):
                    for j, m_charge in enumerate(self.environment_charges.charges_e):
                        delta = qm_geometry.positions[i] - environment_geometry.positions[j]
                        radius2 = np.dot(delta, delta)
                        if radius2 == 0:
                            raise ValueError("QM and environment charge sites must not coincide")
                        radius = np.sqrt(radius2)
                        pair_energy = COULOMB_EV_ANGSTROM * q_charge * m_charge / radius
                        force = pair_energy * delta / radius2
                        energy += pair_energy
                        qm_forces[i] += force
                        environment_forces[j] -= force
        except (FloatingPointError, OverflowError) as exc:
            raise EvaluationError("Coulomb reference is outside finite numerical range") from exc
        return EmbeddedEvaluation(
            energy,
            qm_geometry.atom_ids,
            environment_geometry.atom_ids,
            qm_forces,
            environment_forces,
            {
                "model": "fixed-point-charge-cross-coulomb",
                "qm_charge_source": self.qm_charges.source,
                "environment_charge_source": self.environment_charges.source,
                "coulomb_ev_angstrom": COULOMB_EV_ANGSTROM,
                "coulomb_source": COULOMB_SOURCE,
                "pair_count": len(qm_geometry.atom_ids) * len(environment_geometry.atom_ids),
            },
        )


@dataclass(frozen=True, eq=False)
class ReferenceTerms:
    """Unsigned constituents and their signed total, with explicit atom orders.

    full_low and total follow atom_ids (the full input order); region_low follows
    qm_atom_ids. Embedded terms contain both configured region orders. The total
    subtracts region_low and reference_cross; the stored constituents do not.
    """

    model: ClassVar[str] = "experimental-reference-subtractive"
    atom_ids: tuple[int, ...]
    qm_atom_ids: tuple[int, ...]
    environment_atom_ids: tuple[int, ...]
    full_low: Evaluation
    region_low: Evaluation
    embedded: EmbeddedEvaluation
    reference_cross: EmbeddedEvaluation
    total: Evaluation


class ReferenceSubtractiveCalculator:
    """Finite fixed-partition reference subtraction for model comparison only.

    The QM and environment orders are fixed at construction. The full low-level
    provider always receives QM IDs followed by environment IDs, independent of
    full input ordering. Elements are fixed on the first evaluation. All terms
    are recomputed; no previous successful forces replace a failed calculation.

    The caller must justify the reference charges relative to both the embedded
    provider and the low-level potential. This class cannot infer a low-level
    electrostatic decomposition, electronic states, or model applicability.
    """

    model = ReferenceTerms.model
    _terms_type = ReferenceTerms

    def __init__(
        self,
        full_low_backend,
        region_low_backend,
        embedded_backend,
        reference_cross_backend,
        *,
        qm_atom_ids,
        environment_atom_ids,
    ):
        qm_ids = _atom_ids(qm_atom_ids)
        environment_ids = _atom_ids(environment_atom_ids)
        if set(qm_ids).intersection(environment_ids):
            raise ValueError("QM and environment atom IDs must be disjoint")
        self._validate_reference(reference_cross_backend, qm_ids, environment_ids)
        self.full_low_backend = full_low_backend
        self.region_low_backend = region_low_backend
        self.embedded_backend = embedded_backend
        self.reference_cross_backend = reference_cross_backend
        self._qm_atom_ids = qm_ids
        self._environment_atom_ids = environment_ids
        self._full_atom_ids = qm_ids + environment_ids
        self._identity = None

    @staticmethod
    def _validate_reference(reference, qm_ids, environment_ids):
        if not isinstance(reference, FixedChargeCrossCoulomb):
            raise TypeError("The reference backend must be FixedChargeCrossCoulomb")
        if (reference.qm_charges.atom_ids != qm_ids
                or reference.environment_charges.atom_ids != environment_ids):
            raise ValueError("Reference charge IDs must match the configured regions in order")

    @staticmethod
    def _validate_boundary(full_geometry):
        if full_geometry.periodic:
            raise ValueError("Reference subtraction requires a nonperiodic full geometry")

    @property
    def qm_atom_ids(self):
        return self._qm_atom_ids

    @property
    def environment_atom_ids(self):
        return self._environment_atom_ids

    def evaluate_terms(self, full_geometry: Geometry) -> ReferenceTerms:
        """Evaluate the four terms once and include both sets of response forces."""
        self._validate_boundary(full_geometry)
        if set(full_geometry.atom_ids) != set(self._full_atom_ids):
            raise ValueError("Fixed regions must cover all input atoms exactly")
        full_ordered = full_geometry.select(self._full_atom_ids)
        if self._identity is not None and self._identity != full_ordered.symbols:
            raise ValueError("Element symbols for fixed atom IDs must stay unchanged")
        self._identity = full_ordered.symbols
        qm = full_geometry.select(self.qm_atom_ids)
        environment = full_geometry.select(self.environment_atom_ids)
        full_result = _evaluate(self.full_low_backend, full_ordered)
        region_result = _evaluate(self.region_low_backend, qm)
        embedded = _embedded_evaluate(self.embedded_backend, qm, environment, "Embedded")
        reference = _embedded_evaluate(
            self.reference_cross_backend, qm, environment, "Reference cross"
        )
        rows = {atom_id: row for row, atom_id in enumerate(self._full_atom_ids)}
        full_low = Evaluation(
            full_result.energy,
            full_result.forces[[rows[atom_id] for atom_id in full_geometry.atom_ids]],
        )
        lookup = {atom_id: row for row, atom_id in enumerate(full_geometry.atom_ids)}
        qrows = [lookup[atom_id] for atom_id in self.qm_atom_ids]
        mrows = [lookup[atom_id] for atom_id in self.environment_atom_ids]
        forces = full_low.forces.copy()
        forces[qrows] += (
            embedded.qm_forces_ev_per_angstrom
            - region_result.forces
            - reference.qm_forces_ev_per_angstrom
        )
        forces[mrows] += (
            embedded.environment_forces_ev_per_angstrom
            - reference.environment_forces_ev_per_angstrom
        )
        total = Evaluation(
            full_low.energy + embedded.energy_ev - region_result.energy - reference.energy_ev,
            forces,
        )
        return self._terms_type(
            full_geometry.atom_ids,
            self.qm_atom_ids,
            self.environment_atom_ids,
            full_low,
            region_result,
            embedded,
            reference,
            total,
        )

    def evaluate(self, full_geometry: Geometry) -> Evaluation:
        """Return the same total assembled by evaluate_terms."""
        return self.evaluate_terms(full_geometry).total
