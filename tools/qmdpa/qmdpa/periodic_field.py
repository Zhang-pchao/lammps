# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Fixed neutral Gaussian fields and their explicit periodic charge reference.

This module supplies the reference cross term, not a second electronic solver.
The signed reciprocal cube, physical widths, mean-zero gauge and fixed cell
are model identity. Environment self/pair and QM internal terms are absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import EvaluationError, _atom_ids, _readonly_array
from .embedding import EmbeddedEvaluation, FixedEmbeddingCharges, _real_array

# Pinned ABACUS ModuleBase constants and the qualified abacuslite units boundary.
# These older native constants intentionally differ from current CODATA/ASE.
ABACUS_BOHR_ANGSTROM = 0.5291770
ABACUS_RYDBERG_EV = 13.605698
COULOMB_EV_ANGSTROM = 2.0 * ABACUS_RYDBERG_EV * ABACUS_BOHR_ANGSTROM


@dataclass(frozen=True, eq=False)
class PeriodicGaussianField:
    """Immutable nonempty neutral charge set, fixed orthorhombic cell and cube.

    sigma_angstrom contains physical standard deviations, not PySCF Gaussian
    radii and not an auxiliary Ewald splitting parameter. A zero charge set
    remains valid. The empty-environment endpoint uses the existing ME route.
    """

    atom_ids: tuple[int, ...]
    charges_e: np.ndarray
    sigma_angstrom: np.ndarray
    cell_angstrom: np.ndarray
    kmax: int
    source: str

    def __post_init__(self):
        ids = _atom_ids(self.atom_ids)
        charges = _real_array(self.charges_e, "Periodic environment charges")
        widths = _real_array(self.sigma_angstrom, "Gaussian standard deviations")
        cell = _real_array(self.cell_angstrom, "Fixed cell lengths")
        if charges.shape != (len(ids),) or widths.shape != charges.shape:
            raise ValueError("Each environment identity requires one charge and width")
        if np.any(widths <= 0):
            raise ValueError("Gaussian standard deviations must be positive")
        if cell.shape != (3,) or np.any(cell <= 0):
            raise ValueError("Periodic field requires three positive orthorhombic cell lengths")
        if type(self.kmax) is not int or not 1 <= self.kmax <= 32:
            raise ValueError("Periodic field kmax must be an integer from 1 through 32")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Periodic field requires explicit charge provenance")
        if abs(math.fsum(charges)) > 1e-12:
            raise ValueError("Periodic Gaussian environment must be neutral")
        object.__setattr__(self, "atom_ids", ids)
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "sigma_angstrom", widths)
        object.__setattr__(self, "cell_angstrom", cell)

    def validate_geometries(self, qm, environment):
        expected = np.diag(self.cell_angstrom)
        for geometry in (qm, environment):
            if not geometry.periodic or not np.array_equal(geometry.cell, expected):
                raise ValueError("QM and environment require the identical declared fixed cell")
        if environment.atom_ids != self.atom_ids:
            raise ValueError("Environment atom order differs from the fixed Gaussian charges")
        if set(qm.atom_ids).intersection(environment.atom_ids):
            raise ValueError("Periodic QM and environment atom identities must be disjoint")

    def record(self):
        """Return a detached JSON-compatible field identity."""
        return {
            "atom_ids": list(self.atom_ids), "charges_e": self.charges_e.tolist(),
            "sigma_angstrom": self.sigma_angstrom.tolist(),
            "cell_angstrom": self.cell_angstrom.tolist(), "kmax": self.kmax,
            "source": self.source,
        }


class PeriodicGaussianReference:
    """Neutral fixed QM point charges interacting with the declared Gaussian field."""

    model = "periodic-gaussian-charge-cross-reference"

    def __init__(self, qm_charges, field):
        if not isinstance(qm_charges, FixedEmbeddingCharges):
            raise TypeError("QM reference requires FixedEmbeddingCharges")
        if qm_charges.radii_angstrom is not None:
            raise ValueError("The QM reference uses point charges")
        if not isinstance(field, PeriodicGaussianField):
            raise TypeError("The reference requires a PeriodicGaussianField")
        if abs(math.fsum(qm_charges.charges_e)) > 1e-12:
            raise ValueError("Periodic reference QM charges must be neutral")
        if set(qm_charges.atom_ids).intersection(field.atom_ids):
            raise ValueError("Periodic reference charge identities must be disjoint")
        self.qm_charges = qm_charges
        self.field = field
        cube = np.arange(-field.kmax, field.kmax + 1, dtype=float)
        indices = np.stack(np.meshgrid(cube, cube, cube, indexing="ij"), axis=-1).reshape(-1, 3)
        indices = indices[np.any(indices != 0, axis=1)]
        self._modes = _readonly_array(2.0 * np.pi * indices / field.cell_angstrom,
                                     "Reciprocal vectors")
        square = np.sum(self._modes**2, axis=1)
        self._coefficients = _readonly_array(
            4.0 * np.pi * COULOMB_EV_ANGSTROM / (np.prod(field.cell_angstrom) * square),
            "Reciprocal coefficients",
        )

    def evaluate(self, qm, environment):
        self.field.validate_geometries(qm, environment)
        if qm.atom_ids != self.qm_charges.atom_ids:
            raise ValueError("QM atom order differs from the fixed reference charges")
        qm_forces = np.zeros_like(qm.positions)
        environment_forces = np.zeros_like(environment.positions)
        energy = 0.0
        # Wrap individual images to reduce phase loss for equivalent representations.
        qr = np.mod(qm.positions, self.field.cell_angstrom)
        er = np.mod(environment.positions, self.field.cell_angstrom)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Bounded mode batches avoid a Q x M x G pair tensor.
                for first in range(0, len(self._modes), 1024):
                    modes = self._modes[first:first + 1024]
                    coefficient = self._coefficients[first:first + 1024]
                    square = np.sum(modes**2, axis=1)
                    qfactor = self.qm_charges.charges_e[:, None] * np.exp(-1j * qr @ modes.T)
                    efactor = self.field.charges_e[:, None] * np.exp(
                        -0.5 * self.field.sigma_angstrom[:, None]**2 * square
                        - 1j * er @ modes.T
                    )
                    qsum = qfactor.sum(axis=0)
                    esum = efactor.sum(axis=0)
                    energy += float(np.sum(coefficient * np.real(qsum.conj() * esum)))
                    qm_forces += (coefficient * np.imag(qfactor.conj() * esum)) @ modes
                    environment_forces -= (coefficient * np.imag(qsum.conj() * efactor)) @ modes
        except FloatingPointError as exc:
            raise EvaluationError("Periodic reference exceeds finite numerical range") from exc
        return EmbeddedEvaluation(
            energy, qm.atom_ids, environment.atom_ids, qm_forces, environment_forces,
            {"model": self.model, "field": self.field.record(),
             "qm_charge_source": self.qm_charges.source,
             "coulomb_ev_angstrom": COULOMB_EV_ANGSTROM,
             "zero_mode": "omitted-neutral-3d-mean-zero",
             "environment_self_and_pair": "excluded"},
        )


def native_field_text(field, environment):
    """Serialize one immutable per-request Gaussian field using native units."""
    if environment.atom_ids != field.atom_ids:
        raise ValueError("Environment order differs while preparing native field")
    positions = np.mod(environment.positions, field.cell_angstrom) / ABACUS_BOHR_ANGSTROM
    widths = field.sigma_angstrom / ABACUS_BOHR_ANGSTROM
    rows = ["QMDPA_GAUSSIAN_FIELD_V1", str(len(field.atom_ids))]
    for atom_id, charge, width, xyz in zip(field.atom_ids, field.charges_e, widths, positions):
        numbers = " ".join(f"{value:.17g}" for value in (charge, width, *xyz))
        rows.append(f"{atom_id} {numbers}")
    return "\n".join(rows) + "\n"


def parse_native_field_result(text, field):
    """Read one complete field block and cross-check the native energy ledger."""
    import re

    lines = [line.strip() for line in text.splitlines()]
    begin = "QMDPA_GAUSSIAN_FIELD_V1_RESULT"
    end = "END_QMDPA_GAUSSIAN_FIELD_V1_RESULT"
    if lines.count(begin) != 1 or lines.count(end) != 1:
        raise ValueError("Expected exactly one complete native Gaussian field result")
    first, last = lines.index(begin), lines.index(end)
    rows = lines[first + 1:last]
    if len(rows) != 5 + len(field.atom_ids):
        raise ValueError("Native Gaussian field row count differs")
    metadata = {}
    for line, name in zip(rows[:4], (
            "reciprocal_kmax", "ionic_energy_Ry", "electronic_energy_Ry", "environment_count")):
        values = line.split()
        if len(values) != 2 or values[0] != name:
            raise ValueError("Native Gaussian field metadata is incomplete")
        metadata[name] = values[1]
    if (metadata["reciprocal_kmax"] != str(field.kmax)
            or metadata["environment_count"] != str(len(field.atom_ids))
            or rows[4] != "environment_forces_Ry_per_Bohr"):
        raise ValueError("Native Gaussian field identity or units differ")
    forces = []
    for line, atom_id in zip(rows[5:], field.atom_ids):
        values = line.split()
        if len(values) != 4 or values[0] != str(atom_id):
            raise ValueError("Native Gaussian environment force identities differ")
        forces.append([float(x) for x in values[1:]])
    force = _readonly_array(forces, "Native environment forces")
    force = force * ABACUS_RYDBERG_EV / ABACUS_BOHR_ANGSTROM
    ionic = float(metadata["ionic_energy_Ry"]) * ABACUS_RYDBERG_EV
    electronic = float(metadata["electronic_energy_Ry"]) * ABACUS_RYDBERG_EV
    if not math.isfinite(ionic) or not math.isfinite(electronic):
        raise ValueError("Native Gaussian field energy is not finite")
    ledger = re.findall(r"^\s*E_gaussian_field\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
                        text, re.MULTILINE)
    if len(ledger) != 1 or not math.isfinite(float(ledger[0][1])):
        raise ValueError("Native Gaussian field energy ledger is missing or repeated")
    if abs(float(ledger[0][1]) - ionic - electronic) > 1e-7:
        raise ValueError("Native Gaussian field energy ledger differs from the force block")
    return force, {"field": field.record(), "ionic_field_ev": ionic,
                   "electronic_field_ev": electronic, "zero_mode": "omitted-neutral-3d-mean-zero",
                   "environment_self_and_pair": "excluded"}
