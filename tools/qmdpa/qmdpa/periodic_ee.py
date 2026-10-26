# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Fixed-cell Gaussian reference subtraction using the existing total assembler."""

import numpy as np

from .periodic_field import PeriodicGaussianReference
from .reference import ReferenceSubtractiveCalculator, ReferenceTerms


class PeriodicReferenceTerms(ReferenceTerms):
    """Same signed constituents with a separate periodic Hamiltonian identity."""

    model = "periodic-electrostatic-reference-subtractive"


class PeriodicReferenceSubtractiveCalculator(ReferenceSubtractiveCalculator):
    """Complete periodic total with a nonempty neutral Gaussian environment.

    Full and QM-fragment low-level providers receive the same fixed cell.
    The reference supplies both sets of derivatives; no extra native DPA or
    environment Coulomb term is permitted. The empty-environment endpoint is
    explicitly evaluated with the existing same-cell mechanical route.
    """

    model = PeriodicReferenceTerms.model
    _terms_type = PeriodicReferenceTerms

    @staticmethod
    def _validate_reference(reference, qm_ids, environment_ids):
        if not isinstance(reference, PeriodicGaussianReference):
            raise TypeError("Periodic subtraction requires PeriodicGaussianReference")
        if (reference.qm_charges.atom_ids != qm_ids
                or reference.field.atom_ids != environment_ids):
            raise ValueError("Periodic reference charge IDs must match both regions in order")

    def _validate_boundary(self, full_geometry):
        cell = np.diag(self.reference_cross_backend.field.cell_angstrom)
        if not full_geometry.periodic or not np.array_equal(full_geometry.cell, cell):
            raise ValueError("Periodic subtraction requires the identical fixed orthorhombic cell")
