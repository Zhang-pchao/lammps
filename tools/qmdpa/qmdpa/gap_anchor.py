# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Experimental ground-ME anchor plus an embedded excitation gap.

U_s = U_low(full) + E_0(Q; 0) - U_low(Q) + E_s(Q; v_M) - E_0(Q; v_M).

This model is distinct from additive EE and fixed-charge reference subtraction.
All field derivatives are retained. It does not repair the ground ME baseline,
supply mutual polarization, or qualify the pretrained cross-region interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .additive import _embedded_evaluate
from .core import Evaluation, Geometry, _atom_ids, _evaluate
from .embedding import EmbeddedEvaluation, FixedEmbeddingCharges


def _same_field(left, right):
    if not isinstance(left, FixedEmbeddingCharges) or not isinstance(right, FixedEmbeddingCharges):
        return False
    return (
        left.atom_ids == right.atom_ids
        and left.source == right.source
        and np.array_equal(left.charges_e, right.charges_e)
        and left.radii_angstrom is None
        and right.radii_angstrom is None
    )


@dataclass(frozen=True, eq=False)
class GapAnchorTerms:
    """Five physical terms, the ground baseline, gap correction and total.

    Full and total forces follow atom_ids. Isolated ground and fragment-low
    forces follow qm_atom_ids. Embedded and gap forces explicitly carry both
    region orders. Every returned force array is an immutable snapshot.
    """

    hamiltonian: ClassVar[str] = "finite-electrostatic-gap-anchor"
    atom_ids: tuple[int, ...]
    qm_atom_ids: tuple[int, ...]
    environment_atom_ids: tuple[int, ...]
    full_low: Evaluation
    region_low: Evaluation
    isolated_ground: Evaluation
    embedded_ground: EmbeddedEvaluation
    embedded_selected: EmbeddedEvaluation
    ground_me: Evaluation
    gap: EmbeddedEvaluation
    total: Evaluation


class GapAnchorCalculator:
    """Compose the experimental energy on fixed, nonempty finite regions.

    Both embedded providers must expose the same immutable fixed point charges
    through their charges attribute. The caller must match isolated/embedded
    quantum method, basis, charge and spin; make_rhf_gap_anchor constructs the
    currently supported RHF/TDA combination from one frozen state reference.
    No backend failure is converted into another model or stale result.
    """

    def __init__(
        self,
        full_low_backend,
        region_low_backend,
        isolated_ground_backend,
        embedded_ground_backend,
        embedded_selected_backend,
        *,
        qm_atom_ids,
        environment_atom_ids,
    ):
        qm_ids = _atom_ids(qm_atom_ids)
        environment_ids = _atom_ids(environment_atom_ids)
        if set(qm_ids).intersection(environment_ids):
            raise ValueError("QM and environment atom IDs must be disjoint")
        charges = getattr(embedded_ground_backend, "charges", None)
        selected_charges = getattr(embedded_selected_backend, "charges", None)
        if not _same_field(charges, selected_charges):
            raise ValueError("Ground and selected providers require the same fixed point-charge field")
        if charges.atom_ids != environment_ids:
            raise ValueError("Fixed charges must follow the configured environment ID order")
        self.full_low_backend = full_low_backend
        self.region_low_backend = region_low_backend
        self.isolated_ground_backend = isolated_ground_backend
        self.embedded_ground_backend = embedded_ground_backend
        self.embedded_selected_backend = embedded_selected_backend
        self._charges = charges
        self._qm_atom_ids = qm_ids
        self._environment_atom_ids = environment_ids
        self._full_atom_ids = qm_ids + environment_ids
        self._identity = None

    @property
    def charges(self):
        return self._charges

    @property
    def qm_atom_ids(self):
        return self._qm_atom_ids

    @property
    def environment_atom_ids(self):
        return self._environment_atom_ids

    def _validated_geometries(self, full_geometry):
        """Validate fixed identities before evaluating any physical term."""
        if full_geometry.periodic:
            raise ValueError("Gap anchoring requires a nonperiodic full geometry")
        if set(full_geometry.atom_ids) != set(self._full_atom_ids):
            raise ValueError("Fixed regions must cover all input atoms exactly")
        ordered = full_geometry.select(self._full_atom_ids)
        if self._identity is not None and self._identity != ordered.symbols:
            raise ValueError("Element symbols for fixed atom IDs must stay unchanged")
        if not _same_field(self.charges, getattr(self.embedded_ground_backend, "charges", None)) or not _same_field(
            self.charges, getattr(self.embedded_selected_backend, "charges", None)
        ):
            raise ValueError("The configured fixed charge field changed")
        self._identity = ordered.symbols
        qm = full_geometry.select(self.qm_atom_ids)
        environment = full_geometry.select(self.environment_atom_ids)
        return ordered, qm, environment

    def evaluate_terms(self, full_geometry: Geometry) -> GapAnchorTerms:
        """Return the same total energy and force evaluated by evaluate."""
        ordered, qm, environment = self._validated_geometries(full_geometry)
        full_ordered = _evaluate(self.full_low_backend, ordered)
        region_low = _evaluate(self.region_low_backend, qm)
        isolated = _evaluate(self.isolated_ground_backend, qm)
        ground = _embedded_evaluate(self.embedded_ground_backend, qm, environment, "Embedded ground")
        # Supplying this same provider explicitly selects s=0. Identical terms
        # cancel exactly without a second solve, rather than relying on noise.
        selected = ground if self.embedded_selected_backend is self.embedded_ground_backend else _embedded_evaluate(
            self.embedded_selected_backend, qm, environment, "Embedded selected"
        )
        gap = EmbeddedEvaluation(
            selected.energy_ev - ground.energy_ev,
            self.qm_atom_ids,
            self.environment_atom_ids,
            selected.qm_forces_ev_per_angstrom - ground.qm_forces_ev_per_angstrom,
            selected.environment_forces_ev_per_angstrom - ground.environment_forces_ev_per_angstrom,
            {"hamiltonian": GapAnchorTerms.hamiltonian, "term": "embedded-excitation-gap"},
        )
        rows = {atom: i for i, atom in enumerate(self._full_atom_ids)}
        full = Evaluation(
            full_ordered.energy,
            full_ordered.forces[[rows[atom] for atom in full_geometry.atom_ids]],
        )
        lookup = {atom: i for i, atom in enumerate(full_geometry.atom_ids)}
        qrows = [lookup[atom] for atom in self.qm_atom_ids]
        mrows = [lookup[atom] for atom in self.environment_atom_ids]
        ground_forces = full.forces.copy()
        ground_forces[qrows] += isolated.forces - region_low.forces
        ground_me = Evaluation(full.energy + isolated.energy - region_low.energy, ground_forces)
        total_forces = ground_me.forces.copy()
        total_forces[qrows] += gap.qm_forces_ev_per_angstrom
        total_forces[mrows] += gap.environment_forces_ev_per_angstrom
        total = Evaluation(ground_me.energy + gap.energy_ev, total_forces)
        return GapAnchorTerms(
            full_geometry.atom_ids, self.qm_atom_ids, self.environment_atom_ids,
            full, region_low, isolated, ground, selected, ground_me, gap, total,
        )

    def evaluate(self, full_geometry: Geometry) -> Evaluation:
        return self.evaluate_terms(full_geometry).total


def make_rhf_gap_anchor(
    full_low_backend, region_low_backend, reference, *, reuse_identical_geometry=False
):
    """Build one consistent existing CPU RHF/TDA route from an embedded reference.

    Ground terms use the existing RKS(xc='hf') providers, whose parity with
    official RHF must be checked in the actual-backend qualification. No new
    electronic method, implicit state seed or density approximation is selected.
    """
    from .backends import PySCFBackend
    from .embedding import PySCFElectrostaticBackend
    from .excited import PySCFTDAElectrostaticBackend, TDAReference

    if type(reference) is not TDAReference or reference.environment_geometry is None:
        raise ValueError("Gap anchoring requires a frozen embedded TDA reference")
    charges = reference.charges
    if not isinstance(charges, FixedEmbeddingCharges) or charges.radii_angstrom is not None:
        raise ValueError("The RHF/TDA gap anchor supports explicit fixed point charges")
    if type(reuse_identical_geometry) is not bool:
        raise TypeError("Exact geometry reuse must be explicitly boolean")
    settings = reference.settings
    ground_settings = {
        name: getattr(settings, name)
        for name in ("xc", "basis", "charge", "multiplicity", "conv_tol",
                     "conv_tol_grad", "max_cycle", "threads")
    }
    ground = PySCFBackend(**ground_settings)
    embedded_ground = PySCFElectrostaticBackend(charges, **ground_settings)
    selected = PySCFTDAElectrostaticBackend(
        settings, reference, charges, reuse_identical_geometry=reuse_identical_geometry
    )
    return GapAnchorCalculator(
        full_low_backend, region_low_backend, ground, embedded_ground, selected,
        qm_atom_ids=reference.qm_geometry.atom_ids,
        environment_atom_ids=reference.environment_geometry.atom_ids,
    )


def make_pbe0_gap_anchor(
    full_low_backend, region_low_backend, reference, *, reuse_identical_geometry=False,
    matrix_assembly="operator", reuse_ground_terms=False,
):
    """Build a standalone bounded PBE0 anchor with matched ground/selected terms.

    Ground and selected states share basis, XC, grids, charge, convergence and
    source pins. The fixed-charge field is the same object in both embedded
    terms. Complete-Hamiltonian and native qualification are separate gates.
    """
    from .pbe0_tda import (
        PBE0TDAReference,
        PySCFPBE0ElectrostaticGroundBackend,
        PySCFPBE0GroundBackend,
        PySCFPBE0TDAElectrostaticBackend,
    )

    if type(reference) is not PBE0TDAReference or reference.environment_geometry is None:
        raise ValueError("PBE0 gap anchoring requires a frozen embedded PBE0 reference")
    if type(reuse_identical_geometry) is not bool:
        raise TypeError("Exact geometry reuse must be explicitly boolean")
    if type(reuse_ground_terms) is not bool:
        raise TypeError("reuse_ground_terms must be a boolean")
    settings, charges = reference.settings, reference.charges
    options = {"reuse_identical_geometry": True} if reuse_ground_terms else {}
    ground = PySCFPBE0GroundBackend(settings, **options)
    embedded_ground = PySCFPBE0ElectrostaticGroundBackend(settings, charges, **options)
    selected = PySCFPBE0TDAElectrostaticBackend(
        settings, reference, charges, reuse_identical_geometry=reuse_identical_geometry,
        matrix_assembly=matrix_assembly,
    )
    return GapAnchorCalculator(
        full_low_backend, region_low_backend, ground, embedded_ground, selected,
        qm_atom_ids=reference.qm_geometry.atom_ids,
        environment_atom_ids=reference.environment_geometry.atom_ids,
    )


def make_gpu_rhf_gap_anchor(
    full_low_backend, region_low_backend, reference, *, reuse_identical_geometry=False
):
    """Match conventional GPU RHF ground and TDA terms from one fixed reference.

    Reuse the total-energy calculator and existing complete GPU derivatives.
    This factory supports only the plain finite point-charge anchor, not
    corrected fragments, selected DFT or a fallback to CPU execution.
    """
    from .gpu_backends import GPU4PySCFBackend, GPU4PySCFElectrostaticBackend
    from .gpu_excited import GPU4PySCFTDAElectrostaticBackend, GPUTDAReference

    if type(reference) is not GPUTDAReference or reference.environment_geometry is None:
        raise ValueError("GPU gap anchoring requires a frozen embedded GPU RHF/TDA reference")
    if type(reuse_identical_geometry) is not bool:
        raise TypeError("Exact geometry reuse must be explicitly boolean")
    charges = reference.charges
    if not isinstance(charges, FixedEmbeddingCharges) or charges.radii_angstrom is not None:
        raise ValueError("GPU gap anchoring requires explicit fixed point charges")
    settings = reference.settings
    if settings.charge != 0 or settings.multiplicity != 1:
        raise ValueError("GPU gap anchoring currently requires a neutral singlet")
    ground_settings = {
        name: getattr(settings, name)
        for name in ("xc", "basis", "charge", "multiplicity", "conv_tol",
                     "conv_tol_grad", "max_cycle", "threads")
    }
    ground = GPU4PySCFBackend(**ground_settings)
    embedded_ground = GPU4PySCFElectrostaticBackend(charges, **ground_settings)
    selected = GPU4PySCFTDAElectrostaticBackend(
        settings, reference, charges, reuse_identical_geometry=reuse_identical_geometry
    )
    return GapAnchorCalculator(
        full_low_backend, region_low_backend, ground, embedded_ground, selected,
        qm_atom_ids=reference.qm_geometry.atom_ids,
        environment_atom_ids=reference.environment_geometry.atom_ids,
    )
