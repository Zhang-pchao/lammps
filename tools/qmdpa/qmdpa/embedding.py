# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite, fixed-charge PySCF embedding with forces on both sets of atoms.

This standalone provider returns QM internal energy plus its interaction with
external charges. Environment internal energy and cross-region short-range
interactions are absent. It is not a complete hybrid potential and must not be
inserted into the isolated-region subtraction without an explicit energy model.
No periodic electrostatics, charge derivation, adaptive membership or charge
response is supplied. Oxide interfaces require a separate boundary and charge
provenance design; molecular interface tests do not qualify that application.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .backends import PySCFBackend
from .core import Evaluation, EvaluationError, Geometry, _atom_ids, _readonly_array


def _real_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    return _readonly_array(array, name)


@dataclass(frozen=True, eq=False)
class FixedEmbeddingCharges:
    """Immutable charges in electron units, in the declared environment ID order.

    ``source`` records the caller's explicit provenance; charges are never
    inferred from elements or a pretrained potential. Optional Gaussian radii
    are in Angstrom, with exponent zeta = 1 / radius**2 (not a standard deviation).
    Both charges and radii are independent of all coordinates.
    """

    atom_ids: tuple[int, ...]
    charges_e: np.ndarray
    source: str
    radii_angstrom: np.ndarray | None = None

    def __post_init__(self):
        ids = _atom_ids(self.atom_ids)
        charges = _real_array(self.charges_e, "Embedding charges")
        if charges.shape != (len(ids),):
            raise ValueError("Each environment atom ID requires exactly one fixed charge")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("An explicit embedding charge source is required")
        radii = None
        if self.radii_angstrom is not None:
            radii = _real_array(self.radii_angstrom, "Gaussian charge radii")
            if radii.shape != charges.shape or np.any(radii <= 0):
                raise ValueError("Each Gaussian charge requires one positive finite radius")
        object.__setattr__(self, "atom_ids", ids)
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "radii_angstrom", radii)


@dataclass(frozen=True, eq=False)
class EmbeddedEvaluation:
    """QM-plus-embedding energy and both force arrays, in eV and eV/Angstrom."""

    energy_ev: float
    qm_atom_ids: tuple[int, ...]
    environment_atom_ids: tuple[int, ...]
    qm_forces_ev_per_angstrom: np.ndarray
    environment_forces_ev_per_angstrom: np.ndarray
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        qm_ids = _atom_ids(self.qm_atom_ids)
        environment_ids = _atom_ids(self.environment_atom_ids)
        if set(qm_ids).intersection(environment_ids):
            raise ValueError("QM and environment atom IDs must be disjoint")
        qm = Evaluation(self.energy_ev, self.qm_forces_ev_per_angstrom)
        environment = _real_array(
            self.environment_forces_ev_per_angstrom, "Environment reaction forces"
        )
        if qm.forces.shape != (len(qm_ids), 3) or environment.shape != (len(environment_ids), 3):
            raise ValueError("Embedded force counts must match their declared atom IDs")
        object.__setattr__(self, "energy_ev", qm.energy)
        object.__setattr__(self, "qm_atom_ids", qm_ids)
        object.__setattr__(self, "environment_atom_ids", environment_ids)
        object.__setattr__(self, "qm_forces_ev_per_angstrom", qm.forces)
        object.__setattr__(self, "environment_forces_ev_per_angstrom", environment)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class PySCFElectrostaticBackend(PySCFBackend):
    """Official PySCF fixed-charge embedding for finite, nonperiodic geometries.

    The QM and environment positions are explicit on every call. Atom IDs,
    symbols and their order stay fixed during this provider's lifetime. At
    converged HF/DFT, the MM derivative uses the total density plus the nuclear
    contribution. Coordinate-dependent charges, radii, virtual sites and
    post-HF response are outside this contract. No previous result is reused.
    """

    _engine_name = "PySCF"
    _density_namespace = np

    @staticmethod
    def _to_host(values):
        return np.asarray(values)

    _density_array = _to_host

    def _check_method_device(self, method):
        """CPU execution needs no optional device check."""

    def _check_solved_method(self, method):
        """CPU densities are validated below before constructing reaction forces."""

    def __init__(
        self,
        charges: FixedEmbeddingCharges,
        *,
        nlc_grid_level=None,
        include_qm_dipole=False,
        **qm_settings,
    ):
        if not isinstance(charges, FixedEmbeddingCharges):
            raise ValueError("charges must be explicit FixedEmbeddingCharges")
        if type(include_qm_dipole) is not bool:
            raise ValueError("include_qm_dipole must be true or false")
        self.include_qm_dipole = include_qm_dipole
        super().__init__(nlc_grid_level=nlc_grid_level, **qm_settings)
        try:
            from pyscf import qmmm
        except ImportError as exc:
            raise ImportError(
                "Official PySCF qmmm is required for electrostatic embedding"
            ) from exc
        self.charges = charges
        self._qmmm = qmmm
        self._identity = None

    def _method_diagnostics(self, method):
        """Report the grids actually used by this DFT method."""
        diagnostics = {"grid_response": self.grid_response,
                       "nlc_enabled": bool(method.do_nlc())}
        for name, grid in (("regular", method.grids), ("nlc", method.nlcgrids)):
            diagnostics[f"{name}_grid_level"] = int(grid.level)
            diagnostics[f"{name}_grid_points"] = (
                None if grid.coords is None else int(len(grid.coords))
            )
        return diagnostics

    def evaluate(self, qm_geometry: Geometry, environment_geometry: Geometry) -> EmbeddedEvaluation:
        if qm_geometry.periodic or environment_geometry.periodic:
            raise ValueError("Electrostatic embedding currently requires nonperiodic geometries")
        if environment_geometry.atom_ids != self.charges.atom_ids:
            raise ValueError("Environment atom IDs must match the fixed charges in order")
        if set(qm_geometry.atom_ids).intersection(environment_geometry.atom_ids):
            raise ValueError("QM and environment atom IDs must be disjoint")
        identity = (
            qm_geometry.atom_ids,
            qm_geometry.symbols,
            environment_geometry.atom_ids,
            environment_geometry.symbols,
        )
        if self._identity is not None and identity != self._identity:
            raise ValueError("QM and environment membership, symbols and order must stay fixed")
        # The official nuclear kernels contain explicit 1/r terms, including
        # their Gaussian form, and do not define coincident charge sites.
        separation = qm_geometry.positions[:, None, :] - environment_geometry.positions[None, :, :]
        if np.any(np.all(separation == 0, axis=2)):
            raise ValueError("QM nuclei and external charge sites must not coincide")
        self._identity = identity
        try:
            method = self._make_method(qm_geometry)
            method = self._qmmm.mm_charge(
                method,
                environment_geometry.positions,
                self.charges.charges_e,
                radii=self.charges.radii_angstrom,
                unit="Angstrom",
            )
            self._check_method_device(method)
            energy = method.kernel()
            if not method.converged:
                raise EvaluationError(
                    f"{self._engine_name} SCF did not converge within {self.max_cycle} cycles"
                )
            self._check_solved_method(method)
            gradient_method = self._make_gradient(method)
            qm_gradient = self._to_host(gradient_method.kernel())
            density = self._density_array(method.make_rdm1())
            if density.ndim == 3 and density.shape[0] == 2:
                density = density.sum(axis=0)  # UKS alpha + beta; no extra factor of two.
            if density.ndim != 2 or density.shape[0] != density.shape[1]:
                raise EvaluationError("PySCF returned an invalid total density matrix")
            if not self._density_namespace.isfinite(density).all():
                raise EvaluationError("PySCF returned a nonfinite total density matrix")
            electronic = self._to_host(gradient_method.grad_hcore_mm(density))
            nuclear = self._to_host(gradient_method.grad_nuc_mm())
            expected = environment_geometry.positions.shape
            if electronic.shape != expected or nuclear.shape != expected:
                raise EvaluationError("MM gradient components must match the environment atoms")
            diagnostics = {
                "xc": self.xc,
                "basis": self.basis,
                "charge": self.charge,
                "multiplicity": self.multiplicity,
                "scf_converged": True,
                "conv_tol": float(method.conv_tol),
                "conv_tol_grad": (
                    float(method.conv_tol_grad)
                    if method.conv_tol_grad is not None
                    else float(method.conv_tol) ** 0.5
                ),
                "charge_model": "point" if self.charges.radii_angstrom is None else "gaussian",
                "charge_source": self.charges.source,
            }
            diagnostics.update(self._method_diagnostics(method))
            if self.density_fit is not None:
                diagnostics["density_fit"] = {
                    **self.density_fit, "only_dfj": False, "auxbasis_response": True
                }
            if self.include_qm_dipole:
                # The inherited SCF dipole uses only method.mol (QM nuclei)
                # and this total QM density; external charges are excluded.
                dipole = _real_array(
                    method.dip_moment(dm=density, unit="Debye", origin=np.zeros(3), verbose=0),
                    "QM-only dipole",
                )
                if dipole.shape != (3,):
                    raise EvaluationError("PySCF returned an invalid QM-only dipole")
                diagnostics["qm_dipole_debye"] = tuple(float(x) for x in dipole)
                diagnostics["qm_dipole_origin_bohr"] = (0.0, 0.0, 0.0)
            factor = self._hartree_ev / self._bohr_angstrom
            return EmbeddedEvaluation(
                float(energy) * self._hartree_ev,
                qm_geometry.atom_ids,
                environment_geometry.atom_ids,
                -qm_gradient * factor,
                -(electronic + nuclear) * factor,
                diagnostics,
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"{self._engine_name} electrostatic evaluation failed: {exc}") from exc
