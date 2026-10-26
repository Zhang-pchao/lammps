# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Optional isolated and finite point-charge GPU4PySCF ground providers."""

import numpy as np

from .backends import PySCFBackend, validate_density_fit
from .core import Evaluation, EvaluationError
from .embedding import FixedEmbeddingCharges, PySCFElectrostaticBackend


def validate_gpu_settings(settings):
    """Validate the bounded GPU method without loading optional quantum engines."""
    settings = dict(settings)
    if settings.get("scf_solver") is not None:
        raise ValueError("The diis-newton solver currently requires isolated CPU ground PySCF")
    if settings.get("scf_level_shift_hartree") is not None:
        raise ValueError("Explicit SCF level shifting currently requires the CPU PySCF adapter")
    xc = settings.get("xc", "pbe")
    if not isinstance(xc, str) or xc.strip().lower() not in ("hf", "pbe", "pbe0"):
        raise ValueError("GPU4PySCF adapter supports conventional RHF or density-fitted PBE/PBE0")
    if type(settings.get("multiplicity", 1)) is not int or settings.get("multiplicity", 1) != 1:
        raise ValueError("GPU4PySCF adapter requires a closed-shell singlet")
    if xc.strip().lower() != "hf" and settings.get("grid_response", True) is not True:
        raise ValueError("GPU4PySCF adapter requires complete grid response")
    if settings.get("nlc_grid_level") is not None:
        raise ValueError("This GPU4PySCF adapter does not implement NLC")
    if settings.get("include_qm_dipole", False) is not False:
        raise ValueError("GPU4PySCF embedding does not provide QM dipole diagnostics")
    fitting = validate_density_fit(settings.get("density_fit"))
    if xc.strip().lower() == "hf":
        if fitting is not None:
            raise ValueError("GPU4PySCF RHF requires conventional integrals without density_fit")
    elif fitting is None:
        raise ValueError("GPU4PySCF adapter requires explicit density_fit.auxbasis")
    return {**settings, "xc": xc.strip().lower(), "density_fit": fitting}


class GPU4PySCFBackend(PySCFBackend):
    """Official singlet RHF or density-fitted PBE/PBE0 with complete forces.

    This adapter supports the isolated molecular API and fixed finite native
    mechanical route. It does not implement embedding, excited states, stress
    or multiple devices. RHF uses conventional integrals without a DFT grid.
    PBE/PBE0 require a named auxiliary basis and complete grid/basis response. No CPU fallback occurs.
    The caller supplies a compatible GPU4PySCF/CuPy/CUDA runtime and exactly one
    visible GPU; constructing a provider imports that optional runtime lazily.
    """

    def __init__(self, **settings):
        settings = validate_gpu_settings(settings)
        super().__init__(**settings)
        try:
            import cupy
            import gpu4pyscf  # noqa: F401 - registers official PySCF conversions
        except ImportError as exc:
            raise ImportError("GPU4PySCF requires compatible gpu4pyscf, CuPy and CUDA libraries") from exc
        self._cupy = cupy
        if cupy.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError("GPU4PySCF adapter requires exactly one visible CUDA device")

    @staticmethod
    def _require_gpu_object(value, name):
        # Official qmmm/DF wrappers may be generated in pyscf.lib, while their
        # scientific method and gradient ancestors remain GPU implementations.
        if not any(base.__module__.startswith("gpu4pyscf.") for base in type(value).__mro__):
            raise EvaluationError(f"GPU4PySCF {name} returned a CPU or unsupported implementation")

    def _make_method(self, geometry):
        # Recheck mutable settings before every conversion, without changing them.
        validate_gpu_settings({name: getattr(self, name) for name in (
            "xc", "multiplicity", "grid_response", "nlc_grid_level", "density_fit")})
        if self.xc.strip().lower() == "hf":
            from pyscf import scf

            molecule = self._gto.M(
                atom=list(zip(geometry.symbols, geometry.positions.tolist())),
                unit="Angstrom", basis=self.basis, charge=self.charge,
                spin=self.multiplicity - 1, verbose=0,
            )
            method = scf.RHF(molecule)
            method.conv_tol = self.conv_tol
            if self.conv_tol_grad is not None:
                method.conv_tol_grad = self.conv_tol_grad
            method.max_cycle = self.max_cycle
        else:
            method = super()._make_method(geometry)
        method = method.to_gpu()
        self._require_gpu_object(method, "conversion")
        return method

    def _check_method_device(self, method):
        self._require_gpu_object(method, "method")

    def _check_solved_method(self, method):
        if not isinstance(method.mo_coeff, self._cupy.ndarray):
            raise EvaluationError("GPU4PySCF SCF did not return GPU orbital coefficients")

    def _make_gradient(self, method):
        gradient = (method.nuc_grad_method() if self.xc.strip().lower() == "hf"
                    else super()._make_gradient(method))
        self._require_gpu_object(gradient, "gradient")
        return gradient

    def _to_host(self, values):
        return np.asarray(self._cupy.asnumpy(values) if isinstance(values, self._cupy.ndarray)
                          else values)

    def evaluate(self, geometry):
        if geometry.periodic:
            raise ValueError("GPU4PySCF molecular backend requires an isolated geometry")
        try:
            with self._cupy.cuda.Device(0):
                method = self._make_method(geometry)
                energy = method.kernel()
                if not method.converged:
                    raise EvaluationError(
                        f"GPU4PySCF SCF did not converge within {self.max_cycle} cycles"
                    )
                self._check_solved_method(method)
                gradient = self._make_gradient(method)
                raw = gradient.kernel()
                self._cupy.cuda.get_current_stream().synchronize()
                # Official gradients may already return a host array. The SCF
                # and gradient classes, not the output container, fix execution.
                forces = -self._to_host(raw) * (self._hartree_ev / self._bohr_angstrom)
                if forces.shape != geometry.positions.shape:
                    raise EvaluationError("GPU4PySCF gradient count does not match the input atoms")
                return Evaluation(float(energy) * self._hartree_ev, forces)
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"GPU4PySCF evaluation failed: {exc}") from exc


class GPU4PySCFElectrostaticBackend(PySCFElectrostaticBackend, GPU4PySCFBackend):
    """Official finite point-charge GPU embedding with complete reaction forces.

    Share the CPU embedding evaluator's identity, energy and force contracts.
    The GPU density remains on device for electronic charge-site derivatives.
    The native route is finite additive EE. Selected states remain unsupported.
    """

    _engine_name = "GPU4PySCF"
    _to_host = GPU4PySCFBackend._to_host
    _check_method_device = GPU4PySCFBackend._check_method_device
    _check_solved_method = GPU4PySCFBackend._check_solved_method

    def __init__(self, charges, **settings):
        if not isinstance(charges, FixedEmbeddingCharges):
            raise ValueError("charges must be explicit FixedEmbeddingCharges")
        if charges.radii_angstrom is not None:
            raise ValueError("GPU4PySCF embedding currently supports point charges only")
        if settings.get("include_qm_dipole", False) is not False:
            raise ValueError("GPU4PySCF embedding does not provide QM dipole diagnostics")
        super().__init__(charges, **settings)
        try:
            from gpu4pyscf.qmmm import itrf
        except ImportError as exc:
            raise ImportError("GPU4PySCF finite qmmm is required for embedding") from exc
        self._qmmm = itrf
        self._density_namespace = self._cupy

    def _method_diagnostics(self, method):
        if self.xc.strip().lower() != "hf":
            return super()._method_diagnostics(method)
        # A conventional RHF calculation has no DFT quadrature or auxiliary basis.
        return {
            "method": "rhf", "integrals": "conventional",
            "grid_response": None, "nlc_enabled": False,
            "regular_grid_level": None, "regular_grid_points": None,
            "nlc_grid_level": None, "nlc_grid_points": None,
        }

    def _density_array(self, values):
        if not isinstance(values, self._cupy.ndarray):
            raise EvaluationError("GPU4PySCF did not return a GPU density matrix")
        return values

    def evaluate(self, qm_geometry, environment_geometry):
        from dataclasses import replace

        if self.charges.radii_angstrom is not None or self.include_qm_dipole is not False:
            raise ValueError("GPU embedding requires point charges and no QM dipole request")
        with self._cupy.cuda.Device(0):
            result = super().evaluate(qm_geometry, environment_geometry)
            self._cupy.cuda.get_current_stream().synchronize()
        return replace(result, diagnostics={**result.diagnostics,
                       "engine": "gpu4pyscf", "device": "cuda", "cpu_fallback": False})
