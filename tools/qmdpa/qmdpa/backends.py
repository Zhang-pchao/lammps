# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Optional real DeePMD and PySCF providers for the subtractive calculator."""

from __future__ import annotations

import operator
from pathlib import Path

import numpy as np

from .core import Evaluation, EvaluationError, Geometry, _readonly_array


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


class DeepMDBackend:
    """Evaluate one DeePMD model, preserving the input atom order.

    ``charge_spin`` is the explicit [charge, multiplicity] condition, separate
    from ``fparam``. Frame parameters are a vector. Atomic parameters are
    either one uniform vector or an (atoms, parameters) array in input order.
    A separate provider should be used when full and QM parameter sets differ.

    Explicit no_jit=True disables scripting only for raw PyTorch .pt files.
    It does not enforce deterministic execution or change compiled archives.
    Global runtime controls remain the caller's responsibility.
    """

    def __init__(self, model, *, charge_spin=None, fparam=None, aparam=None, no_jit=False):
        if type(no_jit) is not bool:
            raise ValueError("no_jit must be true or false")
        self.model = str(Path(model))
        if no_jit and Path(self.model).suffix != ".pt":
            raise ValueError("no_jit=True requires a raw PyTorch .pt checkpoint")
        self.no_jit = no_jit
        try:
            from deepmd.infer import DeepPot
        except ImportError as exc:
            raise ImportError("DeePMD-kit is required for the DeePMD backend") from exc
        self._potential = DeepPot(self.model, no_jit=True) if self.no_jit else DeepPot(self.model)
        self.type_map = tuple(self._potential.get_type_map())
        if not self.type_map or len(set(self.type_map)) != len(self.type_map):
            raise ValueError("The model must provide a nonempty, unique element type map")
        self._types = {symbol: index for index, symbol in enumerate(self.type_map)}
        self.charge_spin = None
        if charge_spin is not None:
            self.charge_spin = _readonly_array(charge_spin, "Charge and multiplicity")
            if self.charge_spin.shape != (2,):
                raise ValueError("charge_spin must contain charge and multiplicity")
            if not self._potential.has_chg_spin_ebd():
                raise ValueError("This model does not accept a charge_spin condition")
        elif self._potential.has_chg_spin_ebd():
            raise ValueError("An explicit charge_spin condition is required for this model")
        dimension = self._potential.get_dim_fparam()
        self.fparam = None if fparam is None else _readonly_array(fparam, "Frame parameters")
        if self.fparam is not None and (dimension == 0 or self.fparam.shape != (dimension,)):
            raise ValueError(f"fparam must be a vector of the model's {dimension} parameters")
        if self.fparam is None and dimension and not self._potential.has_default_fparam():
            raise ValueError("The model requires explicit frame parameters")
        dimension = self._potential.get_dim_aparam()
        self.aparam = None if aparam is None else _readonly_array(aparam, "Atomic parameters")
        if self.aparam is not None and (
            dimension == 0 or self.aparam.ndim not in (1, 2) or self.aparam.shape[-1] != dimension
        ):
            raise ValueError(f"aparam must have {dimension} parameters per atom")
        if self.aparam is None and dimension:
            raise ValueError("The model requires explicit atomic parameters")

    def evaluate(self, geometry: Geometry) -> Evaluation:
        try:
            atom_types = [self._types[symbol] for symbol in geometry.symbols]
        except KeyError as exc:
            raise ValueError(f"Element {exc.args[0]} is not present in the model type map") from exc
        if self.aparam is not None and self.aparam.ndim == 2:
            if self.aparam.shape[0] != len(geometry.atom_ids):
                raise ValueError("Atomic parameter rows must match the evaluated atoms")
        kwargs = {}
        if self.charge_spin is not None:
            kwargs["charge_spin"] = self.charge_spin.reshape(1, -1)
        if self.fparam is not None:
            kwargs["fparam"] = self.fparam
        if self.aparam is not None:
            kwargs["aparam"] = self.aparam
        cell = None if geometry.cell is None else geometry.cell.reshape(1, 9)
        try:
            output = self._potential.eval(
                geometry.positions.reshape(1, -1, 3), cell, atom_types, **kwargs
            )
            energy = np.asarray(output[0])
            forces = np.asarray(output[1])
            if energy.size != 1 or forces.shape != (1, len(geometry.atom_ids), 3):
                raise ValueError("Unexpected DeePMD energy or force shape")
            return Evaluation(float(energy.reshape(-1)[0]), forces[0])
        except Exception as exc:
            raise EvaluationError(f"DeePMD evaluation failed: {exc}") from exc


def validate_density_fit(value):
    """Normalize the explicit ground-state auxiliary basis without an engine import."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"auxbasis"}:
        raise ValueError("density_fit requires exactly one explicit auxbasis name")
    name = value["auxbasis"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("density_fit.auxbasis must be a nonempty basis name")
    # External basis files need independent content identities; this first
    # route accepts named bases from the installed PySCF distribution only.
    import re

    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+*\-]*", name) is None:
        raise ValueError("density_fit.auxbasis must name an installed basis, not a file")
    return {"auxbasis": name}


def validate_scf_level_shift(value):
    """Optional solver control; the reported energy uses the unshifted Fock operator."""
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < 0):
        raise ValueError("scf_level_shift_hartree must be finite and nonnegative or null")
    return float(value)


def validate_scf_solver(value, gradient_tolerance):
    if value is None:
        return None
    if value != "diis-newton":
        raise ValueError("scf_solver must be diis-newton or null")
    if gradient_tolerance is None:
        raise ValueError("The diis-newton solver requires an explicit conv_tol_grad")
    return value


class PySCFBackend:
    """Isolated molecular DFT with checked SCF convergence and analytic forces.

    Multiplicity is converted to PySCF's ``spin = N_alpha - N_beta``.
    A supplied thread count sets PySCF's process-wide thread count once during
    construction. ``grid_response`` explicitly includes the numerical
    integration grid response in DFT gradients by default. No SCF restart or
    previous-frame force fallback is used. ``conv_tol_grad`` optionally controls
    the SCF orbital-gradient stopping criterion independently of energy change;
    None retains the PySCF default. ``nlc_grid_level`` optionally sets the
    nonlocal-correlation grid independently of ``grid_level``; None leaves
    that grid at the PySCF default. Optional ``density_fit={"auxbasis": name}``
    uses official Coulomb/exchange fitting with auxiliary-basis response enabled
    in every force evaluation. It changes the integral approximation and is
    not an exact-reuse optimization; None preserves conventional integrals.
    Optional scf_level_shift_hartree uses the official SCF convergence control;
    it adds no physical energy term and never triggers an automatic retry.
    """

    def __init__(
        self,
        *,
        xc="pbe",
        basis="def2-svp",
        charge=0,
        multiplicity=1,
        conv_tol=1e-9,
        conv_tol_grad=None,
        max_cycle=100,
        grid_level=3,
        nlc_grid_level=None,
        grid_response=True,
        threads=None,
        density_fit=None,
        scf_level_shift_hartree=None,
        scf_solver=None,
    ):
        self.scf_solver = validate_scf_solver(scf_solver, conv_tol_grad)
        self.scf_level_shift_hartree = validate_scf_level_shift(scf_level_shift_hartree)
        self.density_fit = validate_density_fit(density_fit)
        self.charge = _integer(charge, "Charge")
        self.multiplicity = _integer(multiplicity, "Multiplicity", 1)
        self.max_cycle = _integer(max_cycle, "Maximum SCF cycles", 1)
        self.grid_level = _integer(grid_level, "Integration grid level", 0)
        self.nlc_grid_level = (
            None if nlc_grid_level is None else _integer(nlc_grid_level, "NLC grid level", 0)
        )
        if type(grid_response) is not bool:
            raise ValueError("grid_response must be true or false")
        self.grid_response = grid_response
        if not isinstance(xc, str) or not xc.strip():
            raise ValueError("The density functional must be a nonempty string")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("The basis must be a nonempty string")
        self.xc = xc
        self.basis = basis
        if conv_tol_grad is not None and (
            isinstance(conv_tol_grad, bool)
            or not isinstance(conv_tol_grad, (int, float))
            or not np.isfinite(conv_tol_grad)
            or conv_tol_grad <= 0.0
        ):
            raise ValueError("SCF orbital-gradient tolerance must be finite and positive or None")
        self.conv_tol_grad = conv_tol_grad
        self.conv_tol = float(conv_tol)
        if not np.isfinite(self.conv_tol) or self.conv_tol <= 0.0:
            raise ValueError("SCF convergence tolerance must be finite and positive")
        if threads is not None:
            threads = _integer(threads, "Thread count", 1)
        try:
            from pyscf import dft, gto, lib
            from pyscf.data import nist
        except ImportError as exc:
            raise ImportError("PySCF is required for the molecular QM backend") from exc
        self._dft = dft
        self._gto = gto
        self._hartree_ev = nist.HARTREE2EV
        self._bohr_angstrom = nist.BOHR
        self.threads = threads
        if threads is not None:
            lib.num_threads(threads)

    def _make_method(self, geometry: Geometry):
        """Build a fresh molecular SCF object with this provider's settings."""
        molecule = self._gto.M(
            atom=list(zip(geometry.symbols, geometry.positions.tolist())),
            unit="Angstrom",
            basis=self.basis,
            charge=self.charge,
            spin=self.multiplicity - 1,
            verbose=0,
        )
        method = self._dft.RKS(molecule) if molecule.spin == 0 else self._dft.UKS(molecule)
        if self.density_fit is not None:
            method = method.density_fit(
                auxbasis=self.density_fit["auxbasis"], only_dfj=False
            )
        method.xc = self.xc
        method.conv_tol = self.conv_tol
        if self.conv_tol_grad is not None:
            method.conv_tol_grad = self.conv_tol_grad
        method.max_cycle = self.max_cycle
        shift = validate_scf_level_shift(self.scf_level_shift_hartree)
        if shift is not None:
            method.level_shift = shift
        method.grids.level = self.grid_level
        if self.nlc_grid_level is not None:
            method.nlcgrids.level = self.nlc_grid_level
        return method

    def _make_gradient(self, method):
        """Use the same complete gradient controls for isolated and embedded QM."""
        gradient = method.nuc_grad_method()
        gradient.grid_response = self.grid_response
        if self.density_fit is not None:
            if (getattr(method, "with_df", None) is None
                    or not hasattr(gradient, "auxbasis_response")):
                raise EvaluationError(
                    "PySCF density fitting requires an auxiliary-basis response gradient"
                )
            gradient.auxbasis_response = True
        return gradient

    def _solve_method(self, method):
        """Run the declared solver once; optional refinement is always explicit."""
        energy = method.kernel()
        if self.scf_solver == "diis-newton":
            # Official second-order optimization starts from the prescribed DIIS
            # result even if that first stage did not reach convergence. This is
            # one declared two-stage solver, never an exception-triggered retry.
            method = method.newton()
            method.ah_lindep = 1e-20
            # The inner solver tests sqrt(ah_conv_tol). Keep it well below
            # the requested physical gradient to avoid a zero-update stall.
            method.ah_conv_tol = min(1e-24, (.1*self.conv_tol_grad)**2)
            method.ah_start_tol = min(1e-10, .1*self.conv_tol_grad)
            energy = method.kernel(method.mo_coeff, method.mo_occ)
        if not method.converged:
            raise EvaluationError(f"PySCF SCF did not converge within {self.max_cycle} cycles")
        if self.scf_solver == "diis-newton":
            residual = float(np.linalg.norm(method.get_grad(method.mo_coeff, method.mo_occ)))
            if not np.isfinite(residual) or residual > self.conv_tol_grad:
                raise EvaluationError("Final SCF orbital gradient exceeds the declared tolerance")
        return method, energy

    def evaluate(self, geometry: Geometry) -> Evaluation:
        if geometry.periodic:
            raise ValueError("The PySCF molecular backend requires an isolated geometry")
        try:
            method = self._make_method(geometry)
            method, energy = self._solve_method(method)
            gradient_method = self._make_gradient(method)
            gradient = gradient_method.kernel()
            forces = -np.asarray(gradient) * (self._hartree_ev / self._bohr_angstrom)
            if forces.shape != geometry.positions.shape:
                raise EvaluationError("PySCF gradient count does not match the input atoms")
            return Evaluation(float(energy) * self._hartree_ev, forces)
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"PySCF evaluation failed: {exc}") from exc


def make_qm_backend(settings, *, directory):
    """Construct the selected QM provider; optional engines remain lazy imports."""
    options = dict(settings)
    engine = options.pop("engine", "pyscf")
    if engine == "pyscf":
        return PySCFBackend(**options)
    if engine == "abacus":
        from .abacus import AbacusBackend

        return AbacusBackend(**options, directory=directory)
    raise ValueError("Unknown QM engine")
