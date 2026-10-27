# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Bounded experimental PBE0/TDA API; native qualification is separate.

This isolated module shares the state, response-density, complete-force and
exact-reuse machinery. Its source pin and reference identity are independent
of the RHF overlay. A bounded explicit XC quadrature response completes the
PBE0 gradient; every configuration still needs numerical derivative evidence.
"""
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import excited
from .backends import PySCFBackend
from .core import Evaluation, EvaluationError
from .embedding import EmbeddedEvaluation, FixedEmbeddingCharges, PySCFElectrostaticBackend

_SOURCE_SHA256 = {
    "__init__.py": "18d7ca2dba59d89e69256917990b01606af5999a8e22d7adec5edd4565850451",
    "grad/tdrks.py": "d2a83558f53cd2b908b9f2ba3eaf0c63a8b52c70ec3188ecfbb585d6d15311d1",
    "grad/tdrhf.py": "2b7fbc07833ff3f4abc5c4e4de62cd40ba52cbf1d2accf5ed8e41b70e167169f",
    "grad/rks.py": "6ba3b49d5a9e8c7b4c2f286ff3f22d030217ec93335152bf5093ca52a9c4711c",
    "dft/rks.py": "cdab251083ed4255b2404a1d2d2a4c0b3dac43ae2083331820468566cc50c787",
    "dft/numint.py": "8edd57a497cd70a449cdf22b3f7b78b8e6ff2d46e65b404b6f8d21e557ddfe0f",
    "dft/gen_grid.py": "98e2f5549e0d81529cd5001f08e4773aceaa67049848f9cde6208625dccd4d51",
    "dft/libxc.py": "8fcccb199826a26ac9af3a0cb13b2d9f6a2aa1c9acd4e8eb5c91909778a77277",
    "scf/cphf.py": "32e1e48fbb082c47344df21bf1f131ee8193831f1fe641be8c425134c57e636a",
    "scf/hf.py": "7065e082a41cd0a8dc2b353e16271eaa4d4e2427483269a85462b6cf8b4def5e",
    "tdscf/rks.py": "0d84cdbae376302b7659baf12e33eb16d549b8ed55599de5030a1c532890c08b",
    "tdscf/rhf.py": "f824062d647ca0f681f767b5f832af8712ae33e47456b924938780153964266e",
    "gto/basis/def2-svp.dat": "18ef6cea20ffdf12b07e90602653e6b8310aed241a4868064d451168cfefa71e",
    "qmmm/itrf.py": "e23c84f29f19666c8acaea3262fef041492b92e82cf8bd541425cc7cd25a357b",
    "grad/rhf.py": "d30fd1ab2a00997183802da28da775f6a29a6c9705c55b19c04d66a1a575267f",
    "tdscf/_lr_eig.py": "9618660167332b566d4410fc2f00f4a2078b0046f2df05b76819455afb64fc7c"
}


@dataclass(frozen=True)
class PBE0TDASettings(excited.TDASettings):
    """Explicit restricted PBE0 basis/grid pairs, not arbitrary TDDFT."""
    xc: str = "pbe0"
    conv_tol: float = 1e-12
    cphf_conv_tol: float = 1e-12
    tda_solver: str = "dense"
    grid_level: int = 4
    grid_prune: str = "nwchem_prune"
    grid_response: bool = True

    def __post_init__(self):
        if not isinstance(self.xc, str) or self.xc.strip().lower() != "pbe0":
            raise ValueError("This experimental route requires xc='pbe0'")
        object.__setattr__(self, "xc", "pbe0")
        self._validate_common(dimension_limit=2048)
        if ((self.basis, self.grid_level) not in (("def2-svp", 4), ("sto-3g", 1))
                or self.charge != 0 or self.multiplicity != 1
                or self.nstates != 3 or self.tda_solver != "dense"
                or self.tda_dense_max_dimension not in (256, 1024, 2048)
                or type(self.grid_level) is not int
                or self.grid_prune != "nwchem_prune"
                or self.grid_response is not True):
            raise ValueError(
                "PBE0 requires neutral singlet, def2-SVP/grid4 or explicit STO-3G/grid1, NWChem pruning "
                "with grid response, three roots and dense cap256 or explicit "
                "experimental cap1024/cap2048"
            )
        for name, maximum in (("conv_tol", 1e-12), ("conv_tol_grad", 1e-8),
                              ("cphf_conv_tol", 1e-12), ("tda_conv_tol", 1e-10),
                              ("response_residual_tol", 1e-7), ("density_trace_tol", 1e-7)):
            if getattr(self, name) > maximum:
                raise ValueError("PBE0 candidate control is too loose: "+name)


def _source_hashes_for(settings):
    """Keep legacy reference identity while pinning the explicit smaller basis."""
    hashes = dict(_SOURCE_SHA256)
    if settings.basis == "sto-3g":
        hashes["gto/basis/sto-3g.dat"] = (
            "0514de5da5df92e477ce69ef906c81589a310260557bcda84531b144eed00cc3"
        )
    return hashes


# Reviewed PySCF response-hook commit 9b5f7e3b2 changes RHF grad_elec and adds
# solve_cphf. RKS overrides grad_elec and retains its own CPHF call. Both exact
# source profiles are admitted; reference/runtime equality remains mandatory.
_RESPONSE_TDRHF_SHA256 = "67c0ef15985c37a6b74b4dff4ed1eefb4317ecad1fc7e9fd3e1ee1ea82f4a466"


def _verify_pbe0_runtime(runtime, settings):
    if not isinstance(runtime, dict) or set(runtime) != {"pyscf_version", "source_sha256"}:
        raise ValueError("PBE0 reference runtime requires only PySCF version and source hashes")
    legacy = _source_hashes_for(settings)
    response = {**legacy, "grad/tdrhf.py": _RESPONSE_TDRHF_SHA256}
    for expected in (legacy, response):
        if runtime.get("source_sha256") == expected:
            excited._verify_runtime_record(runtime, expected)
            return
    raise EvaluationError("PySCF source hashes differ from both inspected PBE0 gradient contracts")


class PBE0TDAReference(excited.TDAReference):
    """Method-specific immutable state record; RHF records cannot be substituted."""
    _settings_type = PBE0TDASettings
    _method_id = "cpu-pbe0-tda-singlet-fixed-reference"
    _source_hashes = _SOURCE_SHA256

    def _validate_runtime(self, runtime):
        _verify_pbe0_runtime(runtime, self.settings)

    def __post_init__(self):
        super().__post_init__()
        if self.charges is not None and self.charges.radii_angstrom is not None:
            raise ValueError("PBE0 candidate supports fixed point charges only")


def _load_runtime(settings):
    import pyscf
    from pyscf import dft, gto, lib, qmmm, scf
    from pyscf.data import nist
    from pyscf.grad import tdrhf, tdrks
    root = Path(pyscf.__file__).resolve().parent
    record = {"pyscf_version": pyscf.__version__, "source_sha256": {
        name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in _source_hashes_for(settings)}}
    _verify_pbe0_runtime(record, settings)
    if settings.threads is not None:
        lib.num_threads(settings.threads)
    return SimpleNamespace(gto=gto, scf=scf, dft=dft, qmmm=qmmm, tdrhf=tdrhf,
        tdrks=tdrks, record=record, hartree_ev=float(nist.HARTREE2EV),
        bohr_angstrom=float(nist.BOHR))



def _validate_matrix_assembly(value):
    if not isinstance(value, str) or value not in ("operator", "official-get-ab-896"):
        raise ValueError("PBE0 matrix assembly requires operator or official-get-ab-896")
    return value


def _official_pbe0_matrix(td, nocc, nvir):
    """Official complete get_ab with bounded instance-local quadrature storage.

    Every grid point is retained. The original operator still checks the final
    eigenvectors, and the original integrator is restored on success or failure.
    This changes matrix assembly only, not the functional, basis or gradients.
    """
    from copy import copy
    method = td._scf
    if (method.xc != "pbe0" or method.grids.level != 4
            or getattr(method, "with_df", None) is not None or method.nlc):
        raise EvaluationError("Official assembly requires the bounded PBE0 method")
    original = method._numint
    limited = copy(original)
    counts = {"points": 0, "blocks": 0}

    def block_loop(mol, grids, nao=None, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        size = 896 if blksize is None else min(896, blksize)
        for block in original.block_loop(mol, grids, nao, deriv, max_memory,
                                         non0tab=non0tab, blksize=size, buf=buf):
            points = len(block[2])
            if not 0 < points <= 896:
                raise EvaluationError("PBE0 matrix quadrature block exceeds its bound")
            counts["points"] += points
            counts["blocks"] += 1
            yield block

    limited.block_loop = block_loop
    method._numint = limited
    try:
        a, _ = td.get_ab()
    finally:
        method._numint = original
    if counts["points"] != len(method.grids.weights):
        raise EvaluationError("PBE0 matrix quadrature did not cover every grid point")
    return excited._real(a, "Official PBE0 TDA matrix").reshape(
        nocc * nvir, nocc * nvir), {
            "tda_matrix_assembly": "official-get-ab-896",
            "tda_matrix_grid_points": counts["points"],
            "tda_matrix_grid_blocks": counts["blocks"],
        }

def _solve(qm, settings, runtime, environment=None, charges=None, *, matrix_assembly="operator"):
    if charges is not None and charges.radii_angstrom is not None:
        raise ValueError("PBE0 candidate supports fixed point charges only")
    def make_method(molecule):
        method = runtime.dft.RKS(molecule)
        method.xc = settings.xc
        method.grids.level = settings.grid_level
        method.grids.prune = runtime.dft.gen_grid.nwchem_prune
        method.nlc = ""
        return method
    assembly = _validate_matrix_assembly(matrix_assembly)
    execution = {}
    if assembly != "operator":
        execution["dense_solver"] = lambda td, controls: excited._dense_tda(
            td, controls, matrix_builder=_official_pbe0_matrix)
    solved = excited._solve(qm, settings, runtime, environment, charges,
                            method_factory=make_method, **execution)
    solved.tda_diagnostics.update(xc=settings.xc, grid_level=settings.grid_level,
        grid_prune=settings.grid_prune, grid_points=len(solved.method.grids.weights),
        rks_grid_response=settings.grid_response, density_fitting=False, nlc=False)
    return solved


def build_pbe0_tda_reference(qm_geometry, settings, *, initial_root, label,
                             environment_geometry=None, charges=None):
    """Pin one fixed PBE0 state; this does not qualify a native Hamiltonian."""
    return excited._build_reference(qm_geometry, settings,
        reference_type=PBE0TDAReference, runtime_loader=_load_runtime, solver=_solve,
        initial_root=initial_root, label=label,
        environment_geometry=environment_geometry, charges=charges)


class PySCFPBE0TDABackend(excited.PySCFTDABackend):
    """Experimental isolated total PBE0 plus TDA gap and checked derivative."""
    _settings_type = PBE0TDASettings
    _reference_type = PBE0TDAReference
    _method_id = "cpu-pbe0-tda-singlet"
    _energy_definition = "E_PBE0+omega"

    def __init__(self, settings, reference, *, reuse_identical_geometry=False,
                 matrix_assembly="operator"):
        self._matrix_assembly = _validate_matrix_assembly(matrix_assembly)
        self._fixed_matrix_assembly = self._matrix_assembly
        super().__init__(settings, reference,
                         reuse_identical_geometry=reuse_identical_geometry)

    @property
    def matrix_assembly(self):
        return self._matrix_assembly

    def _load_method_runtime(self, settings):
        return _load_runtime(settings)

    def _reuse_key(self, *args):
        if self._matrix_assembly != self._fixed_matrix_assembly:
            raise EvaluationError("PBE0 matrix assembly changed; construct a new provider")
        return super()._reuse_key(*args)

    def _solve_surface(self, qm, settings, runtime, environment, charges):
        if self._matrix_assembly != self._fixed_matrix_assembly:
            raise EvaluationError("PBE0 matrix assembly changed; construct a new provider")
        if self._matrix_assembly == "operator":
            return _solve(qm, settings, runtime, environment, charges)
        return _solve(qm, settings, runtime, environment, charges,
                      matrix_assembly=self._matrix_assembly)

    def _surface_gradient(self, solved, index, settings, runtime):
        captured_grid = {}

        class GridResponseGradient(runtime.tdrks.Gradients):
            def extra_force(self, atom, response):
                if not captured_grid:
                    correction, diagnostics = _pbe0_grid_response(solved.method, response)
                    captured_grid.update(correction=correction, diagnostics=diagnostics)
                return (super().extra_force(atom, response)
                        + captured_grid["correction"][atom])

        try:
            gradient, density, diagnostics = excited._total_gradient(
                solved, index, settings, runtime, gradient_base=GridResponseGradient,
                refine=False,
            )
            if not captured_grid:
                raise EvaluationError("Missing PBE0 quadrature response")
            diagnostics.update(captured_grid["diagnostics"])
            diagnostics.update(
                rks_grid_response=True,
                grid_response_force_assembly=True,
                grid_response_gradient_hartree_bohr=captured_grid["correction"].tolist(),
            )
            return gradient, density, diagnostics
        finally:
            # The temporary gradient class can outlive this call until cyclic
            # collection. Release its closed-over SCF/grid workspace now.
            solved = None
            captured_grid.clear()



class PySCFPBE0TDAElectrostaticBackend(
    excited.PySCFTDAElectrostaticBackend, PySCFPBE0TDABackend
):
    """Same total state with both QM and fixed point-charge feedback forces."""

    def __init__(self, settings, reference, charges, *, reuse_identical_geometry=False,
                 matrix_assembly="operator"):
        assembly = _validate_matrix_assembly(matrix_assembly)
        super().__init__(settings, reference, charges,
                         reuse_identical_geometry=reuse_identical_geometry)
        self._matrix_assembly = self._fixed_matrix_assembly = assembly


def _pbe0_ground_settings(settings):
    if type(settings) is not PBE0TDASettings:
        raise TypeError("Matched PBE0 ground terms require PBE0TDASettings")
    return {name: getattr(settings, name) for name in (
        "xc", "basis", "charge", "multiplicity", "conv_tol", "conv_tol_grad",
        "max_cycle", "threads", "grid_level", "grid_response",
    )}


class _MatchedPBE0GroundMethod:
    def _configure_ground_reuse(self, enabled):
        if type(enabled) is not bool:
            raise TypeError("reuse_identical_geometry must be a boolean")
        self._ground_reuse_enabled = enabled
        self._ground_reuse_settings = self.settings
        self._ground_reuse_runtime = self._runtime
        self._ground_reuse_runtime_record = excited._canonical(self._runtime.record)
        self._ground_reuse_charges = excited._charges_record(getattr(self, "charges", None))
        self._ground_reuse_extra = (
            getattr(self, "nlc_grid_level", None), getattr(self, "include_qm_dipole", False))
        self._ground_reuse_counts = {"requests": 0, "hits": 0, "calculations": 0, "failures": 0}
        self.clear_reuse()

    def clear_reuse(self):
        """Discard the last complete result; never retain a partial calculation."""
        self._last_ground_result = None

    def reuse_statistics(self):
        """Call counts only; a saved call is not a complete workload speedup."""
        return {"enabled": self._ground_reuse_enabled, **self._ground_reuse_counts}

    @staticmethod
    def _detached_ground(result):
        if isinstance(result, EmbeddedEvaluation):
            return EmbeddedEvaluation(result.energy_ev, result.qm_atom_ids,
                result.environment_atom_ids, result.qm_forces_ev_per_angstrom,
                result.environment_forces_ev_per_angstrom, deepcopy(dict(result.diagnostics)))
        return Evaluation(result.energy, result.forces)

    def _ground_reuse_key(self, geometries):
        embedded = hasattr(self, "charges")
        if len(geometries) != (2 if embedded else 1):
            raise ValueError("Ground reuse requires all declared QM and environment geometries")
        qm = geometries[0]
        environment = geometries[1] if embedded else None
        charges = self.charges if embedded else None
        excited._validate_geometry(qm, environment, charges)
        expected = _pbe0_ground_settings(self.settings)
        if (self.settings != self._ground_reuse_settings
                or any(getattr(self, k) != v for k, v in expected.items())
                or getattr(self, "density_fit", None) is not None
                or self._runtime is not self._ground_reuse_runtime
                or excited._canonical(self._runtime.record) != self._ground_reuse_runtime_record
                or excited._charges_record(charges) != self._ground_reuse_charges
                or (getattr(self, "nlc_grid_level", None), getattr(self, "include_qm_dipole", False))
                != self._ground_reuse_extra):
            raise EvaluationError("Ground provider configuration changed; construct a new provider")
        if embedded:
            identity = (qm.atom_ids, qm.symbols, environment.atom_ids, environment.symbols)
            if self._identity is not None and self._identity != identity:
                raise ValueError("QM and environment membership, symbols and order must stay fixed")
        return tuple((g.atom_ids, g.symbols, g.positions.tobytes(),
                      None if g.cell is None else g.cell.tobytes()) for g in geometries)

    def evaluate(self, *geometries):
        self._ground_reuse_counts["requests"] += 1
        try:
            key = None
            if self._ground_reuse_enabled:
                key = self._ground_reuse_key(geometries)
                if self._last_ground_result is not None and self._last_ground_result[0] == key:
                    result = self._detached_ground(self._last_ground_result[1])
                    self._ground_reuse_counts["hits"] += 1
                    return result
            self.clear_reuse()
            self._ground_reuse_counts["calculations"] += 1
            result = super().evaluate(*geometries)
            if self._ground_reuse_enabled:
                self._last_ground_result = (key, self._detached_ground(result))
            return result
        except BaseException:
            self.clear_reuse()
            self._ground_reuse_counts["failures"] += 1
            raise

    def _make_method(self, geometry):
        expected = _pbe0_ground_settings(self.settings)
        if (getattr(self, "density_fit", None) is not None
                or any(getattr(self, key) != value for key, value in expected.items())):
            raise EvaluationError("Matched PBE0 ground settings changed")
        method = super()._make_method(geometry)
        method.grids.prune = self._dft.gen_grid.nwchem_prune
        method.nlc = ""
        if getattr(method, "with_df", None) is not None:
            raise EvaluationError("Matched PBE0 ground terms do not support density fitting")
        return method


class PySCFPBE0GroundBackend(_MatchedPBE0GroundMethod, PySCFBackend):
    """Ground component with exactly the bounded PBE0 TDA energy settings."""

    def __init__(self, settings, *, reuse_identical_geometry=False):
        if type(reuse_identical_geometry) is not bool:
            raise TypeError("reuse_identical_geometry must be a boolean")
        options = _pbe0_ground_settings(settings)
        self.settings = settings
        self._runtime = _load_runtime(settings)
        super().__init__(**options)
        self._configure_ground_reuse(reuse_identical_geometry)


class PySCFPBE0ElectrostaticGroundBackend(
    _MatchedPBE0GroundMethod, PySCFElectrostaticBackend
):
    """Same PBE0 ground state and analytic QM/point-charge environment forces."""

    def __init__(self, settings, charges, *, reuse_identical_geometry=False):
        if type(reuse_identical_geometry) is not bool:
            raise TypeError("reuse_identical_geometry must be a boolean")
        options = _pbe0_ground_settings(settings)
        if not isinstance(charges, FixedEmbeddingCharges) or charges.radii_angstrom is not None:
            raise ValueError("Matched PBE0 ground terms require fixed point charges")
        self.settings = settings
        self._runtime = _load_runtime(settings)
        super().__init__(charges, **options)
        self._configure_ground_reuse(reuse_identical_geometry)


def _pbe0_grid_response(method, response):
    """Atomic quadrature derivative for the bounded restricted PBE0 route.

    The XC Lagrangian uses the converged restricted TDA response densities.
    This includes both partition-weight and grid-coordinate derivatives.
    Each geometry and embedding still requires source-specific numerical gates.
    Returned derivatives are Hartree/Bohr, ordered by QM atom.
    """
    import numpy as np
    from pyscf import dft
    from pyscf.grad import rks

    if (not isinstance(method, dft.rks.RKS) or method.xc.strip().lower() != "pbe0"
            or method.mol.spin != 0 or method.do_nlc()
            or getattr(method, "with_df", None) is not None
            or (method.mol.basis, method.grids.level) not in (("def2-svp", 4), ("sto-3g", 1))
            or method.grids.prune is not dft.gen_grid.nwchem_prune):
        raise ValueError("Grid response requires a supported restricted PBE0 basis/grid pair without DF or NLC")
    molecule = method.mol
    nao = molecule.nao_nr()
    arrays = []
    for key in ("oo0", "dmz1doo", "dmxpy"):
        value = excited._real(response[key], "Grid response "+key)
        if value.shape != (nao, nao):
            raise ValueError("Grid-response density must match the QM basis")
        arrays.append(value)
    if np.max(np.abs(arrays[0]-arrays[0].T)) > 1e-12:
        raise EvaluationError("Ground-state density must be symmetric")
    densities = (2*arrays[0], .5*(arrays[1]+arrays[1].T), arrays[2]+arrays[2].T)
    ni = method._numint
    if ni._xc_type(method.xc) != "GGA":
        raise ValueError("Grid-response candidate requires a GGA XC expression")
    ni.libxc.test_deriv_order(method.xc, 3, raise_error=True)
    correction = np.zeros((molecule.natm, 3))
    point_count = 0
    hessian_indices = ((4, 5, 6), (5, 7, 8), (6, 8, 9))

    def density_and_spatial_derivative(ao, density):
        product = ao[0] @ density
        values = np.empty((4, ao.shape[1]))
        values[0] = np.einsum("gi,gi->g", product, ao[0])
        values[1:] = 2*np.einsum("xgi,gi->xg", ao[1:4], product)
        derivatives = np.empty((3, 4, ao.shape[1]))
        derivatives[:, 0] = values[1:]
        for k in range(3):
            first_product = ao[k+1] @ density
            for j in range(3):
                derivatives[j, k+1] = 2*(
                    np.einsum("gi,gi->g", ao[hessian_indices[j][k]], product)
                    + np.einsum("gi,gi->g", ao[j+1], first_product)
                )
        return values, derivatives

    atom_count = 0
    for atom, (coordinates, weights, weight_derivatives) in enumerate(
        rks.grids_response_cc(method.grids)
    ):
        if atom >= molecule.natm or weight_derivatives.shape != (molecule.natm, 3, len(weights)):
            raise EvaluationError("Unexpected atom ordering or quadrature weight derivatives")
        atom_count += 1
        point_count += len(weights)
        for start in range(0, len(weights), 2048):
            stop = min(start+2048, len(weights))
            ao = molecule.eval_gto("GTOval_sph_deriv2", coordinates[start:stop])
            values = [density_and_spatial_derivative(ao, density) for density in densities]
            (rho, drho), (delta, ddelta), (transition, dtransition) = values
            exc, vxc, fxc, kxc = ni.eval_xc_eff(method.xc, rho, 3, xctype="GGA")
            lagrangian = (exc*rho[0] + np.einsum("xg,xg->g", vxc, delta)
                          + np.einsum("xyg,xg,yg->g", fxc, transition, transition))
            ground_factor = (vxc + np.einsum("xyg,yg->xg", fxc, delta)
                             + np.einsum("xyzg,yg,zg->xg", kxc, transition, transition))
            spatial = (np.einsum("xg,axg->ag", ground_factor, drho)
                       + np.einsum("xg,axg->ag", vxc, ddelta)
                       + 2*np.einsum("xyg,yg,axg->ag", fxc, transition, dtransition))
            if not np.isfinite(lagrangian).all() or not np.isfinite(spatial).all():
                raise EvaluationError("Nonfinite XC grid-response integrand")
            correction += np.einsum("g,axg->ax", lagrangian,
                                    weight_derivatives[:, :, start:stop])
            correction[atom] += spatial @ weights[start:stop]
    if atom_count != molecule.natm or not np.isfinite(correction).all():
        raise EvaluationError("Incomplete XC quadrature response")
    return correction, {
        "grid_response_model": "restricted-pbe0-tda-xc-lagrangian-v1",
        "quadrature_response_points": point_count,
        "weight_response": True,
        "coordinate_response": True,
    }
