# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Optional isolated, source-pinned singlet GPU RHF/TDA state surface.

The official GPU operator, eigensolver and total gradient are used. Shared
CPU-side bookkeeping validates candidate roots and fixed AO transition overlap.
Finite point-charge response is optional and explicit; no CPU fallback occurs.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

from . import excited
from .core import EvaluationError
from .gpu_backends import GPU4PySCFBackend

_GPU_VERSION = "1.8.1"
_CUPY_VERSION = "13.4.1"
_EMBEDDING_SOURCES = {'gpu4pyscf/df/df_jk.py': 'f0dd515dc30446ec3f526df3bd7d296a1f7653689e7f54624ac4b861fee0a506',
 'gpu4pyscf/qmmm/itrf.py': '33dbe9552f5069ced7322a4ab8c331e67040b28d4ed2c7777879fbed33792124',
 'pyscf/qmmm/mm_mole.py': 'c9b444a2876122fc32e4f0df7622e94844643d4da943d3cb9d3be92b13d82665'}
_PYSCF_SOURCES = {'gto/mole.py': 'd3c3f33722a3a2daa948437a22ee53f447b4a7c4128203c94bfe36a37bf4c182',
 'gto/moleintor.py': 'e9546009caa89f8c743c3de224d8c324741017d030c8b973e963f01ade7aef2d',
 'scf/hf.py': '7065e082a41cd0a8dc2b353e16271eaa4d4e2427483269a85462b6cf8b4def5e',
 'tdscf/rhf.py': 'f824062d647ca0f681f767b5f832af8712ae33e47456b924938780153964266e'}
_GPU_SOURCES = {'tdscf/rhf.py': '2eaf8ed0fef764905615f9b19b24f1ec3db25881e29327fd428e6bc97f9eee69', 'tdscf/_lr_eig.py': 'd96dfa53838e20e132252f5ff7b9c7fdae54a344caca1db4de66f0d4df2fb9c3', 'grad/tdrhf.py': '914fcbb4469eb90dab2759f0777e14a595ed242343d42b457308d8f6288b073e', 'grad/rhf.py': '2f57c80c57f14137419602e832d5ac28143cefc936aee27e3160348cb87d9f4d', 'scf/cphf.py': '54b8127820096f4e69ebe6c43037703a81e151ee72277dcb792bb2339a0cd988', 'scf/hf.py': 'c4608fb29bd6b985b17235e972dad68b11d2fca06b38e565c0ee7ebb560275e9', 'lib/cupy_helper.py': 'bb1a644014977ae2e1fb3fdb14580e9efe9619f43502a15f482a56776458c3dd'}


@dataclass(frozen=True)
class GPUTDASettings(excited.TDASettings):
    """Closed-shell conventional RHF/TDA with explicit GPU state identity."""

    device: str = "cuda"

    def __post_init__(self):
        if not isinstance(self.xc, str) or self.xc.strip().lower() != "hf":
            raise ValueError("GPU TDA supports conventional singlet RHF only")
        object.__setattr__(self, "xc", "hf")
        self._validate_common(device="cuda")


def _verify_gpu_runtime(record, *, electrostatic=False):
    expected = {"pyscf_version", "source_sha256", "gpu4pyscf_version",
                "cupy_version", "gpu_source_sha256"}
    if electrostatic:
        expected.add("embedding_source_sha256")
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError("GPU state reference requires complete runtime identity")
    if electrostatic and record["embedding_source_sha256"] != _EMBEDDING_SOURCES:
        raise EvaluationError("GPU embedding runtime differs from the inspected density contract")
    excited._verify_runtime_record(record, _PYSCF_SOURCES)
    if (record["gpu4pyscf_version"] != _GPU_VERSION
            or record["cupy_version"] != _CUPY_VERSION
            or record["gpu_source_sha256"] != _GPU_SOURCES):
        raise EvaluationError("GPU state runtime differs from the inspected gradient contract")


@dataclass(frozen=True, eq=False)
class GPUTDAReference(excited.TDAReference):
    """A GPU reference cannot be substituted for an older CPU reference."""

    _settings_type = GPUTDASettings
    _method_id = "gpu-rhf-tda-singlet-fixed-reference"

    def _validate_runtime(self, runtime):
        if self.charges is not None and self.charges.radii_angstrom is not None:
            raise ValueError("GPU selected-state reference requires finite point charges")
        _verify_gpu_runtime(runtime, electrostatic=self.charges is not None)


def _load_runtime(settings, *, electrostatic=False):
    try:
        import cupy
        import gpu4pyscf
        import pyscf
        from gpu4pyscf.grad import tdrhf
        from pyscf import gto, lib, scf
        from pyscf.data import nist
    except ImportError as exc:
        raise ImportError("GPU TDA requires compatible GPU4PySCF and CuPy") from exc
    if cupy.cuda.runtime.getDeviceCount() != 1:
        raise EvaluationError("GPU TDA requires exactly one visible CUDA device")
    if settings.threads is not None:
        lib.num_threads(settings.threads)
    runtime = SimpleNamespace(
        gto=gto, scf=scf, hartree_ev=float(nist.HARTREE2EV),
        bohr_angstrom=float(nist.BOHR),
    )
    root = Path(gpu4pyscf.__file__).resolve().parent
    cpu_root = Path(pyscf.__file__).resolve().parent
    runtime.record = {
        "pyscf_version": pyscf.__version__,
        "source_sha256": {
            name: hashlib.sha256((cpu_root / name).read_bytes()).hexdigest()
            for name in _PYSCF_SOURCES},
        "gpu4pyscf_version": gpu4pyscf.__version__,
        "cupy_version": cupy.__version__, "gpu_source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in _GPU_SOURCES},
    }
    if electrostatic:
        from gpu4pyscf.qmmm import itrf

        runtime.gpu_qmmm = itrf
        runtime.record["embedding_source_sha256"] = {
            name: hashlib.sha256(
                ((root if name.startswith("gpu4pyscf/") else cpu_root)
                 / name.split("/", 1)[1]).read_bytes()
            ).hexdigest() for name in _EMBEDDING_SOURCES
        }
    _verify_gpu_runtime(runtime.record, electrostatic=electrostatic)
    runtime.cupy = cupy
    runtime.gpu_tdrhf = tdrhf
    return runtime


def _host(runtime, value):
    return np.asarray(runtime.cupy.asnumpy(value)
                      if isinstance(value, runtime.cupy.ndarray) else value)


def _host_view(method, td, runtime):
    """Read-only host view for shared validation; operator action stays on GPU."""
    vind, diagonal = td.gen_vind()

    def action(vectors):
        return _host(runtime, vind(runtime.cupy.asarray(vectors)))

    orbitals = SimpleNamespace(mo_coeff=_host(runtime, method.mo_coeff),
                               mo_occ=_host(runtime, method.mo_occ))
    view = SimpleNamespace(
        _scf=orbitals, gen_vind=lambda: (action, _host(runtime, diagonal)),
        positive_eig_threshold=td.positive_eig_threshold,
        e=None if td.e is None else _host(runtime, td.e),
        xy=None if td.xy is None else [
            (_host(runtime, x), _host(runtime, y)) for x, y in td.xy],
        converged=_host(runtime, td.converged),
    )
    return orbitals, view


def _solve(qm_geometry, settings, runtime, environment_geometry=None, charges=None):
    excited._validate_geometry(qm_geometry, environment_geometry, charges)
    if charges is not None and charges.radii_angstrom is not None:
        raise ValueError("GPU selected-state embedding requires point charges")
    if (charges is not None) != hasattr(runtime, "gpu_qmmm"):
        raise EvaluationError("GPU state runtime and embedding mode differ")
    cp = runtime.cupy
    with cp.cuda.Device(0):
        molecule = excited._make_molecule(qm_geometry, settings, runtime)
        method = runtime.scf.RHF(molecule).to_gpu()
        GPU4PySCFBackend._require_gpu_object(method, "selected SCF")
        method.conv_tol = settings.conv_tol
        method.conv_tol_grad = settings.conv_tol_grad
        method.max_cycle = settings.max_cycle
        if charges is not None:
            method = runtime.gpu_qmmm.mm_charge(
                method, environment_geometry.positions, charges.charges_e, unit="Angstrom"
            )
            GPU4PySCFBackend._require_gpu_object(method, "selected embedding")
        ground = float(method.kernel())
        if not method.converged or not np.isfinite(ground):
            raise EvaluationError("GPU TDA SCF did not converge to a finite energy")
        if not isinstance(method.mo_coeff, cp.ndarray):
            raise EvaluationError("GPU TDA SCF returned CPU orbitals")
        td = method.TDA()
        GPU4PySCFBackend._require_gpu_object(td, "TDA")
        td.singlet = True
        td.nstates = settings.nstates
        td.conv_tol = settings.tda_conv_tol
        td.max_cycle = settings.tda_max_cycle
        if settings.tda_solver == "dense":
            _, view = _host_view(method, td, runtime)

            def diagonalize(matrix):
                values, vectors = cp.linalg.eigh(cp.asarray(matrix))
                return _host(runtime, values), _host(runtime, vectors)

            diagnostics = excited._dense_tda(view, settings, diagonalize=diagonalize)
            td.e = cp.asarray(view.e)
            td.xy = [(cp.asarray(x), y) for x, y in view.xy]
            td.converged = view.converged
        else:
            td.kernel()
            diagnostics = {"tda_solver": "iterative", "tda_lindep": float(td.lindep)}
        if not all(isinstance(x, cp.ndarray) for x, _ in td.xy):
            raise EvaluationError("GPU TDA returned CPU amplitudes")
        orbitals, view = _host_view(method, td, runtime)
        energies, transitions, residuals = excited._candidate_data(
            orbitals, view, settings, runtime
        )
        diagnostics.update(residuals)
        diagnostics.update(engine="gpu4pyscf", device="cuda", cpu_fallback=False)
        cp.cuda.get_current_stream().synchronize()
    return SimpleNamespace(
        molecule=molecule, method=method, td=td, energies=energies,
        transitions=transitions, tda_diagnostics=diagnostics,
        ground_energy_ev=ground * runtime.hartree_ev,
        overlap=excited._real(molecule.intor_symmetric("int1e_ovlp"), "AO overlap"),
    )


def _checked_response(solve, to_host, to_device, settings, diagnostics):
    """Check and, when needed, refine the official CPHF result before use."""
    def checked(fvind, mo_energy, mo_occ, h1, *args, **kwargs):
        if diagnostics:
            raise EvaluationError("Expected exactly one GPU TDA orbital response")
        if args or kwargs.get("s1") is not None or kwargs.get("level_shift", 0) != 0:
            raise EvaluationError("Unsupported GPU TDA response equation")
        def host_solve(action, energies, occupations, right_hand_side):
            pair = solve(fvind, mo_energy, mo_occ, to_device(right_hand_side), **kwargs)
            return to_host(pair[0]), pair[1]

        pair, refinement = excited._refine_cphf(
            host_solve, lambda z: to_host(fvind(to_device(z))),
            to_host(mo_energy), to_host(mo_occ), to_host(h1), settings,
        )
        diagnostics.update(
            refinement,
            cphf_solver="official-gpu-normalized-defect-correction",
            cphf_residual_max=refinement["cphf_residual_after_refinement"],
            cphf_refinement_count=refinement["cphf_refinement_iterations"],
        )
        return to_device(pair[0]), None
    return checked


def _total_gradient(solved, index, settings, runtime, *, environment_density=False):
    """Use an instance-local wrapper around the pinned official gradient code.

    GPU4PySCF 1.8.1 calls cphf.solve directly instead of an instance solve hook.
    Rebinding only this function's globals supplies a checked solver to this
    gradient instance. No package/global monkeypatch or copied physics is used.
    """
    cp = runtime.cupy
    gradient = solved.td.nuc_grad_method()
    GPU4PySCFBackend._require_gpu_object(gradient, "selected gradient")
    original = runtime.gpu_tdrhf.grad_elec
    if getattr(gradient.grad_elec, "__func__", None) is not original:
        raise EvaluationError("Unexpected GPU selected gradient implementation")
    diagnostic = {}
    checked = _checked_response(
        runtime.gpu_tdrhf.cphf.solve, lambda x: _host(runtime, x), cp.asarray,
        settings, diagnostic,
    )
    namespace = {**original.__globals__, "cphf": SimpleNamespace(solve=checked)}
    defaults = original.__defaults__
    if environment_density:
        if (getattr(gradient.solvent_response, "__func__", None)
                is not runtime.gpu_tdrhf.Gradients.solvent_response
                or gradient.solvent_response(None) != 0.0
                or getattr(solved.td, "with_solvent", None) is not None
                or defaults[-1] is not False):
            raise EvaluationError("GPU relaxed-density export requires the inspected zero-response base")
        # The official base response is identically zero. This flag retains its
        # relaxed-density intermediates without adding a continuum model.
        defaults = defaults[:-1] + (True,)
    local = FunctionType(original.__code__, namespace, original.__name__,
                         defaults, original.__closure__)
    local.__kwdefaults__ = original.__kwdefaults__
    gradient.grad_elec = MethodType(local, gradient)
    gradient.cphf_conv_tol = settings.cphf_conv_tol
    gradient.cphf_max_cycle = settings.cphf_max_cycle
    try:
        with cp.cuda.Device(0):
            values = excited._real(_host(runtime, gradient.kernel(state=index + 1)),
                                   "GPU total selected-state gradient")
            cp.cuda.get_current_stream().synchronize()
        if not diagnostic or values.shape != (solved.molecule.natm, 3):
            raise EvaluationError("GPU TDA gradient or orbital-response check is incomplete")
        density = None
        if environment_density:
            density = cp.asarray(solved.method.make_rdm1()) + cp.asarray(gradient._dmz1doo)
            density = (density + density.T) * 0.5
            if density.shape != solved.overlap.shape or not bool(cp.isfinite(density).all()):
                raise EvaluationError("GPU relaxed density has invalid shape or values")
            trace = float(cp.einsum("ij,ji->", density, cp.asarray(solved.overlap)))
            error = abs(trace - solved.molecule.nelectron)
            if not np.isfinite(trace) or error > settings.density_trace_tol:
                raise EvaluationError("GPU relaxed density does not preserve the electron count")
            diagnostic.update(
                density_trace=trace, density_trace_error=error,
                density_trace_tolerance=settings.density_trace_tol,
                relaxed_density="official-total-rhf-tda",
                density_export="base-zero-response-bookkeeping",
            )
        return values, density, dict(diagnostic)
    finally:
        # Remove the temporary bound-method cycle and all workspace owners.
        del gradient.grad_elec
        namespace.clear()


def build_gpu_tda_reference(qm_geometry, settings, *, initial_root, label,
                            environment_geometry=None, charges=None):
    return excited._build_reference(
        qm_geometry, settings, reference_type=GPUTDAReference,
        runtime_loader=lambda settings: _load_runtime(
            settings, electrostatic=charges is not None
        ), solver=_solve, initial_root=initial_root, label=label,
        environment_geometry=environment_geometry, charges=charges,
    )


class GPU4PySCFTDABackend(excited.PySCFTDABackend):
    """Isolated GPU total selected energy/force with fixed reference tracking."""

    _settings_type = GPUTDASettings
    _reference_type = GPUTDAReference
    _method_id = "gpu-rhf-tda-singlet"

    def _load_method_runtime(self, settings):
        return _load_runtime(settings, electrostatic=self._embedded)

    def _solve_surface(self, qm, settings, runtime, environment, charges):
        return _solve(qm, settings, runtime, environment, charges)

    def _surface_gradient(self, solved, index, settings, runtime):
        return _total_gradient(solved, index, settings, runtime,
                               environment_density=self._embedded)

    def _validate_identity(self, qm_geometry, environment_geometry, charges):
        # Enforce immutable identity even when exact-result reuse is disabled.
        self._reuse_key(qm_geometry, environment_geometry, charges, True)
        super()._validate_identity(qm_geometry, environment_geometry, charges)

class GPU4PySCFTDAElectrostaticBackend(
    excited.PySCFTDAElectrostaticBackend, GPU4PySCFTDABackend
):
    """Finite point-charge selected total energy and complete reaction forces."""

    _embedded = True
