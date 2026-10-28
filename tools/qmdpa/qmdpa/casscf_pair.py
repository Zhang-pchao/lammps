# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Finite neutral singlet SA-CASSCF(2,2) surfaces and signed derivative coupling.

Two equally weighted states, real orbitals, fixed basis/atom identities, cold
RHF/CASSCF, no embedding, ECP, ETF or active-space changes. Source identities
pin the native NAC orientation established by full-wavefunction derivatives.
"""

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import EvaluationError, Geometry
from .ehrenfest import SurfacePair, _real, correspondence, mapped_pair

NAC_SOURCE_SHA256 = "f20eb4e1d99c89b3a7043273b0b9847ebe910771205d82fdb227a2fbc0e693e2"
_RUNTIME_MODULES = (
    "scf.hf", "mcscf.mc1step", "mcscf.addons", "mcscf.newton_casscf",
    "grad.sacasscf", "grad.lagrange", "nac.sacasscf", "fci.addons",
    "fci.direct_spin1", "fci.spin_op", "gto.mole", "gto.moleintor",
    "lib.linalg_helper", "soscf.ciah", "data.nist",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class CASSCFPairSettings:
    """Explicit solver controls; the active space and spin model are fixed."""

    basis: str = "sto-3g"
    scf_tol: float = 1e-14
    scf_grad_tol: float = 1e-10
    mc_tol: float = 1e-14
    mc_grad_tol: float = 1e-9
    ci_tol: float = 1e-14
    response_atol: float = 1e-12
    response_rtol: float = 1e-11
    max_cycle: int = 100
    response_max_cycle: int = 150
    max_memory_mb: int = 1000
    min_overlap: float = 0.8
    min_margin: float = 0.2
    min_gap_ev: float = 1e-3
    nac_antisymmetry_tol: float = 1e-8

    def __post_init__(self):
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("A nonempty orbital basis name is required")
        for name in ("max_cycle", "response_max_cycle", "max_memory_mb"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in set(asdict(self)) - {"basis", "max_cycle", "response_max_cycle", "max_memory_mb"}:
            value = _real(getattr(self, name), name)
            if value.shape != () or float(value) <= 0:
                raise ValueError(f"{name} must be a positive scalar")
            object.__setattr__(self, name, float(value))
        if self.min_overlap > 1 or self.min_margin > 2:
            raise ValueError("State-overlap limits exceed their allowed ranges")


def _runtime():
    import pyscf
    from pyscf import fci, gto, lib, mcscf, scf
    from pyscf.data import nist
    from pyscf.nac.sacasscf import NonAdiabaticCouplings

    modules = {name: importlib.import_module("pyscf." + name) for name in _RUNTIME_MODULES}
    hashes = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for name, module in modules.items()}
    if pyscf.__version__ != "2.14.0" or hashes["nac.sacasscf"] != NAC_SOURCE_SHA256:
        raise EvaluationError("SA-CASSCF requires the inspected NAC response/orientation implementation")
    lib.num_threads(1)
    return SimpleNamespace(gto=gto, scf=scf, mcscf=mcscf, fci=fci,
                           nac=NonAdiabaticCouplings, hartree_ev=nist.HARTREE2EV,
                           bohr_angstrom=nist.BOHR,
                           record={"pyscf_version": pyscf.__version__, "source_sha256": hashes})


def _molecule(geometry, settings, runtime):
    if not isinstance(geometry, Geometry) or geometry.periodic:
        raise ValueError("SA-CASSCF pair requires a finite Geometry")
    mol = runtime.gto.M(atom=list(zip(geometry.symbols, geometry.positions)),
                        unit="Angstrom", basis=settings.basis, charge=0, spin=0,
                        symmetry=False, verbose=0, max_memory=settings.max_memory_mb)
    if mol.nelectron < 2 or mol.nelectron % 2 or mol.has_ecp():
        raise ValueError("SA-CASSCF pair requires a neutral even-electron all-electron singlet")
    return mol


def _solve(geometry, settings, runtime):
    mol = _molecule(geometry, settings, runtime)
    mf = runtime.scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = settings.scf_tol, settings.scf_grad_tol
    mf.max_cycle = settings.max_cycle
    mf.kernel()
    if not mf.converged:
        raise EvaluationError("Initial RHF did not converge")
    mc = runtime.mcscf.CASSCF(mf, 2, 2).set(natorb=True)
    mc.fix_spin_(ss=0, shift=1)
    mc = mc.state_average([0.5, 0.5])
    mc.conv_tol, mc.conv_tol_grad = settings.mc_tol, settings.mc_grad_tol
    # CIAH checks its residual against sqrt(ah_conv_tol).  Keep that inner
    # gate below the requested orbital gradient to avoid a zero-step stall.
    mc.ah_conv_tol = min(mc.ah_conv_tol, (0.1 * settings.mc_grad_tol)**2)
    mc.fcisolver.conv_tol = settings.ci_tol
    mc.max_cycle_macro = settings.max_cycle
    mc.kernel()
    if not mc.converged or not np.all(np.isfinite(mc.e_states)):
        raise EvaluationError("Two-state SA-CASSCF did not converge")
    spin_squared = [runtime.fci.spin_op.spin_square0(ci, 2, (1, 1))[0] for ci in mc.ci]
    if np.max(np.abs(spin_squared)) > 1e-8:
        raise EvaluationError("The two CASSCF roots are not both singlets")
    if abs(mc.e_states[1] - mc.e_states[0]) * runtime.hartree_ev < settings.min_gap_ev:
        raise EvaluationError("Two-state energy gap is too small for this NAC model")
    return mc


@dataclass(frozen=True, eq=False)
class CASWavefunction:
    """Two real CI vectors and occupied-core/active orbitals at one geometry."""

    geometry: Geometry
    ncore: int
    orbitals: np.ndarray
    ci: np.ndarray

    def __post_init__(self):
        if (not isinstance(self.geometry, Geometry) or self.geometry.periodic
                or type(self.ncore) is not int or self.ncore < 0):
            raise ValueError("Wavefunctions require finite geometry and nonnegative core count")
        mo = _real(self.orbitals, "Core and active orbitals")
        ci = _real(self.ci, "Two-state CI coefficients")
        if mo.ndim != 2 or mo.shape[1] != self.ncore + 2 or mo.shape[0] < mo.shape[1] or ci.shape != (2, 2, 2):
            raise ValueError("Invalid CAS(2,2) orbital or CI shape")
        if np.max(np.abs(ci.reshape(2, 4) @ ci.reshape(2, 4).T - np.eye(2))) > 1e-8:
            raise EvaluationError("CI states are not orthonormal")
        if np.max(np.abs(ci - ci.transpose(0, 2, 1))) > 1e-8:
            raise EvaluationError("CAS(2,2) coefficients are not restricted singlets")
        object.__setattr__(self, "orbitals", mo)
        object.__setattr__(self, "ci", ci)

    @classmethod
    def from_solution(cls, geometry, solution):
        return cls(geometry, int(solution.ncore),
                   solution.mo_coeff[:, :solution.ncore + 2], solution.ci)

    def record(self):
        return {"atom_ids": list(self.geometry.atom_ids), "symbols": list(self.geometry.symbols),
                "positions_angstrom": self.geometry.positions.tolist(), "ncore": self.ncore,
                "orbitals": self.orbitals.tolist(), "ci": self.ci.tolist()}

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or set(record) != {
                "atom_ids", "symbols", "positions_angstrom", "ncore", "orbitals", "ci"}:
            raise ValueError("Wavefunction record has missing or unknown fields")
        geometry = Geometry(record["atom_ids"], record["symbols"], record["positions_angstrom"])
        return cls(geometry, record["ncore"], record["orbitals"], record["ci"])


def wavefunction_overlap(left, right, settings, runtime):
    """Signed full-determinant overlaps, including inactive occupied minors."""
    if (left.ncore != right.ncore or left.geometry.atom_ids != right.geometry.atom_ids
            or left.geometry.symbols != right.geometry.symbols):
        raise ValueError("Wavefunction correspondence requires unchanged atoms and core count")
    mol_left = _molecule(left.geometry, settings, runtime)
    mol_right = _molecule(right.geometry, settings, runtime)
    expected_core = (mol_left.nelectron - 2) // 2
    if expected_core != left.ncore or left.orbitals.shape[0] != mol_left.nao_nr() or right.orbitals.shape[0] != mol_right.nao_nr():
        raise ValueError("Wavefunction basis/electron count does not match its molecular model")
    ao_overlap = runtime.gto.intor_cross("int1e_ovlp", mol_left, mol_right)
    orbital_overlap = left.orbitals.T @ ao_overlap @ right.orbitals
    occupied = [list(range(left.ncore)) + [left.ncore + i] for i in range(2)]
    minors = np.array([[np.linalg.det(orbital_overlap[np.ix_(a, b)])
                        for b in occupied] for a in occupied])
    return _real(np.einsum("iab,ac,bd,jcd->ij", left.ci, minors, minors, right.ci),
                 "Many-electron overlap")


@dataclass(frozen=True, eq=False)
class CASSCFPairReference:
    settings: CASSCFPairSettings
    wavefunction: CASWavefunction
    runtime_json: str

    def __post_init__(self):
        if not isinstance(self.settings, CASSCFPairSettings) or not isinstance(self.wavefunction, CASWavefunction):
            raise TypeError("Reference requires CASSCF settings and a wavefunction")
        record = json.loads(self.runtime_json)
        if (not isinstance(record, dict) or set(record) != {"pyscf_version", "source_sha256"}
                or record["pyscf_version"] != "2.14.0"
                or set(record["source_sha256"]) != set(_RUNTIME_MODULES)
                or record["source_sha256"]["nac.sacasscf"] != NAC_SOURCE_SHA256):
            raise ValueError("Reference runtime does not carry the inspected NAC contract")
        object.__setattr__(self, "runtime_json", _canonical(record))

    def record(self):
        payload = {"schema_version": 1, "model": "finite-sa-casscf22-pair",
                   "settings": asdict(self.settings), "wavefunction": self.wavefunction.record(),
                   "runtime": json.loads(self.runtime_json)}
        return {**payload, "sha256": _digest(payload)}

    @property
    def sha256(self):
        return self.record()["sha256"]

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or set(record) != {
                "schema_version", "model", "settings", "wavefunction", "runtime", "sha256"}:
            raise ValueError("Reference record has missing or unknown fields")
        payload = {k: v for k, v in record.items() if k != "sha256"}
        if (type(record["schema_version"]) is not int or record["schema_version"] != 1
                or record["model"] != "finite-sa-casscf22-pair" or _digest(payload) != record["sha256"]):
            raise ValueError("Reference identity is invalid")
        reference = cls(CASSCFPairSettings(**record["settings"]),
                        CASWavefunction.from_record(record["wavefunction"]), _canonical(record["runtime"]))
        if reference.sha256 != record["sha256"]:
            raise ValueError("Reference record is not canonical")
        return reference


def build_casscf_pair_reference(geometry, settings):
    if not isinstance(settings, CASSCFPairSettings):
        raise TypeError("Explicit CASSCFPairSettings are required")
    runtime = _runtime()
    solved = _solve(geometry, settings, runtime)
    wavefunction = CASWavefunction.from_solution(geometry, solved)
    overlap = wavefunction_overlap(wavefunction, wavefunction, settings, runtime)
    if np.max(np.abs(overlap - np.eye(2))) > 1e-8:
        raise EvaluationError("Reference wavefunctions fail their own overlap identity")
    return CASSCFPairReference(settings, wavefunction, _canonical(runtime.record))


class CASSCFPairProvider:
    """Stateless, signed fixed-reference two-state E/F/NAC provider."""

    def __init__(self, reference):
        if not isinstance(reference, CASSCFPairReference):
            raise TypeError("A prepared CASSCF pair reference is required")
        self.reference = reference
        runtime = _runtime()
        if runtime.record != json.loads(reference.runtime_json):
            raise EvaluationError("CASSCF reference and installed source identities differ")
        self.runtime = runtime
        own_overlap = wavefunction_overlap(reference.wavefunction, reference.wavefunction,
                                           reference.settings, runtime)
        if np.max(np.abs(own_overlap - np.eye(2))) > 1e-8:
            raise EvaluationError("Reference orbital/CI metric is inconsistent")

    def evaluate_pair(self, geometry):
        reference, runtime = self.reference, self.runtime
        if _runtime().record != runtime.record:
            raise EvaluationError("CASSCF runtime identity changed during evaluation")
        if (geometry.atom_ids != reference.wavefunction.geometry.atom_ids
                or geometry.symbols != reference.wavefunction.geometry.symbols):
            raise ValueError("The CASSCF QM atom identities changed")
        settings = reference.settings
        solved = _solve(geometry, settings, runtime)
        wavefunction = CASWavefunction.from_solution(geometry, solved)
        overlap = wavefunction_overlap(reference.wavefunction, wavefunction, settings, runtime)
        roots, signs = correspondence(overlap, minimum_overlap=settings.min_overlap,
                                      minimum_margin=settings.min_margin)

        def response_settings(response):
            response.conv_atol, response.conv_rtol = settings.response_atol, settings.response_rtol
            response.max_cycle = settings.response_max_cycle
            return response

        forces = []
        for state in range(2):
            response = response_settings(solved.nuc_grad_method())
            gradient = response.kernel(state=state)
            if not response.converged:
                raise EvaluationError("CASSCF state gradient response did not converge")
            forces.append(-gradient * runtime.hartree_ev / runtime.bohr_angstrom)
        nac = response_settings(runtime.nac(solved))
        # The inspected native state=(0,1) matches <0|grad 1> by signed
        # many-electron FD; its docstring has the opposite ket/bra ordering.
        d01 = nac.kernel(state=(0, 1), use_etfs=False) / runtime.bohr_angstrom
        if not nac.converged:
            raise EvaluationError("CASSCF derivative-coupling response did not converge")
        d10 = nac.kernel(state=(1, 0), use_etfs=False) / runtime.bohr_angstrom
        if not nac.converged or np.max(np.abs(d01 + d10)) > settings.nac_antisymmetry_tol:
            raise EvaluationError("Native opposite-order derivative couplings disagree")
        pair = SurfacePair(np.asarray(solved.e_states) * runtime.hartree_ev, forces, d01)
        return mapped_pair(pair, roots, signs)
