# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Experimental, fixed-reference CPU singlet RHF/TDA energy and force providers.

The provider factory registers only explicitly gated experimental combinations.
This module returns total excited energy, E_RHF + omega. State selection always uses
the same immutable reference, never the result of the preceding call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from .backends import _integer
from .core import Evaluation, EvaluationError, Geometry, _atom_ids, _readonly_array
from .embedding import EmbeddedEvaluation, FixedEmbeddingCharges

# Internals used by the relaxed-density hook were inspected in this source.
# tdrhf.py includes the project fork instance solve_cphf hook. Its default
# delegates to the unchanged official solver; a version string alone cannot
# establish the hook contract.
_SOURCE_SHA256 = {
    "tdscf/rhf.py": "f824062d647ca0f681f767b5f832af8712ae33e47456b924938780153964266e",
    "tdscf/_lr_eig.py": "9618660167332b566d4410fc2f00f4a2078b0046f2df05b76819455afb64fc7c",
    "grad/tdrhf.py": "67c0ef15985c37a6b74b4dff4ed1eefb4317ecad1fc7e9fd3e1ee1ea82f4a466",
    "grad/rhf.py": "d30fd1ab2a00997183802da28da775f6a29a6c9705c55b19c04d66a1a575267f",
    "scf/cphf.py": "32e1e48fbb082c47344df21bf1f131ee8193831f1fe641be8c425134c57e636a",
    "qmmm/itrf.py": "e23c84f29f19666c8acaea3262fef041492b92e82cf8bd541425cc7cd25a357b",
}
_PYSCF_VERSION = "2.14.0"


def _positive(value, name, *, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive number")
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _real(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in "fiu":
        raise EvaluationError(f"{name} must contain real numbers")
    try:
        return _readonly_array(values, name)
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(record):
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TDASettings:
    """Explicit numerical and state-selection controls for one CPU RHF surface.

    min_gap_ev is the nearest energy gap among the requested candidate roots.
    It cannot exclude a near-degeneracy with a root outside that candidate set.
    No density fitting, frozen orbitals, GPU, DFT, or open-shell route is implied.
    """

    xc: str = "hf"
    basis: str = "def2-svp"
    nstates: int = 3
    min_overlap: float = 0.6
    min_margin: float = 0.1
    min_gap_ev: float = 0.001
    charge: int = 0
    multiplicity: int = 1
    conv_tol: float = 1e-11
    conv_tol_grad: float = 1e-8
    max_cycle: int = 100
    tda_conv_tol: float = 1e-10
    tda_solver: str = "iterative"
    tda_dense_max_dimension: int = 256
    tda_max_cycle: int = 100
    cphf_conv_tol: float = 1e-10
    cphf_max_cycle: int = 100
    response_residual_tol: float = 1e-7
    density_trace_tol: float = 1e-7
    threads: int | None = None
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.xc, str) or self.xc.strip().lower() != "hf":
            raise ValueError(
                "Experimental analytic TDA supports RHF (xc='hf') only. "
                "DFT, NLC and meta-GGA gradients need a separately verified route."
            )
        object.__setattr__(self, "xc", "hf")
        self._validate_common()

    def _validate_common(self, *, dimension_limit=1024, device="cpu"):
        """Shared restricted-state controls; each method keeps its own XC gate."""
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("basis must be a nonempty PySCF basis name")
        if self.tda_solver not in ("iterative", "dense"):
            raise ValueError("tda_solver must be explicitly iterative or dense")
        if self.device != device:
            raise ValueError(f"Experimental TDA requires device={device!r}")
        for name, minimum in (
            ("nstates", 2),
            ("multiplicity", 1),
            ("max_cycle", 1),
            ("tda_max_cycle", 1),
            ("tda_dense_max_dimension", 1),
            ("cphf_max_cycle", 1),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.tda_dense_max_dimension > dimension_limit:
            raise ValueError(f"Dense TDA has a hard dimension cap of {dimension_limit}; the default remains 256")
        object.__setattr__(self, "charge", _integer(self.charge, "charge"))
        if self.multiplicity != 1:
            raise ValueError("Experimental RHF/TDA requires a closed-shell singlet")
        if self.threads is not None:
            object.__setattr__(self, "threads", _integer(self.threads, "threads", 1))
        for name in (
            "min_overlap",
            "min_margin",
            "min_gap_ev",
            "conv_tol",
            "conv_tol_grad",
            "tda_conv_tol",
            "cphf_conv_tol",
            "response_residual_tol",
            "density_trace_tol",
        ):
            maximum = 1.0 if name in ("min_overlap", "min_margin") else None
            object.__setattr__(self, name, _positive(getattr(self, name), name, maximum=maximum))


def _geometry_record(geometry):
    return {
        "atom_ids": list(geometry.atom_ids),
        "symbols": list(geometry.symbols),
        "positions_angstrom": geometry.positions.tolist(),
    }


def _geometry_from_record(record):
    if not isinstance(record, dict) or set(record) != {"atom_ids", "symbols", "positions_angstrom"}:
        raise ValueError("Reference geometry requires IDs, symbols and Angstrom positions")
    return Geometry(record["atom_ids"], record["symbols"], record["positions_angstrom"])


def _charges_record(charges):
    if charges is None:
        return None
    return {
        "atom_ids": list(charges.atom_ids),
        "charges_e": charges.charges_e.tolist(),
        "source": charges.source,
        "radii_angstrom": (
            None if charges.radii_angstrom is None else charges.radii_angstrom.tolist()
        ),
    }


def _charges_from_record(record):
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != {
        "atom_ids",
        "charges_e",
        "source",
        "radii_angstrom",
    }:
        raise ValueError("Reference charges require explicit IDs, values, radii and source")
    return FixedEmbeddingCharges(**record)


def _validate_geometry(qm_geometry, environment_geometry=None, charges=None):
    if not isinstance(qm_geometry, Geometry) or qm_geometry.periodic:
        raise ValueError("TDA requires an explicitly isolated QM Geometry")
    if (environment_geometry is None) != (charges is None):
        raise ValueError("Environment geometry and fixed charges must be supplied together")
    if charges is None:
        return
    if not isinstance(charges, FixedEmbeddingCharges):
        raise TypeError("TDA embedding requires explicit FixedEmbeddingCharges")
    if not isinstance(environment_geometry, Geometry) or environment_geometry.periodic:
        raise ValueError("TDA embedding requires an explicitly isolated environment Geometry")
    if environment_geometry.atom_ids != charges.atom_ids:
        raise ValueError("Environment atom IDs must match fixed charges in order")
    if set(qm_geometry.atom_ids).intersection(environment_geometry.atom_ids):
        raise ValueError("QM and environment atom IDs must be disjoint")
    separation = qm_geometry.positions[:, None] - environment_geometry.positions[None, :]
    if np.any(np.all(separation == 0, axis=2)):
        raise ValueError("QM nuclei and external charge sites must not coincide")


@dataclass(frozen=True, eq=False)
class TDAReference:
    """Immutable fixed reference with safe JSON records and a canonical SHA256.

    runtime_json contains only version and relative source-hash metadata.
    Coordinates are included for cross-AO overlaps, not as a mutable scanner.
    """

    _settings_type = TDASettings
    _method_id = "cpu-rhf-tda-singlet-fixed-reference"
    _source_hashes = _SOURCE_SHA256

    qm_geometry: Geometry
    settings: TDASettings
    transition_density: np.ndarray
    seed_root: int
    excitation_energy_ev: float
    reference_gap_ev: float
    label: str
    runtime_json: str
    environment_geometry: Geometry | None = None
    charges: FixedEmbeddingCharges | None = None
    sha256: str = field(init=False)

    def __post_init__(self):
        if type(self.settings) is not self._settings_type:
            raise TypeError("Reference settings must be TDASettings")
        _validate_geometry(self.qm_geometry, self.environment_geometry, self.charges)
        root = _integer(self.seed_root, "seed_root", 1)
        if root > self.settings.nstates:
            raise ValueError("seed_root exceeds the requested candidate roots")
        object.__setattr__(self, "seed_root", root)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Reference label must be explicit and nonempty")
        transition = _real(self.transition_density, "Reference transition density")
        if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("Reference transition density must be a square AO matrix")
        if transition.size == 0 or not np.any(transition):
            raise ValueError("Reference transition density must be nonzero")
        object.__setattr__(self, "transition_density", transition)
        for name in ("excitation_energy_ev", "reference_gap_ev"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.reference_gap_ev < self.settings.min_gap_ev:
            raise ValueError("Reference state fails the minimum candidate-root gap")
        runtime = json.loads(self.runtime_json)
        self._validate_runtime(runtime)
        object.__setattr__(self, "runtime_json", _canonical(runtime))
        object.__setattr__(self, "sha256", _digest(self._payload()))

    def _validate_runtime(self, runtime):
        if not isinstance(runtime, dict) or set(runtime) != {"pyscf_version", "source_sha256"}:
            raise ValueError("Reference runtime must declare PySCF version and source hashes")
        _verify_runtime_record(runtime, self._source_hashes)

    @property
    def runtime(self):
        """Return a fresh metadata copy; callers cannot mutate the reference."""
        return json.loads(self.runtime_json)

    def _payload(self):
        return {
            "schema_version": 1,
            "method": self._method_id,
            "qm_geometry": _geometry_record(self.qm_geometry),
            "settings": asdict(self.settings),
            "transition_density": self.transition_density.tolist(),
            "seed_root": self.seed_root,
            "excitation_energy_ev": self.excitation_energy_ev,
            "reference_gap_ev": self.reference_gap_ev,
            "label": self.label,
            "runtime": self.runtime,
            "environment_geometry": (
                None
                if self.environment_geometry is None
                else _geometry_record(self.environment_geometry)
            ),
            "charges": _charges_record(self.charges),
        }

    def to_record(self):
        """Return a JSON-safe copy; writing it to disk remains the caller's job."""
        return {**self._payload(), "sha256": self.sha256}

    @classmethod
    def from_record(cls, record):
        """Load a strict, checksum-verified reference without executable objects."""
        keys = {
            "schema_version",
            "method",
            "qm_geometry",
            "settings",
            "transition_density",
            "seed_root",
            "excitation_energy_ev",
            "reference_gap_ev",
            "label",
            "runtime",
            "environment_geometry",
            "charges",
            "sha256",
        }
        if not isinstance(record, dict) or set(record) != keys:
            raise ValueError("Reference record has missing or unknown fields")
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("Unsupported reference schema")
        if record["method"] != cls._method_id:
            raise ValueError("Unsupported reference electronic-state method")
        payload = {key: value for key, value in record.items() if key != "sha256"}
        if record["sha256"] != _digest(payload):
            raise ValueError("Reference SHA256 mismatch")
        setting_keys = {item.name for item in fields(cls._settings_type)}
        if not isinstance(record["settings"], dict) or set(record["settings"]) != setting_keys:
            raise ValueError("Reference must preserve all explicit TDA settings")
        reference = cls(
            _geometry_from_record(record["qm_geometry"]),
            cls._settings_type(**record["settings"]),
            record["transition_density"],
            record["seed_root"],
            record["excitation_energy_ev"],
            record["reference_gap_ev"],
            record["label"],
            _canonical(record["runtime"]),
            None
            if record["environment_geometry"] is None
            else _geometry_from_record(record["environment_geometry"]),
            _charges_from_record(record["charges"]),
        )
        if reference.sha256 != record["sha256"]:
            raise ValueError("Reference values are not in their canonical form")
        return reference


def _verify_runtime_record(record, expected_sources=None):
    if record.get("pyscf_version") != _PYSCF_VERSION:
        raise EvaluationError(f"Experimental TDA requires inspected PySCF {_PYSCF_VERSION}")
    if record.get("source_sha256") != (_SOURCE_SHA256 if expected_sources is None else expected_sources):
        raise EvaluationError("PySCF source hashes differ from the inspected TDA gradient contract")


def _load_runtime(settings):
    try:
        import pyscf
        from pyscf import gto, lib, qmmm, scf
        from pyscf.data import nist
        from pyscf.grad import tdrhf
    except ImportError as exc:
        raise ImportError("PySCF is required for the experimental CPU RHF/TDA provider") from exc
    root = Path(pyscf.__file__).resolve().parent
    record = {
        "pyscf_version": pyscf.__version__,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _SOURCE_SHA256
        },
    }
    _verify_runtime_record(record)
    if settings.threads is not None:
        lib.num_threads(settings.threads)
    return SimpleNamespace(
        gto=gto,
        scf=scf,
        qmmm=qmmm,
        tdrhf=tdrhf,
        record=record,
        hartree_ev=float(nist.HARTREE2EV),
        bohr_angstrom=float(nist.BOHR),
    )


def _make_molecule(geometry, settings, runtime):
    molecule = runtime.gto.M(
        atom=list(zip(geometry.symbols, geometry.positions.tolist())),
        unit="Angstrom",
        basis=settings.basis,
        charge=settings.charge,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    if molecule.nelectron <= 0 or molecule.nelectron % 2:
        raise EvaluationError("RHF/TDA requires a positive even number of electrons")
    if np.any(np.asarray(molecule.atom_charges()) <= 0) or molecule._ecp:
        raise EvaluationError("Ghost atoms and effective-core potentials are outside this route")
    # Named basis only: no ECP dictionary or additional Hamiltonian terms.
    return molecule


def _eigensystem_residuals(vind, energies, vectors, tolerance):
    """Check ||A*x - omega*x||_2 / ||x||_2 against the original Hartree tolerance."""
    energies = _real(energies, "TDA eigenvalues")
    vectors = _real(vectors, "TDA eigenvectors")
    if vectors.ndim != 2 or energies.shape != (len(vectors),) or vectors.shape[1] == 0:
        raise EvaluationError("TDA eigensystem shapes are inconsistent")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise EvaluationError("TDA eigenvectors require finite nonzero norms")
    normalized = vectors / norms[:, None]
    applied = _real(vind(normalized), "TDA operator action")
    if applied.shape != vectors.shape:
        raise EvaluationError("TDA operator action has the wrong shape")
    residual = _real(
        np.linalg.norm(applied - energies[:, None] * normalized, axis=1),
        "Normalized TDA eigensystem residuals",
    )
    if np.any(residual >= tolerance):
        raise EvaluationError(
            "TDA normalized eigensystem residual exceeds the original tolerance: "
            f"{np.max(residual):.9e} >= {tolerance:.9e}"
        )
    return tuple(float(value) for value in residual)


def _dense_tda(td, settings, *, matrix_builder=None, diagonalize=None):
    """Explicit bounded dense TDA, using only the official operator action.

    This is an opt-in eigensolver, not a fallback after iterative failure.
    Default columns use batches of at most 16 unit vectors. An explicit method
    builder may supply the same matrix; original-operator residuals still apply.
    """
    occupations = _real(td._scf.mo_occ, "RHF occupations")
    if occupations.ndim != 1 or not np.isin(occupations, (0, 2)).all():
        raise EvaluationError("Dense TDA requires restricted zero/two occupations")
    nocc = int(np.count_nonzero(occupations))
    nvir = len(occupations) - nocc
    dimension = nocc * nvir
    if nocc == 0 or nvir == 0 or dimension < settings.nstates:
        raise EvaluationError("Dense TDA has too few occupied-virtual candidate dimensions")
    if dimension > settings.tda_dense_max_dimension:
        raise EvaluationError("Dense TDA dimension exceeds its declared cap")
    vind, _ = td.gen_vind()
    assembly = {"tda_dense_batch_size": 16}
    if matrix_builder is None:
        matrix = np.empty((dimension, dimension))
        for start in range(0, dimension, 16):
            stop = min(start + 16, dimension)
            seeds = np.zeros((stop - start, dimension))
            seeds[np.arange(stop - start), np.arange(start, stop)] = 1
            action = _real(vind(seeds), "Dense TDA matrix columns")
            if action.shape != seeds.shape:
                raise EvaluationError("Dense TDA operator batch has the wrong shape")
            matrix[:, start:stop] = action.T
    else:
        matrix, assembly = matrix_builder(td, nocc, nvir)
        matrix = _real(matrix, "Dense TDA matrix")
        if matrix.shape != (dimension, dimension):
            raise EvaluationError("Dense TDA matrix builder returned the wrong shape")
    hermiticity_error = float(np.max(np.abs(matrix - matrix.T)))
    if hermiticity_error > 1e-12:
        raise EvaluationError("Dense TDA matrix fails the 1e-12 Hartree Hermiticity guard")
    # Only roundoff-level asymmetry is averaged. Residuals below are checked
    # against the original operator, independently of this dense matrix.
    solver = np.linalg.eigh if diagonalize is None else diagonalize
    eigenvalues, eigenvectors = solver(0.5 * (matrix + matrix.T))
    eigenvalues = _real(eigenvalues, "Dense TDA eigenvalues")
    eigenvectors = _real(eigenvectors, "Dense TDA eigenvectors")
    threshold = _positive(td.positive_eig_threshold, "TDA positive eigenvalue threshold")
    indices = np.flatnonzero(eigenvalues > threshold)[: settings.nstates]
    if len(indices) != settings.nstates:
        raise EvaluationError("Dense TDA has too few roots above the official positive threshold")
    selected = eigenvalues[indices]
    vectors = eigenvectors[:, indices].T
    residuals = _eigensystem_residuals(vind, selected, vectors, settings.tda_conv_tol)
    td.e = selected.copy()
    td.xy = [(vector.reshape(nocc, nvir) * np.sqrt(0.5), 0) for vector in vectors]
    td.converged = np.ones(settings.nstates, dtype=bool)
    return {
        "tda_solver": "dense",
        "tda_dimension": dimension,
        "tda_dense_max_dimension": settings.tda_dense_max_dimension,
        **assembly,
        "tda_hermiticity_max_error_hartree": hermiticity_error,
        "tda_hermiticity_tolerance_hartree": 1e-12,
        "tda_positive_eigenvalue_threshold_hartree": threshold,
        "tda_matrix_residuals_hartree": residuals,
    }


def _candidate_data(method, td, settings, runtime):
    energies = _real(td.e, "Excitation energies")
    convergence = np.asarray(td.converged)
    if energies.shape != (settings.nstates,) or len(td.xy) != settings.nstates:
        raise EvaluationError("TDA did not return the requested number of candidate roots")
    if convergence.shape != energies.shape or convergence.dtype.kind != "b":
        raise EvaluationError("TDA must report a convergence flag for every candidate root")
    if not convergence.all() or np.any(energies <= 0):
        raise EvaluationError("All candidate TDA roots must converge with positive energies")
    coefficients = _real(method.mo_coeff, "RHF molecular orbitals")
    occupations = _real(method.mo_occ, "RHF occupations")
    if coefficients.ndim != 2 or occupations.shape != (coefficients.shape[1],):
        raise EvaluationError("Invalid RHF molecular orbital shapes")
    if not np.isin(occupations, (0, 2)).all():
        raise EvaluationError("RHF occupations must be exactly zero or two")
    nocc = int(np.count_nonzero(occupations))
    nvir = coefficients.shape[1] - nocc
    if not np.all(occupations[:nocc] == 2) or not np.all(occupations[nocc:] == 0):
        raise EvaluationError("RHF occupied and virtual orbitals must be ordered")
    if nocc == 0 or nvir == 0:
        raise EvaluationError("TDA requires occupied and virtual orbitals")
    transitions = []
    amplitudes = []
    for x, y in td.xy:
        x = _real(x, "TDA X amplitudes")
        y = _real(y, "TDA Y amplitudes")
        if x.shape != (nocc, nvir) or np.any(y != 0):
            raise EvaluationError("Expected restricted TDA X amplitudes and zero Y")
        if not np.isclose(2 * np.sum(x * x), 1, atol=1e-7, rtol=0):
            raise EvaluationError("Restricted TDA amplitudes violate 2*norm(X)^2 = 1")
        transitions.append(coefficients[:, :nocc] @ x @ coefficients[:, nocc:].T)
        amplitudes.append(x.ravel())
    vind, _ = td.gen_vind()
    residuals = _eigensystem_residuals(
        vind, energies, np.asarray(amplitudes), settings.tda_conv_tol
    )
    return (
        energies * runtime.hartree_ev,
        transitions,
        {
            "tda_eigensystem_residuals_hartree": residuals,
            "tda_eigensystem_residual_max_hartree": max(residuals),
            "tda_eigensystem_residual_tolerance_hartree": settings.tda_conv_tol,
        },
    )


def _solve(qm_geometry, settings, runtime, environment_geometry=None, charges=None, *,
           method_factory=None, dense_solver=None):
    molecule = _make_molecule(qm_geometry, settings, runtime)
    method = runtime.scf.RHF(molecule) if method_factory is None else method_factory(molecule)
    method.conv_tol = settings.conv_tol
    method.conv_tol_grad = settings.conv_tol_grad
    method.max_cycle = settings.max_cycle
    if charges is not None:
        method = runtime.qmmm.mm_charge(
            method,
            environment_geometry.positions,
            charges.charges_e,
            radii=charges.radii_angstrom,
            unit="Angstrom",
        )
    ground_energy = method.kernel()
    if not method.converged:
        raise EvaluationError("SCF did not converge")
    ground_energy = _real(ground_energy, "SCF total energy")
    if ground_energy.ndim != 0:
        raise EvaluationError("SCF total energy must be scalar")
    ground_energy = float(ground_energy)
    td = method.TDA()
    td.singlet = True
    td.nstates = settings.nstates
    td.conv_tol = settings.tda_conv_tol
    td.max_cycle = settings.tda_max_cycle
    if settings.tda_solver == "dense":
        solver_diagnostics = (_dense_tda if dense_solver is None else dense_solver)(td, settings)
    else:
        td.kernel()
        solver_diagnostics = {"tda_solver": "iterative", "tda_lindep": float(td.lindep)}
    energies, transitions, residual_diagnostics = _candidate_data(method, td, settings, runtime)
    solver_diagnostics.update(residual_diagnostics)
    return SimpleNamespace(
        molecule=molecule,
        method=method,
        td=td,
        energies=energies,
        transitions=transitions,
        tda_diagnostics=solver_diagnostics,
        ground_energy_ev=ground_energy * runtime.hartree_ev,
        overlap=_real(molecule.intor_symmetric("int1e_ovlp"), "AO overlap"),
    )


def transition_overlap(reference, candidate, reference_overlap, candidate_overlap, cross_overlap):
    """Absolute, normalized AO transition-density overlap (not a CI overlap).

    Cross overlaps include moving AO centers. This criterion is invariant to
    an overall transition-amplitude sign; it is not a degeneracy resolution.
    """
    reference = _real(reference, "Reference transition density")
    candidate = _real(candidate, "Candidate transition density")
    sr = _real(reference_overlap, "Reference AO overlap")
    sc = _real(candidate_overlap, "Candidate AO overlap")
    cross = _real(cross_overlap, "Cross AO overlap")
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[0] != reference.shape[1]
        or candidate.shape[0] != candidate.shape[1]
        or sr.shape != reference.shape
        or sc.shape != candidate.shape
        or cross.shape != (reference.shape[0], candidate.shape[0])
    ):
        raise EvaluationError("Transition-density and AO overlap shapes are inconsistent")
    norm_r = float(np.sum(reference * (sr @ reference @ sr)))
    norm_c = float(np.sum(candidate * (sc @ candidate @ sc)))
    if norm_r <= 0 or norm_c <= 0 or not np.isfinite([norm_r, norm_c]).all():
        raise EvaluationError("Transition densities require finite positive AO norms")
    numerator = abs(float(np.sum(reference * (cross @ candidate @ cross.T))))
    score = numerator / np.sqrt(norm_r * norm_c)
    if not np.isfinite(score) or score > 1 + 1e-7:
        raise EvaluationError("Invalid normalized transition-density overlap")
    return float(min(score, 1.0))


def select_tda_state(scores, energies_ev, settings):
    """Select one root or fail explicitly on insufficient/ambiguous information."""
    scores = _real(scores, "State overlaps")
    energies = _real(energies_ev, "Candidate excitation energies")
    if scores.shape != (settings.nstates,) or energies.shape != scores.shape:
        raise EvaluationError("State selection requires every requested candidate root")
    if np.any(scores < 0) or np.any(scores > 1) or np.any(energies <= 0):
        raise EvaluationError("State scores or excitation energies are invalid")
    order = np.argsort(-scores, kind="stable")
    index = int(order[0])
    margin = float(scores[index] - scores[order[1]])
    gap = float(np.min(np.abs(np.delete(energies, index) - energies[index])))
    if scores[index] < settings.min_overlap:
        raise EvaluationError("Selected TDA state fails the minimum reference overlap")
    if margin < settings.min_margin:
        raise EvaluationError("TDA state is ambiguous by its overlap margin")
    if gap < settings.min_gap_ev:
        raise EvaluationError("Selected TDA state is too close to another candidate energy")
    return index, {
        "selected_root": index + 1,
        "overlap": float(scores[index]),
        "overlap_margin": margin,
        "candidate_gap_ev": gap,
        "candidate_overlaps": tuple(float(value) for value in scores),
        "candidate_excitation_energies_ev": tuple(float(value) for value in energies),
    }


def build_tda_reference(
    qm_geometry, settings, *, initial_root, label, environment_geometry=None, charges=None,
):
    """Build the existing source-pinned RHF/TDA reference."""
    return _build_reference(
        qm_geometry, settings, reference_type=TDAReference, runtime_loader=_load_runtime,
        solver=_solve, initial_root=initial_root, label=label,
        environment_geometry=environment_geometry, charges=charges,
    )


def _build_reference(
    qm_geometry,
    settings,
    *,
    reference_type, runtime_loader, solver,
    initial_root,
    label,
    environment_geometry=None,
    charges=None,
):
    """Converge one declared geometry and pin one seed state for every later call."""
    if type(settings) is not reference_type._settings_type:
        raise TypeError("settings must be TDASettings")
    _validate_geometry(qm_geometry, environment_geometry, charges)
    initial_root = _integer(initial_root, "initial_root", 1)
    if initial_root > settings.nstates:
        raise ValueError("initial_root exceeds the requested candidate roots")
    runtime = runtime_loader(settings)
    solved = solver(qm_geometry, settings, runtime, environment_geometry, charges)
    return _reference_from_solved(
        qm_geometry, settings, solved, runtime, reference_type=reference_type,
        initial_root=initial_root, label=label,
        environment_geometry=environment_geometry, charges=charges,
    )


def _reference_from_solved(
    qm_geometry, settings, solved, runtime, *, reference_type,
    initial_root, label, environment_geometry=None, charges=None,
):
    """Pin a state from a checked common solution without another quantum call."""
    transition = solved.transitions[initial_root - 1]
    scores = [
        transition_overlap(transition, other, solved.overlap, solved.overlap, solved.overlap)
        for other in solved.transitions
    ]
    index, diagnostic = select_tda_state(scores, solved.energies, settings)
    if index != initial_root - 1:
        raise EvaluationError("Seed state was not uniquely recovered from its own reference")
    return reference_type(
        qm_geometry,
        settings,
        transition,
        initial_root,
        float(solved.energies[index]),
        diagnostic["candidate_gap_ev"],
        label,
        _canonical(runtime.record),
        environment_geometry,
        charges,
    )


def _cphf_residual(z, w, mo_energy, nocc, fvind):
    """Residual of the fixed-basis equation A(z) = -w in atomic units."""
    z = _real(z, "CPHF response")
    w = _real(w, "CPHF right-hand side")
    energies = _real(mo_energy, "RHF orbital energies")
    if (
        z.ndim != 2
        or nocc < 1
        or z.shape[1] != nocc
        or z.shape[0] < 1
        or w.shape != z.shape
        or energies.shape != (nocc + z.shape[0],)
    ):
        raise EvaluationError("CPHF response data shapes are inconsistent")
    response = _real(fvind(z.ravel()), "CPHF response action")
    if response.size != z.size:
        raise EvaluationError("CPHF response action has the wrong size")
    residual = (energies[nocc:, None] - energies[None, :nocc]) * z + response.reshape(z.shape) + w
    return _real(residual, "CPHF equation residual")


def _refine_cphf(solve, fvind, mo_energy, mo_occ, h1, settings):
    """Apply at most two normalized defect corrections before gradient assembly.

    With r = A(z) + h1 and s = max(abs(r)), the same official solver
    returns u for A(u) = -r/s. Thus z + s*u corrects the original equation.
    Normalizing the small right-hand side avoids the absolute Krylov
    linear-dependence floor without changing the acceptance tolerance.
    """
    occupations = _real(mo_occ, "RHF occupations")
    if occupations.ndim != 1 or not np.isin(occupations, (0, 2)).all():
        raise EvaluationError("CPHF refinement requires restricted zero/two occupations")
    nocc = int(np.count_nonzero(occupations))
    nvir = len(occupations) - nocc
    if (
        nocc == 0
        or nvir == 0
        or not np.all(occupations[:nocc] == 2)
        or not np.all(occupations[nocc:] == 0)
    ):
        raise EvaluationError("CPHF refinement requires ordered occupied and virtual orbitals")
    h1 = _real(h1, "CPHF right-hand side")
    if h1.shape != (nvir, nocc):
        raise EvaluationError("CPHF right-hand side has the wrong shape")

    def solve_response(right_hand_side):
        pair = solve(fvind, mo_energy, mo_occ, right_hand_side)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or pair[1] is not None:
            raise EvaluationError("Expected the field-independent CPHF response pair")
        response = _real(pair[0], "Solved CPHF response")
        if response.shape != h1.shape:
            raise EvaluationError("Solved CPHF response has the wrong shape")
        return response

    response = solve_response(h1)
    residual = _cphf_residual(response, h1, mo_energy, nocc, fvind)
    history = [float(np.max(np.abs(residual)))]
    for _ in range(2):
        if history[-1] <= settings.response_residual_tol:
            break
        scale = history[-1]
        correction = solve_response(residual / scale)
        candidate = _real(response + scale * correction, "Refined CPHF response")
        candidate_residual = _cphf_residual(candidate, h1, mo_energy, nocc, fvind)
        candidate_norm = float(np.max(np.abs(candidate_residual)))
        if candidate_norm >= history[-1]:
            raise EvaluationError(
                "Normalized CPHF defect correction did not strictly decrease the residual: "
                f"{history[-1]:.9e} -> {candidate_norm:.9e}"
            )
        response, residual = candidate, candidate_residual
        history.append(candidate_norm)
    if history[-1] > settings.response_residual_tol:
        raise EvaluationError(
            "CPHF residual exceeds the original tolerance after two bounded refinements: "
            f"{history[-1]:.9e} > {settings.response_residual_tol:.9e}"
        )
    return (response, None), {
        "cphf_refinement_method": "normalized-defect-correction",
        "cphf_refinement_iterations": len(history) - 1,
        "cphf_residual_before_refinement": history[0],
        "cphf_residual_after_refinement": history[-1],
        "cphf_residual_history": tuple(history),
        "cphf_residual_tolerance": settings.response_residual_tol,
    }


def _relaxed_density(hook, overlap, electron_count, settings):
    """Extract the coefficient of an external one-electron perturbation.

    The official RHF/TDA gradient contracts h1 with 2*oo0 + dmz1doo.
    h1 for a displaced external charge is symmetric, so sym(dmz1doo)
    has the same contraction and is the relaxed AO density used below.
    """
    required = {"oo0", "dmz1doo", "z1", "wvo", "fvind", "mo_energy", "nocc", "nvir"}
    if not required.issubset(hook):
        raise EvaluationError("Inspected TDA relaxed-density hook data are missing")
    occupied = _real(hook["oo0"], "Occupied AO density")
    response_density = _real(hook["dmz1doo"], "TDA response density")
    overlap = _real(overlap, "AO overlap")
    if occupied.shape != overlap.shape or response_density.shape != overlap.shape:
        raise EvaluationError("Relaxed AO density shape does not match the basis")
    density = 2 * occupied + 0.5 * (response_density + response_density.T)
    if not np.allclose(density, density.T, atol=1e-10, rtol=0):
        raise EvaluationError("Relaxed AO density is not symmetric")
    trace = float(np.einsum("ij,ji->", density, overlap))
    if not np.isfinite(trace) or abs(trace - electron_count) > settings.density_trace_tol:
        raise EvaluationError("Relaxed AO density does not preserve the electron count")
    z = _real(hook["z1"], "CPHF response")
    if z.shape != (hook["nvir"], hook["nocc"]):
        raise EvaluationError("CPHF response data shapes are inconsistent")
    residual = _cphf_residual(z, hook["wvo"], hook["mo_energy"], hook["nocc"], hook["fvind"])
    residual_max = float(np.max(np.abs(residual)))
    if not np.isfinite(residual_max) or residual_max > settings.response_residual_tol:
        raise EvaluationError("TDA CPHF equation residual exceeds its declared tolerance")
    return density, {"density_trace": trace, "cphf_residual_max": residual_max}


def _total_gradient(solved, index, settings, runtime, *, gradient_base=None, refine=True):
    captured = {}
    response_solve = {}
    seen_atoms = []
    response_fields = ("oo0", "dmz1doo", "z1", "wvo", "mo_energy")

    base = runtime.tdrhf.Gradients if gradient_base is None else gradient_base

    class CapturingGradient(base):
        def solve_cphf(self, fvind, mo_energy, mo_occ, h1):
            if not refine:
                raise EvaluationError("Unexpected orbital-response solver hook for this method")
            if response_solve:
                raise EvaluationError("Expected exactly one TDA orbital-response solve")
            pair, diagnostics = _refine_cphf(
                super().solve_cphf, fvind, mo_energy, mo_occ, h1, settings
            )
            response_solve.update(response=pair[0], diagnostics=diagnostics)
            return pair

        def extra_force(self, atom_id, envs):
            if refine and (not response_solve or not np.array_equal(
                envs.get("z1"), response_solve["response"]
            )):
                raise EvaluationError("TDA gradient did not use the checked orbital response")
            if not captured:
                density, diagnostics = _relaxed_density(
                    envs, solved.overlap, solved.molecule.nelectron, settings
                )
                captured.update(
                    {
                        "density": density,
                        "diagnostics": diagnostics,
                        "inputs": {name: _real(envs[name], name) for name in response_fields},
                        "dimensions": (envs["nocc"], envs["nvir"]),
                    }
                )
            elif captured["dimensions"] != (envs.get("nocc"), envs.get("nvir")) or any(
                name not in envs or not np.array_equal(envs[name], captured["inputs"][name])
                for name in response_fields
            ):
                raise EvaluationError("TDA response data changed between atom hooks")
            # fvind includes an expensive response-potential contraction; check
            # its residual once, while checking identical inputs on every atom.
            seen_atoms.append(atom_id)
            return super().extra_force(atom_id, envs)

    try:
        gradient_method = CapturingGradient(solved.td)
        gradient_method.cphf_conv_tol = settings.cphf_conv_tol
        gradient_method.cphf_max_cycle = settings.cphf_max_cycle
        gradient = _real(gradient_method.kernel(state=index + 1), "Total excited QM gradient")
        if gradient.shape != (solved.molecule.natm, 3):
            raise EvaluationError("Total excited QM gradient has the wrong shape")
        if seen_atoms != list(range(solved.molecule.natm)):
            raise EvaluationError("TDA gradient did not supply one response hook per QM atom")
        if refine:
            captured["diagnostics"].update(response_solve["diagnostics"])
        else:
            captured["diagnostics"].update(
                cphf_solver="official-unrefined-residual-checked", cphf_refinement_count=0
            )
        return gradient, captured["density"], captured["diagnostics"]
    finally:
        # A temporary subclass has cyclic references to its methods. Do not let
        # those cycles retain SCF, grid and response workspaces between calls.
        solved = None
        captured.clear()
        response_solve.clear()


def _surface_energy_diagnostics(energy, diagnostics):
    """Validate the same-call total-state ledger without selecting a state."""
    if not isinstance(diagnostics, Mapping):
        raise EvaluationError("TDA surface diagnostics must be a mapping")
    required = {
        "method", "energy_definition", "ground_energy_ev", "excitation_energy_ev",
        "selected_root", "candidate_excitation_energies_ev", "scf_converged",
        "candidate_roots_converged", "forces_evaluated",
    }
    if not required.issubset(diagnostics):
        raise EvaluationError("TDA surface diagnostics omit required energy or force fields")
    result = dict(diagnostics)
    if (
        result["method"] not in {"cpu-rhf-tda-singlet", "cpu-pbe0-tda-singlet",
                                 "gpu-rhf-tda-singlet"}
        or result["energy_definition"] != {
            "cpu-rhf-tda-singlet": "E_RHF+omega",
            "gpu-rhf-tda-singlet": "E_RHF+omega",
            "cpu-pbe0-tda-singlet": "E_PBE0+omega",
        }.get(result["method"])
        or result["scf_converged"] is not True
        or result["candidate_roots_converged"] is not True
    ):
        raise EvaluationError("TDA surface requires converged total energy for its declared method")
    for name in ("ground_energy_ev", "excitation_energy_ev"):
        value = _real(result[name], name)
        if value.ndim != 0:
            raise EvaluationError("TDA surface energy components must be scalars")
        result[name] = float(value)
    candidates = _real(result["candidate_excitation_energies_ev"], "Candidate excitations")
    try:
        root = _integer(result["selected_root"], "selected_root", 1)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(str(exc)) from exc
    if (
        candidates.ndim != 1
        or len(candidates) < 2
        or root > len(candidates)
        or np.any(candidates <= 0)
        or result["excitation_energy_ev"] != candidates[root - 1]
    ):
        raise EvaluationError("TDA surface excitation must match its selected candidate root")
    # This is the existing native-record bookkeeping tolerance, not an E/F
    # accuracy criterion. The provider itself constructs this same sum exactly.
    if abs(energy - (result["ground_energy_ev"] + result["excitation_energy_ev"])) > 1e-7:
        raise EvaluationError("TDA surface total energy must equal its ground energy plus omega")
    result["selected_root"] = root
    result["candidate_excitation_energies_ev"] = tuple(float(value) for value in candidates)
    return result


@dataclass(frozen=True, eq=False)
class TDASurfaceEvaluation:
    """Total excited energy, optional real forces, and explicit state diagnostics.

    Energy-only evaluation sets both force fields to None. It never represents
    an unavailable derivative as a zero force.
    """

    energy_ev: float
    qm_atom_ids: tuple[int, ...]
    environment_atom_ids: tuple[int, ...]
    qm_forces_ev_per_angstrom: np.ndarray | None
    environment_forces_ev_per_angstrom: np.ndarray | None
    diagnostics: object

    def __post_init__(self):
        energy = _real(self.energy_ev, "Total excited energy")
        if energy.ndim != 0:
            raise EvaluationError("Total excited energy must be scalar")
        object.__setattr__(self, "energy_ev", float(energy))
        try:
            qm_ids = _atom_ids(self.qm_atom_ids)
            environment_ids = tuple(self.environment_atom_ids)
            if environment_ids:
                environment_ids = _atom_ids(environment_ids)
        except (TypeError, ValueError) as exc:
            raise EvaluationError(str(exc)) from exc
        if set(qm_ids).intersection(environment_ids):
            raise EvaluationError("QM and environment atom IDs must be disjoint")
        object.__setattr__(self, "qm_atom_ids", qm_ids)
        object.__setattr__(self, "environment_atom_ids", environment_ids)
        diagnostics = _surface_energy_diagnostics(float(energy), self.diagnostics)
        qm_available = self.qm_forces_ev_per_angstrom is not None
        environment_available = self.environment_forces_ev_per_angstrom is not None
        if environment_ids and qm_available != environment_available:
            raise EvaluationError("Embedded TDA surface requires both QM and environment forces")
        if not environment_ids and environment_available:
            raise EvaluationError("Isolated TDA surface must not contain environment forces")
        if diagnostics["forces_evaluated"] is not qm_available:
            raise EvaluationError("TDA surface force availability differs from its diagnostics")
        for name, ids in (
            ("qm_forces_ev_per_angstrom", self.qm_atom_ids),
            ("environment_forces_ev_per_angstrom", self.environment_atom_ids),
        ):
            force = getattr(self, name)
            if force is not None:
                force = _real(force, name)
                if force.shape != (len(ids), 3):
                    raise EvaluationError("Excited force shape does not match the declared atoms")
                object.__setattr__(self, name, force)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    @property
    def ground_energy_ev(self):
        """Ground energy at this geometry under the same method and embedding field."""
        return self.diagnostics["ground_energy_ev"]

    @property
    def excitation_energy_ev(self):
        """Selected excitation gap; this is not the returned total potential energy."""
        return self.diagnostics["excitation_energy_ev"]


class PySCFTDABackend:
    """Fixed-reference RHF/TDA with optional last-successful exact-input reuse.

    Reuse changes execution only. It stores no wavefunction initial guess or
    state-selection history and is empty in every newly constructed provider.
    """

    _settings_type = TDASettings
    _reference_type = TDAReference
    _embedded = False
    _method_id = "cpu-rhf-tda-singlet"
    _energy_definition = "E_RHF+omega"

    def _load_method_runtime(self, settings):
        return _load_runtime(settings)

    def _solve_surface(self, qm, settings, runtime, environment, charges):
        return _solve(qm, settings, runtime, environment, charges)

    def _surface_gradient(self, solved, index, settings, runtime):
        return _total_gradient(solved, index, settings, runtime)

    def __init__(self, settings, reference, *, reuse_identical_geometry=False):
        if type(reuse_identical_geometry) is not bool:
            raise TypeError("reuse_identical_geometry must be a boolean")
        if type(settings) is not self._settings_type or type(reference) is not self._reference_type:
            raise TypeError("TDA provider requires TDASettings and a pinned TDAReference")
        if settings != reference.settings:
            raise ValueError("TDA provider settings differ from the pinned reference")
        if not self._embedded and reference.charges is not None:
            raise ValueError("Isolated TDA provider requires an isolated reference")
        self.settings = settings
        self.reference = reference
        self._runtime = self._load_method_runtime(settings)
        if self._runtime.record != reference.runtime:
            raise EvaluationError("TDA runtime differs from the pinned reference")
        self._reference_molecule = _make_molecule(reference.qm_geometry, settings, self._runtime)
        self._reference_overlap = _real(
            self._reference_molecule.intor_symmetric("int1e_ovlp"), "Reference AO overlap"
        )
        if self._reference_overlap.shape != reference.transition_density.shape:
            raise EvaluationError("Reference transition density does not match its AO basis")

        self._reuse_identical_geometry = reuse_identical_geometry
        self._reuse_settings = settings
        self._reuse_reference = reference
        self._reuse_runtime = self._runtime
        self._reuse_runtime_record = _canonical(self._runtime.record)
        self._last_surface = None
        self._reuse_counts = {"requests": 0, "hits": 0, "calculations": 0, "failures": 0}

    def clear_reuse(self):
        """Discard the saved result without changing the fixed state reference."""
        self._last_surface = None

    def reuse_statistics(self):
        """Return detached call counts; these do not measure saved wall time."""
        return {"enabled": self._reuse_identical_geometry, **self._reuse_counts}

    @staticmethod
    def _detached_surface(result):
        return TDASurfaceEvaluation(
            result.energy_ev, result.qm_atom_ids, result.environment_atom_ids,
            result.qm_forces_ev_per_angstrom, result.environment_forces_ev_per_angstrom,
            deepcopy(dict(result.diagnostics)),
        )

    def _reuse_key(self, qm_geometry, environment_geometry, charges, forces):
        # These public settings/reference objects are immutable. Replacing one
        # requires a new provider, because its reference AO objects were built
        # at construction. Runtime records and charge provenance must also agree.
        if (
            self.settings != self._reuse_settings
            or self.reference is not self._reuse_reference
            or self._runtime is not self._reuse_runtime
            or _canonical(self._runtime.record) != self._reuse_runtime_record
            or _charges_record(charges) != _charges_record(self.reference.charges)
        ):
            raise EvaluationError("TDA provider configuration changed; construct a new provider")

        def geometry_key(geometry):
            if geometry is None:
                return None
            return (geometry.atom_ids, geometry.symbols, geometry.positions.tobytes(),
                    None if geometry.cell is None else geometry.cell.tobytes())

        return (geometry_key(qm_geometry), geometry_key(environment_geometry), forces)

    def _surface(self, qm_geometry, environment_geometry, charges, forces):
        self._reuse_counts["requests"] += 1
        try:
            if type(forces) is not bool:
                raise ValueError("forces must be true or false")
            self._validate_identity(qm_geometry, environment_geometry, charges)
            key = None
            if self._reuse_identical_geometry:
                key = self._reuse_key(qm_geometry, environment_geometry, charges, forces)
                if self._last_surface is not None and key == self._last_surface[0]:
                    result = self._detached_surface(self._last_surface[1])
                    self._reuse_counts["hits"] += 1
                    return result
            # A failed or different request cannot fall back to an older success.
            self.clear_reuse()
            self._reuse_counts["calculations"] += 1
            result = self._calculate_surface(qm_geometry, environment_geometry, charges, forces)
            if self._reuse_identical_geometry:
                self._last_surface = (key, self._detached_surface(result))
            return result
        except BaseException:
            self.clear_reuse()
            self._reuse_counts["failures"] += 1
            raise

    def _validate_identity(self, qm_geometry, environment_geometry, charges):
        _validate_geometry(qm_geometry, environment_geometry, charges)
        expected = self.reference.qm_geometry
        if (qm_geometry.atom_ids, qm_geometry.symbols) != (expected.atom_ids, expected.symbols):
            raise ValueError("QM membership, symbols and order must match the pinned reference")
        expected_environment = self.reference.environment_geometry
        if (environment_geometry is None) != (expected_environment is None):
            raise ValueError("Embedding mode differs from the pinned reference")
        if environment_geometry is not None and (
            environment_geometry.atom_ids,
            environment_geometry.symbols,
        ) != (expected_environment.atom_ids, expected_environment.symbols):
            raise ValueError("Environment identity differs from the pinned reference")

    def _calculate_surface(self, qm_geometry, environment_geometry, charges, forces):
        if type(forces) is not bool:
            raise ValueError("forces must be true or false")
        self._validate_identity(qm_geometry, environment_geometry, charges)
        try:
            solved = self._solve_surface(
                qm_geometry, self.settings, self._runtime, environment_geometry, charges
            )
            index, diagnostics = self._select_surface(solved, forces)
            return self._evaluate_selected_surface(
                solved, index, diagnostics, qm_geometry, environment_geometry, charges, forces
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"Experimental TDA evaluation failed: {exc}") from exc

    def _select_surface(self, solved, forces):
        """Match this immutable reference to a previously checked solution."""
        cross = self._runtime.gto.intor_cross(
            "int1e_ovlp", self._reference_molecule, solved.molecule
        )
        scores = [
            transition_overlap(
                self.reference.transition_density,
                transition,
                self._reference_overlap,
                solved.overlap,
                cross,
            )
            for transition in solved.transitions
        ]
        index, diagnostics = select_tda_state(scores, solved.energies, self.settings)
        diagnostics.update(solved.tda_diagnostics)
        diagnostics.update(
            {
                "reference_sha256": self.reference.sha256,
                "reference_label": self.reference.label,
                "tracking": "fixed-reference-ao-transition-density",
                "qualification": "experimental-requires-matched-runtime-finite-differences",
                "method": self._method_id,
                "energy_definition": self._energy_definition,
                "ground_energy_ev": solved.ground_energy_ev,
                "excitation_energy_ev": float(solved.energies[index]),
                "scf_converged": True,
                "candidate_roots_converged": True,
                "forces_evaluated": forces,
            }
        )
        return index, diagnostics

    def _evaluate_selected_surface(
        self, solved, index, diagnostics, qm_geometry, environment_geometry, charges, forces
    ):
        """Use the same complete response for a single state or a state set."""
        diagnostics = dict(diagnostics)
        qm_force = environment_force = None
        if forces:
            gradient, relaxed_density, response_diagnostics = self._surface_gradient(
                solved, index, self.settings, self._runtime
            )
            diagnostics.update(response_diagnostics)
            factor = self._runtime.hartree_ev / self._runtime.bohr_angstrom
            qm_force = -gradient * factor
            if charges is not None:
                mm_gradient = solved.method.nuc_grad_method()
                electronic = _real(
                    mm_gradient.grad_hcore_mm(relaxed_density), "Excited MM electronic gradient"
                )
                nuclear = _real(mm_gradient.grad_nuc_mm(), "MM nuclear gradient")
                if (
                    electronic.shape != environment_geometry.positions.shape
                    or nuclear.shape != environment_geometry.positions.shape
                ):
                    raise EvaluationError(
                        "Excited MM gradient shape does not match charge sites"
                    )
                environment_force = -(electronic + nuclear) * factor
        return TDASurfaceEvaluation(
            solved.ground_energy_ev + float(solved.energies[index]),
            qm_geometry.atom_ids,
            () if environment_geometry is None else environment_geometry.atom_ids,
            qm_force,
            environment_force,
            diagnostics,
        )

    def evaluate_surface(self, qm_geometry, *, forces=True):
        """Return total energy and diagnostics; optionally calculate analytic forces."""
        return self._surface(qm_geometry, None, None, forces)

    def evaluate(self, qm_geometry):
        """Return the existing isolated-provider contract in eV and eV/Angstrom."""
        result = self.evaluate_surface(qm_geometry)
        return Evaluation(result.energy_ev, result.qm_forces_ev_per_angstrom)


class PySCFTDAElectrostaticBackend(PySCFTDABackend):
    """Experimental total excited QM-plus-fixed-charge energy and both forces.

    Charges and radii are coordinate-independent. No environment self energy,
    short-range interaction, periodic field, polarization or charge response
    is included. This provider alone is not a complete hybrid potential.
    """

    _embedded = True

    def __init__(self, settings, reference, charges, *, reuse_identical_geometry=False):
        if not isinstance(reference, TDAReference) or reference.charges is None:
            raise ValueError("Embedded TDA provider requires an embedded reference")
        if not isinstance(charges, FixedEmbeddingCharges):
            raise TypeError("Embedded TDA provider requires explicit fixed charges")
        if _charges_record(charges) != _charges_record(reference.charges):
            raise ValueError("Fixed charges or provenance differ from the pinned reference")
        self.charges = charges
        super().__init__(settings, reference, reuse_identical_geometry=reuse_identical_geometry)

    def evaluate_surface(self, qm_geometry, environment_geometry, *, forces=True):
        """Return total embedded excited energy, diagnostics and optional both forces."""
        return self._surface(qm_geometry, environment_geometry, self.charges, forces)

    def evaluate(self, qm_geometry, environment_geometry):
        """Return the existing EmbeddedEvaluation contract, including state diagnostics."""
        result = self.evaluate_surface(qm_geometry, environment_geometry)
        return EmbeddedEvaluation(
            result.energy_ev,
            result.qm_atom_ids,
            result.environment_atom_ids,
            result.qm_forces_ev_per_angstrom,
            result.environment_forces_ev_per_angstrom,
            result.diagnostics,
        )
