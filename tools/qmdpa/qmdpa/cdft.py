#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Finite CPU Hirshfeld-constrained UKS pair and fixed-fragment admission.

PySCF owns the solver and all analytic derivatives. This adapter owns units,
atom/state/runtime identity, same-frame records and fail-closed conversion.
"""

from copy import deepcopy
from pathlib import Path
import importlib
import hashlib
import re

import numpy as np

from .core import Evaluation, EvaluationError, Geometry
from .embedding import EmbeddedEvaluation, FixedEmbeddingCharges
from .named_cv import digest
from .state_records import geometry_digest

METHOD = "uks-hirshfeld-cdft"
FORM = "finite-uks-hirshfeld-pair-v1"
HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903
ROLES = ("R", "P")
RUNTIME_SOURCES = ("pyscf.dft.cdft", "pyscf.dft.uks", "pyscf.dft.numint",
                   "pyscf.dft.gen_grid", "pyscf.grad.uks", "pyscf.grad.uhf",
                   "pyscf.grad.rks", "pyscf.grad.rhf", "pyscf.scf.hf",
                   "pyscf.scf.uhf", "pyscf.scf.atom_hf", "pyscf.qmmm.itrf",
                   "pyscf.soscf.newton_ah", "pyscf.soscf.ciah", "pyscf.scf._response_functions")


def numeric(value, shape, label):
    """Reject strings, booleans, broadcasting and nonfinite state arrays."""
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("cDFT " + label + " requires a finite numeric array of the declared shape")
    return array


def atomic_reference_digest(records):
    """Same frozen proatom identity as the solver, without importing PySCF."""
    h = hashlib.sha256()
    for symbol in sorted(records):
        h.update(symbol.encode())
        h.update(repr({symbol: records[symbol]["basis"]}).encode())
        h.update(np.asarray(records[symbol]["density"], dtype="<f8").tobytes())
    return h.hexdigest()



def settings(raw):
    """One explicit finite CPU molecular DFT path; no inherited TDA admission."""
    required = {"xc", "basis", "charge", "multiplicity", "grid_level", "grid_prune",
                "grid_response", "conv_tol", "conv_tol_grad", "max_cycle",
                "constraint_tol", "max_multiplier", "max_constraint_iterations",
                "spin_square_max", "spin_population_tolerance", "inner_solver", "inner_lindep"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("cDFT qm requires all method, grid, constraint and state guards")
    if (raw["xc"] != "pbe0" or raw["basis"] != "def2-svpd"
            or type(raw["charge"]) is not int or raw["charge"] != 1
            or type(raw["multiplicity"]) is not int or raw["multiplicity"] != 2
            or raw["grid_prune"] is not None or raw["grid_response"] is not True
            or raw["inner_solver"] != "newton"):
        raise ValueError("First cDFT admission requires all-electron PBE0/def2-svpd, charge +1, doublet and full unpruned grid response")
    for key in ("grid_level", "max_cycle", "max_constraint_iterations"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError("cDFT " + key + " must be a positive integer")
    for key in required - {"xc", "basis", "charge", "multiplicity", "grid_level",
                           "grid_prune", "grid_response", "max_cycle", "max_constraint_iterations", "inner_solver"}:
        if type(raw[key]) not in (int, float) or not np.isfinite(raw[key]) or raw[key] <= 0:
            raise ValueError("cDFT " + key + " must be finite and positive")
    if raw["inner_lindep"] > 1e-24:
        raise ValueError("cDFT requires the accurate Newton AH subspace cutoff")
    if raw["constraint_tol"] > 1e-8 or raw["conv_tol"] > 1e-12 or raw["conv_tol_grad"] > 1e-7:
        raise ValueError("cDFT convergence controls exceed the admitted strict bounds")
    if not .75 < raw["spin_square_max"] <= 1.0 or raw["spin_population_tolerance"] > .2:
        raise ValueError("cDFT doublet state guards exceed the admitted bounds")
    return deepcopy(raw)


def _geometry(record):
    if not isinstance(record, dict) or set(record) != {"atom_ids", "symbols", "positions_angstrom"}:
        raise ValueError("cDFT reference requires explicit finite geometry")
    return Geometry(record["atom_ids"], record["symbols"], record["positions_angstrom"])


def _record(geometry):
    return {"atom_ids": list(geometry.atom_ids), "symbols": list(geometry.symbols),
            "positions_angstrom": geometry.positions.tolist()}


def validate_reference(raw, *, electrostatic):
    """Check portable state identity without importing a quantum engine."""
    keys = {"schema_version", "form", "settings", "qm_geometry", "environment_geometry",
            "charges", "atom_weights", "atomic_references", "states", "sampled_state",
            "runtime", "electron_counts", "sha256"}
    if (not isinstance(raw, dict) or set(raw) != keys or type(raw["schema_version"]) is not int
            or raw["schema_version"] != 1 or raw["form"] != FORM
            or raw["sha256"] != digest({k: v for k, v in raw.items() if k != "sha256"})):
        raise ValueError("cDFT reference identity is incomplete or corrupt")
    settings(raw["settings"])
    qm = _geometry(raw["qm_geometry"])
    counts = raw["electron_counts"]
    if (not isinstance(counts, list) or len(counts) != 2
            or any(type(n) is not int or n < 0 for n in counts) or counts[0]-counts[1] != 1):
        raise ValueError("cDFT reference electron counts must define the declared doublet")
    weights = raw["atom_weights"]
    if (not isinstance(weights, list) or len(weights) != len(qm.atom_ids)
            or any(type(x) is not int or x not in (-1, 1) for x in weights)
            or set(weights) != {-1, 1}):
        raise ValueError("cDFT requires explicit donor and acceptor atom weights")
    if (raw["environment_geometry"] is not None) != electrostatic:
        raise ValueError("cDFT reference field differs from the requested provider")
    if electrostatic:
        env = _geometry(raw["environment_geometry"])
        charges = FixedEmbeddingCharges(**raw["charges"])
        if (set(qm.atom_ids) & set(env.atom_ids) or charges.atom_ids != env.atom_ids
                or charges.radii_angstrom is not None):
            raise ValueError("cDFT requires disjoint finite QM/environment and explicit point charges")
    elif raw["charges"] is not None:
        raise ValueError("Isolated cDFT cannot contain embedding charges")
    if (raw["sampled_state"] not in ROLES or not isinstance(raw["states"], dict)
            or set(raw["states"]) != set(ROLES)):
        raise ValueError("cDFT pair requires R and P and an explicit sampled state")
    for role, target in zip(ROLES, (-1., 1.), strict=True):
        state = raw["states"][role]
        if (not isinstance(state, dict) or set(state) != {"target", "density", "multiplier", "spin_population", "spin_square"}
                or type(state["target"]) not in (int, float) or state["target"] != target):
            raise ValueError("cDFT R/P target or seed state is invalid")
        dm = np.asarray(state["density"])
        if (dm.ndim != 3 or dm.shape[0] != 2 or dm.shape[1] != dm.shape[2]
                or dm.shape[1] == 0 or dm.dtype.kind not in "fiu" or not np.isfinite(dm).all()
                or not np.allclose(dm, dm.transpose(0, 2, 1), atol=1e-12, rtol=0)):
            raise ValueError("cDFT seed requires two finite square spin densities")
        for name in ("multiplier", "spin_population", "spin_square"):
            if type(state[name]) not in (int, float) or not np.isfinite(state[name]):
                raise ValueError("cDFT seed state feature must be finite")
        if (abs(state["multiplier"]) > raw["settings"]["max_multiplier"]
                or not .749 <= state["spin_square"] <= raw["settings"]["spin_square_max"]):
            raise ValueError("cDFT seed exceeds the multiplier or doublet guard")
    if np.shape(raw["states"]["R"]["density"]) != np.shape(raw["states"]["P"]["density"]):
        raise ValueError("cDFT pair densities must use the same AO basis")
    if not isinstance(raw["runtime"], dict) or set(raw["runtime"]) != {"sources", "libraries", "versions"}:
        raise ValueError("cDFT reference requires actual source and runtime identities")
    if not isinstance(raw["atomic_references"], dict) or set(raw["atomic_references"]) != set(qm.symbols):
        raise ValueError("cDFT proatom species differ from the QM region")
    runtime = raw["runtime"]
    if (not isinstance(runtime["sources"], dict) or set(runtime["sources"]) != set(RUNTIME_SOURCES)
            or not isinstance(runtime["libraries"], dict) or not runtime["libraries"]
            or not isinstance(runtime["versions"], dict)
            or set(runtime["versions"]) != {"pyscf", "numpy", "scipy"}):
        raise ValueError("cDFT reference runtime identity is incomplete")
    if (any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
            for values in (runtime["sources"], runtime["libraries"]) for v in values.values())
            or any(not isinstance(v, str) or not v for v in runtime["versions"].values())):
        raise ValueError("cDFT reference runtime hashes or versions are invalid")
    for data in raw["atomic_references"].values():
        if (not isinstance(data, dict) or set(data) != {"basis", "density"}
                or not isinstance(data["basis"], list) or not data["basis"]):
            raise ValueError("cDFT proatom requires explicit basis and density")
        dm = np.asarray(data["density"])
        if dm.ndim != 2 or dm.shape[0] == 0 or dm.shape[0] != dm.shape[1]:
            raise ValueError("cDFT proatom density must be a nonempty square matrix")
        numeric(data["density"], dm.shape, "proatom density")
        if not np.allclose(dm, dm.T, atol=1e-12, rtol=0):
            raise ValueError("cDFT proatom density must be symmetric")
    return deepcopy(raw)


def validate_diagnostics(data, reference, role):
    """Audit physical, population and independent spin diagnostics of one state."""
    scalars = {"target", "population", "residual", "multiplier_hartree",
               "residual_energy_hartree", "physical_energy_hartree",
               "augmented_energy_hartree", "spin_square", "spin_population",
               "lowdin_signed_population", "lowdin_signed_spin"}
    arrays = {"density_electron_counts", "lowdin_atom_populations", "lowdin_atom_spins"}
    if (not isinstance(data, dict)
            or set(data) != scalars | arrays | {"electron_counts", "reference_sha256", "iterations"}):
        raise ValueError("cDFT diagnostics are incomplete")
    for name in scalars:
        numeric(data[name], (), name)
    count = len(reference["qm_geometry"]["atom_ids"])
    traces = numeric(data["density_electron_counts"], (2,), "electron traces")
    populations = numeric(data["lowdin_atom_populations"], (count,), "atomic populations")
    spins = numeric(data["lowdin_atom_spins"], (count,), "atomic spins")
    options, seed = reference["settings"], reference["states"][role]
    counts = data["electron_counts"]
    if (not isinstance(counts, list) or counts != reference["electron_counts"]
            or any(type(n) is not int for n in counts)
            or not np.allclose(traces, counts, atol=1e-8, rtol=0)):
        raise ValueError("cDFT density electron counts changed")
    weights = np.asarray(reference["atom_weights"])
    checks = [(data["residual"], data["population"]-data["target"], 1e-13),
              (data["residual_energy_hartree"], data["multiplier_hartree"]*data["residual"], 1e-13),
              (data["augmented_energy_hartree"], data["physical_energy_hartree"]+data["residual_energy_hartree"], 1e-11),
              (populations.sum(), sum(counts), 1e-8),
              (spins.sum(), counts[0]-counts[1], 1e-8),
              (data["lowdin_signed_population"], weights @ populations, 1e-10),
              (data["lowdin_signed_spin"], weights @ spins, 1e-10)]
    if any(abs(a-b) > tolerance for a, b, tolerance in checks):
        raise ValueError("cDFT diagnostic identities disagree")
    if (data["reference_sha256"] != atomic_reference_digest(reference["atomic_references"])
            or data["target"] != seed["target"]
            or abs(data["residual"]) > options["constraint_tol"]
            or abs(data["multiplier_hartree"]) > options["max_multiplier"]
            or not .749 <= data["spin_square"] <= options["spin_square_max"]
            or abs(data["spin_population"]-seed["spin_population"]) > options["spin_population_tolerance"]):
        raise ValueError("cDFT constraint, proatom or state identity failed")
    iterations = data["iterations"]
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("cDFT diagnostics require actual solver iterations")
    for item in iterations:
        keys = {"multiplier_hartree", "population", "physical_energy_hartree", "orbital_residual", "scf_cycles"}
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("cDFT solver iteration is incomplete")
        for name in keys - {"scf_cycles"}:
            numeric(item[name], (), "iteration " + name)
        if (type(item["scf_cycles"]) is not int or not 1 <= item["scf_cycles"] <= options["max_cycle"]
                or not 0 <= item["orbital_residual"] <= options["conv_tol_grad"]
                or abs(item["multiplier_hartree"]) > options["max_multiplier"]):
            raise ValueError("cDFT inner solver exceeded its convergence controls")
    for name in ("multiplier_hartree", "population", "physical_energy_hartree"):
        if data[name] != iterations[-1][name]:
            raise ValueError("cDFT diagnostics differ from the final solver state")


def runtime_identity():
    """Hash actual imported solver sources and their installed native libraries."""
    from .config import file_digest
    from pyscf import lib
    import numpy
    import scipy
    import pyscf

    sources = {name: file_digest(importlib.import_module(name).__file__) for name in RUNTIME_SOURCES}
    libraries = {p.name: file_digest(p) for p in sorted(Path(lib.__file__).parent.glob("lib*.so"))}
    if not libraries:
        raise EvaluationError("No installed PySCF native libraries found for identity")
    return {"sources": sources, "libraries": libraries,
            "versions": {"pyscf": pyscf.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__}}


def _method(geometry, options, environment=None, charges=None):
    from pyscf import gto, dft, qmmm

    if geometry.periodic or (environment is not None and environment.periodic):
        raise ValueError("cDFT admits finite molecular geometries only")
    mol = gto.M(atom=list(zip(geometry.symbols, geometry.positions.tolist(), strict=True)),
                basis=options["basis"], charge=options["charge"],
                spin=options["multiplicity"]-1, verbose=0)
    if (mol.nelectron - mol.spin) % 2 or mol.nelec[0]-mol.nelec[1] != 1:
        raise ValueError("cDFT electron count and doublet multiplicity disagree")
    method = dft.UKS(mol)
    method.xc = options["xc"]
    method.grids.level = options["grid_level"]
    method.grids.prune = options["grid_prune"]
    method.small_rho_cutoff = 0
    method.conv_tol = options["conv_tol"]
    method.conv_tol_grad = options["conv_tol_grad"]
    method.max_cycle = options["max_cycle"]
    method.chkfile = None
    if environment is not None:
        method = qmmm.mm_charge(method, environment.positions, charges.charges_e, unit="Angstrom")
    return method


def _references(records):
    from pyscf import gto

    result = {}
    for symbol, data in records.items():
        if set(data) != {"basis", "density"}:
            raise ValueError("cDFT atomic reference needs basis and density")
        atom = gto.M(atom=[(symbol, (0, 0, 0))], basis={symbol: data["basis"]},
                     charge=0, spin=gto.charge(symbol) % 2, verbose=0)
        dm = np.asarray(data["density"], dtype=float)
        if (dm.shape != (atom.nao, atom.nao) or not np.isfinite(dm).all()
                or not np.allclose(dm, dm.T, atol=1e-12, rtol=0)
                or abs(np.einsum("ij,ji", dm, atom.intor("int1e_ovlp"))-atom.nelectron) > 1e-8):
            raise ValueError("cDFT frozen proatom density or electron count is invalid")
        result[symbol] = (atom, dm)
    return result


def _solve(method, weight, target, options, seed=None):
    from pyscf.dft import cdft

    return cdft.solve(method, weight, target,
                      dm0=None if seed is None else seed["density"],
                      multiplier0=0. if seed is None else seed["multiplier"],
                      population_tol=options["constraint_tol"],
                      max_multiplier=options["max_multiplier"],
                      max_iterations=options["max_constraint_iterations"],
                      inner_solver=options["inner_solver"], inner_lindep=options["inner_lindep"])


def build_reference(qm_geometry, options, atom_weights, *, environment=None, charges=None,
                    sampled_state="R", initial_states=None, verbose=0):
    """Prepare both real constrained states and freeze their deterministic seeds."""
    from pyscf import lib
    from pyscf.dft import cdft

    options = settings(options)
    if type(verbose) is not int or not 0 <= verbose <= 5:
        raise ValueError("cDFT preparation verbosity must be an integer from zero to five")
    method = _method(qm_geometry, options, environment, charges)
    method.verbose = verbose
    references = cdft.atomic_references(method.mol)
    weight = cdft.Hirshfeld(method.mol, atom_weights, method.grids, references)
    states = {}
    for role, target in zip(ROLES, (-1., 1.), strict=True):
        lib.logger.info(method, "Preparing constrained state %s with target %.8g", role, target)
        result = _solve(method, weight, target, options,
                        None if initial_states is None else initial_states[role])
        d = result.diagnostics()
        lib.logger.info(method, "Prepared %s physical energy %.15g residual %.8g multiplier %.12g",
                        role, result.energy, result.residual, result.multiplier)
        if not .749 <= d["spin_square"] <= options["spin_square_max"]:
            raise EvaluationError("cDFT seed does not satisfy the doublet spin guard")
        states[role] = {"target": target, "density": result.density.tolist(),
                        "multiplier": result.multiplier,
                        "spin_population": d["spin_population"], "spin_square": d["spin_square"]}
    record = {"schema_version": 1, "form": FORM, "settings": options,
              "qm_geometry": _record(qm_geometry),
              "environment_geometry": None if environment is None else _record(environment),
              "charges": None if charges is None else {
                  "atom_ids": list(charges.atom_ids), "charges_e": charges.charges_e.tolist(),
                  "source": charges.source},
              "atom_weights": list(atom_weights),
              "atomic_references": {s: {"basis": atom._basis[s], "density": dm.tolist()}
                                    for s, (atom, dm) in references.items()},
              "states": states, "sampled_state": sampled_state, "electron_counts": list(method.mol.nelec),
              "runtime": runtime_identity()}
    record["sha256"] = digest(record)
    return validate_reference(record, electrostatic=environment is not None)


class CDFTProvider:
    """Two same-frame constrained QM contributions; never environment DPA or LJ."""

    def __init__(self, options, reference, charges=None):
        self.reference = validate_reference(reference, electrostatic=charges is not None)
        self.options = settings(options)
        if self.options != self.reference["settings"]:
            raise ValueError("cDFT settings differ from the pinned reference")
        if runtime_identity() != self.reference["runtime"]:
            raise EvaluationError("cDFT actual solver source or native libraries differ from the reference")
        self.charges = charges
        expected = self.reference["charges"]
        actual = None if charges is None else {"atom_ids": list(charges.atom_ids),
                 "charges_e": charges.charges_e.tolist(), "source": charges.source}
        if actual != expected or (charges is not None and charges.radii_angstrom is not None):
            raise ValueError("cDFT embedding identity differs from the reference")
        self.references = _references(self.reference["atomic_references"])
        self.clear()

    def clear(self):
        self.last = None

    def evaluate(self, geometry, environment=None):
        self.clear()
        from pyscf.dft import cdft

        for actual, name in ((geometry, "qm_geometry"), (environment, "environment_geometry")):
            expected = self.reference[name]
            if expected is None:
                if actual is not None:
                    raise ValueError("cDFT field presence changed")
            elif (actual is None or actual.periodic
                  or list(actual.atom_ids) != expected["atom_ids"]
                  or list(actual.symbols) != expected["symbols"]):
                raise ValueError("cDFT fixed fragment or environment identity changed")
        try:
            method = _method(geometry, self.options, environment, self.charges)
            if list(method.mol.nelec) != self.reference["electron_counts"]:
                raise ValueError("cDFT total electron counts differ from the reference")
            for symbol, (atom, _) in self.references.items():
                if atom._basis[symbol] != method.mol._basis[symbol]:
                    raise ValueError("cDFT frozen proatom basis differs from the molecular basis")
            overlap = method.get_ovlp()
            for seed in self.reference["states"].values():
                dm = numeric(seed["density"], (2, method.mol.nao, method.mol.nao), "seed density")
                if not np.allclose(np.einsum("sij,ji->s", dm, overlap), method.mol.nelec, atol=1e-8, rtol=0):
                    # A seed at a different geometry is a guess, not an electron-count
                    # assertion in the displaced AO metric. Check only its reference metric.
                    reference_method = _method(_geometry(self.reference["qm_geometry"]), self.options)
                    reference_overlap = reference_method.get_ovlp()
                    if not np.allclose(np.einsum("sij,ji->s", dm, reference_overlap),
                                       method.mol.nelec, atol=1e-8, rtol=0):
                        raise ValueError("cDFT seed density violates its reference electron counts")
            weight = cdft.Hirshfeld(method.mol, self.reference["atom_weights"],
                                    method.grids, self.references)
            states, diagnostics = {}, {}
            for role in ROLES:
                seed = self.reference["states"][role]
                result = _solve(method, weight, seed["target"], self.options, seed)
                d = result.diagnostics()
                validate_diagnostics(d, self.reference, role)
                if (not .749 <= d["spin_square"] <= self.options["spin_square_max"]
                        or abs(d["spin_population"]-seed["spin_population"]) >
                        self.options["spin_population_tolerance"]):
                    raise EvaluationError("cDFT state lost its pinned spin/localization identity")
                force = -result.gradients()*HARTREE_EV/BOHR_ANGSTROM
                if environment is None:
                    value = Evaluation(result.energy*HARTREE_EV, force)
                else:
                    env_force = -result.environment_gradients()*HARTREE_EV/BOHR_ANGSTROM
                    value = EmbeddedEvaluation(result.energy*HARTREE_EV, geometry.atom_ids,
                                               environment.atom_ids, force, env_force, d)
                states[role], diagnostics[role] = value, d
            self.last = {"geometry_sha256": geometry_digest(geometry, environment),
                         "reference_sha256": self.reference["sha256"],
                         "states": states, "diagnostics": diagnostics}
            return states[self.reference["sampled_state"]]
        except BaseException:
            self.clear()
            raise
