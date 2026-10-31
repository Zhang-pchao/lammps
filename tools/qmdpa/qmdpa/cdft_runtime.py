#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Same-frame constrained pair over the existing additive native lifecycle."""

from copy import deepcopy

import numpy as np

from .cdft import METHOD, ROLES, HARTREE_EV, numeric, validate_diagnostics
from .core import Evaluation
from .electronic_gap import gap_frame, native_geometry
from .named_cv import NamedCVSchema, digest
from .state_records import geometry_digest


def enabled(config):
    return config.get("provider", {}).get("state", {}).get("method") == METHOD


def full_region(config):
    q, m = config["qm_region"], config["environment_region"]
    return {key: q[key]+m[key] for key in ("atom_ids", "symbols")}


def schema(config):
    ref = config["provider"]["state"]["reference"]
    return NamedCVSchema({"schema_version": 1, "coordinate_unit": "angstrom",
                          "variables": [{"name": "cdft_gap", "shape": [], "unit": "eV",
                                         "dependencies": {"reference": ref["sha256"],
                                                          "model": config["model_sha256"],
                                                          "cross": digest(config["cross"])}}]})


class CDFTPairCalculator:
    """Share same-call DPA/cross terms; existing AdditiveCalculator assembles totals."""

    def __init__(self, config, provider, environment, cross, torch=None):
        self.config = deepcopy(config)
        self.provider, self.environment, self.cross, self.torch = provider, environment, cross, torch
        # The umbrella owns one static restraint over the constrained gap. It is
        # installed by the runtime initializer and adds its energy and force
        # exactly once to the sampled surface total.
        self.bias = None
        self.clear()

    def clear(self):
        self.last = None
        self.provider.clear()

    def evaluate(self, geometry, step=0):
        from .additive import AdditiveCalculator
        from .inference import strict_runtime_state
        from .multistate_coupling import _CallCache, _SurfaceBackend

        self.clear()
        cfg = self.config
        q = geometry.select(cfg["qm_region"]["atom_ids"])
        m = geometry.select(cfg["environment_region"]["atom_ids"])
        try:
            if self.torch is not None:
                strict_runtime_state(self.torch)
            self.provider.evaluate(q, m)
            quantum = self.provider.last
            cache = _CallCache()
            terms = {}
            for role in ROLES:
                surface = _SurfaceBackend(quantum["states"][role], q, m, self.provider.charges)
                terms[role] = AdditiveCalculator(
                    cache.wrap(self.environment), surface, cache.wrap(self.cross),
                    qm_atom_ids=q.atom_ids, environment_atom_ids=m.atom_ids).evaluate_terms(geometry)
            states = {role: term.total for role, term in terms.items()}
            cv = gap_frame(schema(cfg), geometry, states, ROLES)
            if self.torch is not None:
                strict_runtime_state(self.torch)
            self.last = (geometry, terms, quantum, cv)
            sampled = states[cfg["provider"]["state"]["reference"]["sampled_state"]]
            if self.bias is None:
                return sampled
            # The umbrella is a static restraint on the observed gap. Its energy
            # and force are added to the sampled surface exactly once.
            bias = self.bias.evaluate(cv, step)
            total = Evaluation(sampled.energy+bias["bias_energy_eV"],
                               sampled.forces+np.asarray(bias["forces_eV_per_angstrom"]))
            return total
        except BaseException:
            self.clear()
            raise


def make_calculator(config):
    from .additive import CrossLennardJones
    from .additive_config import cross_parameters
    from .backends import DeepMDBackend
    from .embedding import FixedEmbeddingCharges
    from .inference import prepare_raw_inference, strict_runtime_state
    from .providers import make_qm_provider

    torch = prepare_raw_inference(config["model"])
    from deepmd.pt.utils.env import DEVICE

    if DEVICE.type != "cpu":
        raise ValueError("The cDFT candidate requires actual CPU DPA inference")
    low = DeepMDBackend(config["model"], charge_spin=config["environment_charge_spin"], no_jit=True)
    provider = make_qm_provider(config["qm"], provider=config["provider"],
                                charges=FixedEmbeddingCharges(**config["embedding"]))
    cross = CrossLennardJones(cross_parameters(config["cross"]), source=config["cross"]["source"])
    config["inference_runtime"] = {**strict_runtime_state(torch), "device": str(DEVICE)}
    return CDFTPairCalculator(config, provider, low, cross, torch)


def make_coupling(lmp, comm, config):
    from .lammps import AdditiveExternalCoupling
    from .region import FixedRegion
    from .runtime import _root_call

    calculator = None

    def build():
        nonlocal calculator
        calculator = make_calculator(config)

    _root_call(comm, build)
    full = full_region(config)

    class CDFTExternalCoupling(AdditiveExternalCoupling):
        """Pass the accepted step so the umbrella can advance its own history."""

        _calculator_kind = "constrained-pair"

        def _evaluate_geometry(self, geometry):
            return self.calculator.evaluate(geometry, int(self.lmp.extract_global("ntimestep")))

    return CDFTExternalCoupling(
        lmp, FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"])), calculator,
        type_symbols=config["type_symbols"], comm=comm, state_observer=calculator,
        fix_id="qmdpa_cdft_pair")


def _quantum_record(value, diagnostics):
    return {"energy_eV": value.energy_ev,
            "qm_forces_eV_per_angstrom": value.qm_forces_ev_per_angstrom.tolist(),
            "environment_forces_eV_per_angstrom": value.environment_forces_ev_per_angstrom.tolist(),
            "diagnostics": deepcopy(diagnostics)}


def attach_frame(frame, coupling, config):
    if not enabled(config):
        return
    if coupling.failure is not None or coupling.last_step != frame["step"] or coupling.calculator.last is None:
        raise ValueError("cDFT acceptance requires a successful current callback")
    geometry, terms, quantum, cv = coupling.calculator.last
    current = native_geometry(frame, full_region(config))
    if (geometry.atom_ids != current.atom_ids
            or not np.array_equal(geometry.positions, current.positions)):
        raise ValueError("cDFT records differ from the accepted coordinates")
    first = terms[ROLES[0]]
    record = {"schema_version": 1, "reference_sha256": quantum["reference_sha256"],
              "geometry_sha256": quantum["geometry_sha256"],
              "states": {role: {
                  "energy_eV": terms[role].total.energy,
                  "forces_eV_per_angstrom": terms[role].total.forces.tolist(),
                  "quantum": _quantum_record(quantum["states"][role], quantum["diagnostics"][role])}
                         for role in ROLES},
              "shared": {
                  "environment_energy_eV": first.environment.energy,
                  "environment_forces_eV_per_angstrom": first.environment.forces.tolist(),
                  "cross_energy_eV": first.cross.energy_ev,
                  "cross_qm_forces_eV_per_angstrom": first.cross.qm_forces_ev_per_angstrom.tolist(),
                  "cross_environment_forces_eV_per_angstrom": first.cross.environment_forces_ev_per_angstrom.tolist()},
              "cv": cv.record()}
    record["sha256"] = digest(record)
    candidate = {**frame, "cdft_states": record}
    try:
        validate_frame(candidate, config)
    except BaseException:
        coupling.calculator.clear()
        raise
    frame["cdft_states"] = record


def _close(a, b, tol, label):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.allclose(x, y, atol=tol, rtol=0)):
        raise ValueError("cDFT " + label + " differs beyond tolerance")


def validate_frame(frame, config):
    record = frame.get("cdft_states")
    if not enabled(config):
        if record is not None:
            raise ValueError("Frame contains an undeclared constrained state")
        return
    ref = config["provider"]["state"]["reference"]
    expected = {"schema_version", "reference_sha256", "geometry_sha256",
                "states", "shared", "cv", "sha256"}
    if (not isinstance(record, dict) or set(record) != expected or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["reference_sha256"] != ref["sha256"]
            or record["sha256"] != digest({k: v for k, v in record.items() if k != "sha256"})
            or not isinstance(record["states"], dict) or set(record["states"]) != set(ROLES)):
        raise ValueError("cDFT pair record is missing or corrupt")
    geometry = native_geometry(frame, full_region(config))
    q = geometry.select(config["qm_region"]["atom_ids"])
    m = geometry.select(config["environment_region"]["atom_ids"])
    if record["geometry_sha256"] != geometry_digest(q, m):
        raise ValueError("cDFT state coordinates do not match the accepted frame")
    shared = record["shared"]
    expected_shared = {"environment_energy_eV", "environment_forces_eV_per_angstrom",
                       "cross_energy_eV", "cross_qm_forces_eV_per_angstrom",
                       "cross_environment_forces_eV_per_angstrom"}
    if not isinstance(shared, dict) or set(shared) != expected_shared:
        raise ValueError("cDFT shared additive terms are incomplete")
    for name in expected_shared:
        shape = () if name.endswith("energy_eV") else (q.positions.shape if "_qm_" in name else m.positions.shape)
        numeric(shared[name], shape, name)
    totals = {}
    qidx = [geometry.atom_ids.index(i) for i in q.atom_ids]
    midx = [geometry.atom_ids.index(i) for i in m.atom_ids]
    for role in ROLES:
        state = record["states"][role]
        if not isinstance(state, dict) or set(state) != {"energy_eV", "forces_eV_per_angstrom", "quantum"}:
            raise ValueError("cDFT state total is incomplete")
        numeric(state["energy_eV"], (), "total energy")
        numeric(state["forces_eV_per_angstrom"], geometry.positions.shape, "total forces")
        quantum = state["quantum"]
        if (not isinstance(quantum, dict) or set(quantum) != {
                "energy_eV", "qm_forces_eV_per_angstrom", "environment_forces_eV_per_angstrom", "diagnostics"}):
            raise ValueError("cDFT QM contribution is incomplete")
        numeric(quantum["energy_eV"], (), "QM energy")
        numeric(quantum["qm_forces_eV_per_angstrom"], q.positions.shape, "QM forces")
        numeric(quantum["environment_forces_eV_per_angstrom"], m.positions.shape, "environment forces")
        d = quantum["diagnostics"]
        validate_diagnostics(d, ref, role)
        _close(quantum["energy_eV"], d["physical_energy_hartree"]*HARTREE_EV, 1e-10, "physical energy")
        expected_energy = quantum["energy_eV"]+shared["environment_energy_eV"]+shared["cross_energy_eV"]
        force = np.zeros_like(geometry.positions)
        force[qidx] = (np.asarray(quantum["qm_forces_eV_per_angstrom"])
                       +np.asarray(shared["cross_qm_forces_eV_per_angstrom"]))
        force[midx] = (np.asarray(quantum["environment_forces_eV_per_angstrom"])
                       +np.asarray(shared["environment_forces_eV_per_angstrom"])
                       +np.asarray(shared["cross_environment_forces_eV_per_angstrom"]))
        _close(state["energy_eV"], expected_energy, 1e-10, "additive energy assembly")
        _close(state["forces_eV_per_angstrom"], force, 1e-10, "additive force assembly")
        totals[role] = Evaluation(state["energy_eV"], state["forces_eV_per_angstrom"])
    cv = gap_frame(schema(config), geometry, totals, ROLES)
    if record["cv"] != cv.record():
        raise ValueError("cDFT same-frame gap or Jacobian differs from complete states")
    total = totals[ref["sampled_state"]]
    # The cDFT provider owns the sampled surface only. When a bias owner is
    # active, the native total is the sampled surface plus that bias, and the
    # bias owner is responsible for asserting that assembly. Checking the
    # bare sampled surface against the native total here would contradict the
    # bias owner and would reject every biased run.
    from .cdft_umbrella import enabled as umbrella_enabled

    if umbrella_enabled(config):
        # The umbrella owner validates the native total against the sampled
        # surface plus its own restraint. Nothing is asserted here, so that
        # the two owners do not both claim the native total.
        return
    _close(frame["potential_eV"], total.energy, 1e-7, "native sampled energy")
    atoms = {a["id"]: a for a in frame["atoms"]}
    _close([atoms[i]["f"] for i in geometry.atom_ids], total.forces, 1e-5, "native sampled force")


def compare_restart(frame, reference, config):
    validate_frame(frame, config)
    validate_frame(reference, config)
    if not enabled(config):
        return
    for role in ROLES:
        a, b = frame["cdft_states"]["states"][role], reference["cdft_states"]["states"][role]
        _close(a["energy_eV"], b["energy_eV"], 1e-7, "restart state energy")
        _close(a["forces_eV_per_angstrom"], b["forces_eV_per_angstrom"], 1e-5, "restart state force")
        for key in ("population", "spin_population", "spin_square"):
            _close(a["quantum"]["diagnostics"][key], b["quantum"]["diagnostics"][key], 1e-6,
                   "restart "+key)
