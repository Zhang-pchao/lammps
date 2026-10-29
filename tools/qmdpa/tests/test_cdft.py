#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""cDFT admission and independent synthetic complete-state assembly controls."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from qmdpa.cdft import (FORM, METHOD, HARTREE_EV, RUNTIME_SOURCES, settings,
                        validate_reference, atomic_reference_digest, CDFTProvider)
from qmdpa.cdft_runtime import CDFTPairCalculator, attach_frame, validate_frame, compare_restart
from qmdpa.core import Evaluation, Geometry
from qmdpa.embedding import EmbeddedEvaluation, FixedEmbeddingCharges
from qmdpa.named_cv import digest
from qmdpa.providers import validate_provider
from qmdpa.state_records import geometry_digest


def options():
    return {"xc": "pbe0", "basis": "def2-svpd", "charge": 1, "multiplicity": 2,
            "grid_level": 2, "grid_prune": None, "grid_response": True, "conv_tol": 1e-12,
            "conv_tol_grad": 1e-7, "max_cycle": 100, "constraint_tol": 1e-10,
            "max_multiplier": 1., "max_constraint_iterations": 40,
            "spin_square_max": 1., "spin_population_tolerance": .2,
            "inner_solver": "newton", "inner_lindep": 1e-24}


def fixture():
    g = Geometry([11, 12, 13, 31, 32, 33, 51, 52, 53], ["O", "H", "H"]*3,
                 np.arange(27).reshape(9, 3)*.1)
    q, m = g.select(g.atom_ids[:6]), g.select(g.atom_ids[6:])
    def region(a):
        return {"atom_ids": list(a.atom_ids), "symbols": list(a.symbols),
                "molecules": [list(a.atom_ids)]}
    def record(a):
        return {"atom_ids": list(a.atom_ids), "symbols": list(a.symbols),
                "positions_angstrom": a.positions.tolist()}
    charges = {"atom_ids": list(m.atom_ids), "charges_e": [-.8, .4, .4], "source": "synthetic"}
    ref = {"schema_version": 1, "form": FORM, "settings": options(), "qm_geometry": record(q),
           "environment_geometry": record(m), "charges": charges, "atom_weights": [1]*3+[-1]*3,
           "atomic_references": {s: {"basis": [[0, [1., 1.]]], "density": [[float(n)]]}
                                 for s, n in (("H", 1), ("O", 8))},
           "states": {r: {"target": t, "density": [[[1.]], [[0.]]], "multiplier": .1,
                           "spin_population": -t, "spin_square": .75}
                      for r, t in (("R", -1.), ("P", 1.))},
           "sampled_state": "R", "runtime": {"sources": {n: "a"*64 for n in RUNTIME_SOURCES},
                                        "libraries": {"libcgto.so": "b"*64},
                                        "versions": {n: "test" for n in ("pyscf", "numpy", "scipy")}},
           "electron_counts": [10, 9]}
    ref["sha256"] = digest(ref)
    cfg = {"qm": options(), "provider": {"engine": "pyscf", "device": "cpu",
                                        "state": {"method": METHOD, "reference": ref}},
           "qm_region": region(q), "environment_region": region(m),
           "model_sha256": "a"*64, "cross": {"source": "synthetic"}}
    return g, cfg, FixedEmbeddingCharges(**charges)


def test_explicit_open_shell_admission():
    _, cfg, _ = fixture()
    assert validate_provider(cfg["provider"], electrostatic=True) == cfg["provider"]
    bad = deepcopy(cfg["provider"])
    bad["device"] = "cuda"
    with pytest.raises(ValueError, match="CPU"):
        validate_provider(bad, electrostatic=True)
    for key, value in (("charge", 0), ("multiplicity", 1), ("grid_response", False),
                       ("constraint_tol", 1e-4), ("density_fit", True)):
        bad = options()
        bad[key] = value
        with pytest.raises(ValueError):
            settings(bad)


class Pair:
    def __init__(self, cfg, charges):
        self.reference = cfg["provider"]["state"]["reference"]
        self.charges = charges
        self.fail = False
        self.clear()

    def clear(self):
        self.last = None

    def evaluate(self, q, m):
        self.clear()
        states, diagnostics = {}, {}
        for role, k, target in (("R", 1., -1.), ("P", 2., 1.)):
            if self.fail and role == "P":
                raise RuntimeError("constraint failed")
            e = float(k*(np.sum(q.positions**2)+np.sum(m.positions**2)))
            d = {"target": target, "population": target, "residual": 0.,
                 "multiplier_hartree": .1, "residual_energy_hartree": 0.,
                 "physical_energy_hartree": e/HARTREE_EV,
                 "spin_square": .75, "spin_population": -target, "electron_counts": [10, 9]}
            populations = np.array([3., 3., 3., 3., 3., 4.])
            spins = np.array([1., 0., 0., 0., 0., 0.])
            d.update(density_electron_counts=[10., 9.],
                     lowdin_atom_populations=populations.tolist(), lowdin_atom_spins=spins.tolist(),
                     lowdin_signed_population=float(np.dot(self.reference["atom_weights"], populations)),
                     lowdin_signed_spin=float(np.dot(self.reference["atom_weights"], spins)),
                     augmented_energy_hartree=e/HARTREE_EV,
                     reference_sha256=atomic_reference_digest(self.reference["atomic_references"]),
                     iterations=[{"multiplier_hartree": .1, "population": target,
                                  "physical_energy_hartree": e/HARTREE_EV,
                                  "orbital_residual": 1e-10, "scf_cycles": 2}])
            states[role] = EmbeddedEvaluation(e, q.atom_ids, m.atom_ids,
                                              -2*k*q.positions, -2*k*m.positions, d)
            diagnostics[role] = d
        self.last = {"states": states, "diagnostics": diagnostics,
                     "geometry_sha256": geometry_digest(q, m),
                     "reference_sha256": self.reference["sha256"]}
        return states["R"]


class Low:
    def __init__(self):
        self.calls = 0

    def evaluate(self, g):
        self.calls += 1
        return Evaluation(float(3*np.sum(g.positions**2)), -6*g.positions)


class Cross:
    def evaluate(self, q, m):
        return EmbeddedEvaluation(0., q.atom_ids, m.atom_ids,
                                  np.zeros_like(q.positions), np.zeros_like(m.positions))


def accepted():
    geometry, config, charges = fixture()
    low = Low()
    pair = Pair(config, charges)
    calc = CDFTPairCalculator(config, pair, low, Cross())
    value = calc.evaluate(geometry)
    frame = {"step": 0, "potential_eV": value.energy,
             "atoms": [{"id": i, "x": x.tolist(), "f": f.tolist(), "image": [0, 0, 0]}
                       for i, x, f in zip(geometry.atom_ids, geometry.positions, value.forces, strict=True)]}
    attach_frame(frame, SimpleNamespace(failure=None, last_step=0, calculator=calc), config)
    return geometry, config, low, pair, calc, frame


def test_same_frame_complete_pair_and_shared_environment():
    g, cfg, low, _, _, frame = accepted()
    assert low.calls == 1
    validate_frame(frame, cfg)
    compare_restart(frame, deepcopy(frame), cfg)
    record = frame["cdft_states"]
    assert record["cv"]["values"] == pytest.approx([float(np.sum(g.positions**2))], abs=1e-12)
    np.testing.assert_allclose(record["cv"]["jacobian_per_angstrom"], [2*g.positions])


@pytest.mark.parametrize("damage", ["force", "count", "target", "gap", "physical_energy"])
def test_rehashed_corruption_is_rejected(damage):
    _, cfg, _, _, _, frame = accepted()
    bad = deepcopy(frame)
    record = bad["cdft_states"]
    state = record["states"]["R"]
    if damage == "force":
        state["forces_eV_per_angstrom"][0][0] += 1.
    elif damage == "count":
        state["quantum"]["diagnostics"]["electron_counts"] = [11, 10]
    elif damage == "target":
        state["quantum"]["diagnostics"]["target"] = 1.
    elif damage == "gap":
        record["cv"]["values"][0] *= -1
    else:
        state["quantum"]["diagnostics"]["physical_energy_hartree"] += .1
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_frame(bad, cfg)


def test_failed_second_state_clears_pair_and_preserves_accepted_record():
    g, cfg, _, pair, calc, frame = accepted()
    old = deepcopy(frame)
    pair.fail = True
    with pytest.raises(RuntimeError, match="constraint failed"):
        calc.evaluate(g)
    assert calc.last is None and pair.last is None
    assert frame == old
    validate_frame(frame, cfg)



@pytest.mark.parametrize("damage", ["electron_trace", "spin_sum", "iteration", "proatom",
                                    "array_shape", "string", "schema_bool", "extra", "spin_string"])
def test_diagnostic_damage_cannot_pass_by_rehashing(damage):
    _, cfg, _, _, _, frame = accepted()
    record = frame["cdft_states"]
    d = record["states"]["R"]["quantum"]["diagnostics"]
    if damage == "electron_trace":
        d["density_electron_counts"][0] += 1
    elif damage == "spin_sum":
        d["lowdin_atom_spins"][0] += .01
    elif damage == "iteration":
        d["iterations"][-1]["population"] += .01
    elif damage == "proatom":
        d["reference_sha256"] = "d"*64
    elif damage == "array_shape":
        record["shared"]["cross_qm_forces_eV_per_angstrom"] = [[0., 0., 0.]]
    elif damage == "string":
        record["shared"]["cross_energy_eV"] = "0"
    elif damage == "spin_string":
        d["spin_population"] = str(d["spin_population"])
    elif damage == "schema_bool":
        record["schema_version"] = True
    else:
        d["ignored"] = 1
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_frame(frame, cfg)


@pytest.mark.parametrize("damage", ["runtime", "hash", "spin", "density", "atomic", "settings", "count"])
def test_invalid_frozen_references_rejected(damage):
    _, cfg, _ = fixture()
    ref = cfg["provider"]["state"]["reference"]
    if damage == "runtime":
        ref["runtime"]["sources"] = {}
    elif damage == "hash":
        ref["runtime"]["libraries"]["libcgto.so"] = "unknown"
    elif damage == "spin":
        ref["states"]["R"]["spin_square"] = 1.5
    elif damage == "density":
        ref["states"]["R"]["density"] = [[[1., 1.], [0., 1.]]]*2
    elif damage == "atomic":
        ref["atomic_references"]["H"]["density"] = [["1"]]
    elif damage == "settings":
        ref["settings"]["multiplicity"] = True
    else:
        ref["electron_counts"] = [10.0, 9.0]
    ref["sha256"] = digest({k: v for k, v in ref.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_reference(ref, electrostatic=True)


def test_provider_rejects_actual_runtime_mismatch_before_solver(monkeypatch):
    from qmdpa import cdft
    _, cfg, charges = fixture()
    monkeypatch.setattr(cdft, "runtime_identity", lambda: {})
    with pytest.raises(RuntimeError, match="actual solver"):
        CDFTProvider(cfg["qm"], cfg["provider"]["state"]["reference"], charges)


def test_legacy_mechanical_entry_rejects_cdft(tmp_path):
    import json
    from qmdpa.config import file_digest, load_config

    _, cfg, _ = fixture()
    ref = cfg["provider"]["state"]["reference"]
    ref["environment_geometry"] = ref["charges"] = None
    ref["sha256"] = digest({k: v for k, v in ref.items() if k != "sha256"})
    model, data = tmp_path/"model.pt", tmp_path/"atoms.data"
    model.write_bytes(b"unused model")
    data.write_bytes(b"unused atom data")
    raw = {"schema_version": 1, "data_file": str(data), "model": str(model),
           "model_sha256": file_digest(model), "type_symbols": ["O", "H"],
           "periodic": False, "region": cfg["qm_region"], "qm": cfg["qm"],
           "provider": cfg["provider"], "timestep_fs": .02, "steps": 0}
    path = tmp_path/"mechanical.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="finite additive NVE"):
        load_config(path)
