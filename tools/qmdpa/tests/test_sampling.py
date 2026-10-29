#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Configuration and restart contracts for native sampling, without a backend."""

import hashlib
from copy import deepcopy

import numpy as np
import pytest
from test_dispatch import mechanical, save

from qmdpa import dispatch
from qmdpa.config import load_config
from qmdpa.dynamics import (
    bind_thermostat,
    compare_thermostat_restart,
    integrator_command,
    validate_thermostat_frame,
)
from qmdpa.sampling import (
    bind_sampling,
    compare_sampling_restart,
    cv_value,
    render_plumed,
    validate_sampling_frame,
)


def request(tmp_path, form="distance-difference"):
    kernel = tmp_path / "libplumedKernel.so"
    kernel.write_bytes(b"synthetic kernel identity; never loaded")
    return {
        "engine": "plumed",
        "kernel": {"path": kernel.name, "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest()},
        "cv": {"form": form, "atom_ids": [1, 2, 3] if form == "distance-difference" else [1, 2]},
        "bias": {"form": "harmonic", "center_angstrom": 0.0 if form == "distance-difference" else 1.0,
                 "k_ev_per_angstrom2": 0.2},
    }


def thermostat():
    return {"form": "nose-hoover-chain", "temperature_K": 300.0,
            "damping_fs": 50.0, "chain_length": 3}


def config(tmp_path):
    raw = mechanical(tmp_path)
    raw["region"] = {"atom_ids": [1, 2, 3], "symbols": ["H"]*3, "molecules": [[1, 2, 3]]}
    return dispatch.load_run_config(save(tmp_path, raw))


@pytest.mark.parametrize("form", ["distance", "distance-difference"])
def test_sampling_unified_route_and_legacy_rejection(tmp_path, form):
    raw = mechanical(tmp_path)
    raw["region"] = {"atom_ids": [1, 2, 3], "symbols": ["H"]*3, "molecules": [[1, 2, 3]]}
    before = dispatch.load_run_config(save(tmp_path, raw))
    raw["sampling"] = request(tmp_path, form)
    raw["thermostat"] = thermostat()
    after = dispatch.load_run_config(save(tmp_path, raw))
    assert after["physics_sha256"] != before["physics_sha256"]
    assert after["sampling"]["cv"]["form"] == form
    description = dispatch.describe_run(after)
    assert description["assembly"]["native_pair"] == "deepmd"
    assert description["assembly"]["additional_native_bias"] == "plumed"
    assert description["dynamics"]["thermostat"] == thermostat()
    with pytest.raises(ValueError, match="Unknown"):
        load_config(save(tmp_path, raw))


def test_default_nve_and_constant_nvt_command(tmp_path):
    cfg = config(tmp_path)
    assert integrator_command(cfg) == "fix qmdpa_integrator all nve"
    bind_thermostat(cfg, thermostat())
    assert integrator_command(cfg).endswith("nvt temp 300 300 0.050000000000000003 tchain 3")


@pytest.mark.parametrize("key,value", [
    ("temperature_K", 0), ("temperature_K", True), ("temperature_K", float("nan")),
    ("damping_fs", -1), ("damping_fs", "50"), ("damping_fs", float("inf")),
    ("chain_length", True), ("chain_length", 0), ("chain_length", 11),
    ("chain_length", 1.5), ("form", "langevin"), ("extra", 1),
])
def test_thermostat_invalid_requests(tmp_path, key, value):
    cfg = config(tmp_path)
    raw = thermostat()
    raw[key] = value
    with pytest.raises(ValueError):
        bind_thermostat(cfg, raw)


@pytest.mark.parametrize("key", ["temperature_K", "damping_fs", "chain_length"])
def test_changed_thermostat_enters_physics_identity(tmp_path, key):
    a, b = config(tmp_path), config(tmp_path)
    raw = thermostat()
    bind_thermostat(a, raw)
    raw[key] += 1
    bind_thermostat(b, raw)
    assert a["physics_sha256"] != b["physics_sha256"]


@pytest.mark.parametrize("change", [
    {"periodic": True},
    {"hamiltonian": "finite-electrostatic-reference-subtractive"},
    {"provider": {"engine": "gpu4pyscf", "device": "gpu", "state": {"method": "ground"}}},
    {"provider": {"engine": "pyscf", "device": "cpu", "state": {"method": "rhf-tda"}}},
])
def test_new_profile_scope_is_explicit(tmp_path, change):
    cfg = config(tmp_path)
    cfg.update(change)
    with pytest.raises(ValueError, match="ground ME"):
        bind_sampling(cfg, request(tmp_path), tmp_path)
    with pytest.raises(ValueError, match="ground ME"):
        bind_thermostat(cfg, thermostat())


@pytest.mark.parametrize("ids", [[1, 1, 2], [1, 2, 99], [True, 2, 3], [0, 2, 3], [1, 2]])
def test_cv_atoms_must_be_valid_and_protected(tmp_path, ids):
    cfg = config(tmp_path)
    raw = request(tmp_path)
    raw["cv"]["atom_ids"] = ids
    with pytest.raises(ValueError):
        bind_sampling(cfg, raw, tmp_path)


@pytest.mark.parametrize("key,value", [
    ("k_ev_per_angstrom2", 0), ("k_ev_per_angstrom2", True),
    ("k_ev_per_angstrom2", "0.2"), ("center_angstrom", float("nan")),
    ("center_angstrom", True), ("form", "metadynamics"), ("extra", 1),
])
def test_bias_inputs_are_declared(tmp_path, key, value):
    cfg, raw = config(tmp_path), request(tmp_path)
    raw["bias"][key] = value
    with pytest.raises(ValueError):
        bind_sampling(cfg, raw, tmp_path)


def test_kernel_identity_and_legacy_bias_conflict(tmp_path):
    cfg, raw = config(tmp_path), request(tmp_path)
    cfg["bias"] = {}
    with pytest.raises(ValueError, match="one bias"):
        bind_sampling(cfg, raw, tmp_path)
    del cfg["bias"]
    (tmp_path / raw["kernel"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        bind_sampling(cfg, raw, tmp_path)


def test_signed_distance_difference_and_singular_domain(tmp_path):
    raw = request(tmp_path)
    xyz = {1: np.array([0., 0., 0.]), 2: np.array([1., 0., 0.]),
           3: np.array([3., 0., 0.])}
    assert cv_value(raw["cv"], xyz) == -1.0
    raw["cv"]["atom_ids"] = [3, 2, 1]
    assert cv_value(raw["cv"], xyz) == 1.0
    xyz[2] = xyz[1]
    with pytest.raises(ValueError, match="distances"):
        cv_value(raw["cv"], xyz)


def test_renderer_uses_official_cv_bias_units_and_finite_geometry(tmp_path):
    cfg = config(tmp_path)
    bind_sampling(cfg, request(tmp_path), tmp_path)
    text = render_plumed(cfg["sampling"], tmp_path)
    assert "UNITS LENGTH=A ENERGY=eV TIME=ps" in text
    assert "DISTANCE ATOMS=1,2 NOPBC" in text
    assert "DISTANCE ATOMS=3,2 NOPBC" in text
    assert "COEFFICIENTS=1,-1 PERIODIC=NO" in text
    assert "qmdpa_bias: RESTRAINT ARG=qmdpa_cv" in text
    with pytest.raises(ValueError, match="paths"):
        render_plumed(cfg["sampling"], tmp_path / "unsafe name")


def frame():
    return {
        "potential_eV": 1.1, "kinetic_eV": 0.3,
        "atoms": [{"id": i, "x": [x, 0., 0.]} for i, x in [(1, 0.), (2, 1.), (3, 3.)]],
        "sampling": {"engine": "plumed", "cv_angstrom": -1.,
                     "bias_energy_eV": 0.1, "unbiased_potential_eV": 1.0},
        "thermostat": {"form": "nose-hoover-chain", "energy_eV": 0.2,
                       "extended_energy_eV": 1.6},
    }


def test_sampling_restart_records_are_recomputed_and_required(tmp_path):
    cfg = config(tmp_path)
    bind_sampling(cfg, request(tmp_path), tmp_path)
    bind_thermostat(cfg, thermostat())
    a = frame()
    validate_sampling_frame(a, cfg)
    validate_thermostat_frame(a, cfg)
    compare_sampling_restart(a, deepcopy(a), cfg)
    compare_thermostat_restart(a, deepcopy(a), cfg)
    for key in ("sampling", "thermostat"):
        b = deepcopy(a)
        b.pop(key)
        with pytest.raises(ValueError, match="record"):
            (validate_sampling_frame if key == "sampling" else validate_thermostat_frame)(b, cfg)
    b = deepcopy(a)
    b["sampling"]["cv_angstrom"] += 0.01
    with pytest.raises(ValueError, match="CV"):
        validate_sampling_frame(b, cfg)
    b = deepcopy(a)
    b["thermostat"]["energy_eV"] += 0.01
    b["thermostat"]["extended_energy_eV"] += 0.01
    with pytest.raises(ValueError, match="state energy"):
        compare_thermostat_restart(b, a, cfg)


def test_new_records_are_rejected_for_legacy_runs():
    with pytest.raises(ValueError, match="Unexpected"):
        validate_sampling_frame(frame(), {})
    with pytest.raises(ValueError, match="Unexpected"):
        validate_thermostat_frame(frame(), {})
