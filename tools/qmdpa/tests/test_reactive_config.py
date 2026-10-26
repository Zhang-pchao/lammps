#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Reactive route identity, complete integer frames and native restart rejection."""

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_reactive_water import family, geometry
from test_water_buffer_config import document as legacy

from qmdpa.config import file_digest
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.reactive_config import frame_record, validate_reactive_frame
from qmdpa.reactive_water import HAMILTONIAN
from qmdpa.runtime import _attach_state, _compare_restart, _validate_restart


@pytest.fixture(autouse=True)
def numerical_versions(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "fixture")


def document():
    raw = legacy()
    for key in ("core_group_ids", "switch", "partitions"):
        raw.pop(key)
    fam = family()
    raw.update(hamiltonian=HAMILTONIAN, groups=[asdict(g) for g in fam.groups],
               water_guard={"min_oh_angstrom": .4, "max_oh_angstrom": 1.6,
                            "min_foreign_oh_angstrom": 1.6},
               reactive_region={"pool_group_ids": list(fam.pool), "seed_group_id": fam.seed_group_id,
                                "channels": [asdict(c) for c in fam.channels],
                                **{key: getattr(fam, key) for key in
                                   ("inner_radius_angstrom", "outer_radius_angstrom",
                                    "transfer_inner_angstrom", "transfer_outer_angstrom")}})
    return raw


def load(tmp_path, raw=None):
    (tmp_path/"model.pt").write_bytes(b"unevaluated model fixture")
    (tmp_path/"water.data").write_bytes(b"unevaluated data fixture")
    path = tmp_path/"run.json"
    path.write_text(json.dumps(document() if raw is None else raw))
    return load_run_config(path)


def frame(cfg, transfers=(1., 1.)):
    result = {"step": 0, "potential_eV": -20., "kinetic_eV": 0., "temperature_K": 0.,
              "atoms": [{"id": i+1, "x": list(x), "v": [0., 0., 0.],
                         "f": [0., 0., 0.], "image": [0, 0, 0]}
                        for i, x in enumerate(geometry(transfers).positions)]}
    result["reactive_water"] = frame_record(result, cfg)
    return result


def test_unified_selector_total_assembly_and_complete_ledger(tmp_path):
    cfg = load(tmp_path)
    view = describe_run(cfg)
    assert view["coupling"] == {"surface": "ground", "embedding": "mechanical", "me_model": "reactive-water"}
    assert view["assembly"] == {"native_pair": "zero", "external_quantity": "complete-total-energy-and-forces"}
    assert view["energy_contract"]["native_baseline"] == "zero"
    assert len(cfg["partitions"]) == len(family().partitions)
    raw = document()
    raw.pop("hamiltonian")
    raw["coupling"] = view["coupling"]
    assert load(tmp_path, raw)["physics_sha256"] == cfg["physics_sha256"]


def test_canonical_order_and_restart_budget_identity(tmp_path):
    cfg = load(tmp_path)
    raw = document()
    raw["groups"].reverse()
    raw["reactive_region"]["channels"].reverse()
    raw["reactive_region"]["pool_group_ids"].reverse()
    for key in ("atom_ids", "symbols", "molecules"):
        raw["full_region"][key].reverse()
    raw.update(steps=200, sample_interval=20)
    assert load(tmp_path, raw)["physics_sha256"] == cfg["physics_sha256"]
    raw["water_guard"]["min_oh_angstrom"] = .45
    assert load(tmp_path, raw)["physics_sha256"] != cfg["physics_sha256"]


@pytest.mark.parametrize("failure", ["reference", "source", "state", "periodic", "guard", "region", "partition",
                                     "channel", "duplicate", "pool", "groups", "molecules", "raw", "sampling"])
def test_invalid_or_unqualified_requests_fail_without_a_backend(tmp_path, failure):
    raw = document()
    if failure == "reference":
        raw["buffer_reference"]["model_sha256"] = "0"*64
    elif failure == "source":
        raw["buffer_reference"]["software"]["coupling_sha256"] = "0"*64
    elif failure == "state":
        raw["qm"]["charge"] = 1
    elif failure == "periodic":
        raw["periodic"] = True
    elif failure == "guard":
        raw["water_guard"]["min_oh_angstrom"] = True
    elif failure == "region":
        raw["region"] = raw["full_region"]
    elif failure == "partition":
        raw["partitions"] = []
    elif failure == "channel":
        raw["reactive_region"]["channels"][0]["acceptor_group_id"] = "w4"
    elif failure == "duplicate":
        raw["reactive_region"]["channels"][1] = raw["reactive_region"]["channels"][0]
    elif failure == "pool":
        raw["reactive_region"]["pool_group_ids"] = ["w1"]
    elif failure == "groups":
        raw["groups"][0]["symbols"] = ["H", "H", "O"]
    elif failure == "molecules":
        raw["full_region"]["molecules"] = [[1, 2, 4], [3, 5, 6], [7, 8, 9], [10, 11, 12]]
    elif failure == "raw":
        raw["inference_profile"] = "unqualified-float32"
    else:
        raw["sampling"] = {"engine": "plumed", "cv": {"form": "distance"},
                           "bias": {"form": "harmonic"}, "kernel": {}}
    with patch("qmdpa.backends.DeepMDBackend") as backend:
        with pytest.raises((ValueError, TypeError)):
            load(tmp_path, raw)
        backend.assert_not_called()


def test_accepted_frame_tracks_post_transfer_membership_and_electrons(tmp_path):
    cfg = load(tmp_path)
    value = frame(cfg)
    validate_reactive_frame(value, cfg)
    active = [p for p in value["reactive_water"]["partitions"] if p["weight"] > 0]
    assert len(active) == 1
    assert active[0]["qm_atom_ids"] == [1, 3, 5, 7, 8, 9]
    assert active[0]["qm_electrons"] == 20
    assert dict(active[0]["group_formal_charges"]) == {"w1": -1, "w2": 0, "w3": 1, "w4": 0}
    coupling = SimpleNamespace(failure=None, last_step=0)
    del value["reactive_water"]
    _attach_state(value, coupling, cfg)
    validate_reactive_frame(value, cfg)
    coupling.last_step = 1
    with pytest.raises(ValueError, match="successful current callback"):
        _attach_state(value, coupling, cfg)


@pytest.mark.parametrize("field", ["qm_electrons", "qm_atom_ids", "group_hydrogens", "group_formal_charges", "weight"])
def test_corrupted_integer_state_or_weights_reject_before_native_restart(tmp_path, field):
    cfg = load(tmp_path)
    value = frame(cfg)
    restart = tmp_path/"native.restart"
    restart.write_bytes(b"unevaluated native fixture")
    record = {"physics_sha256": cfg["physics_sha256"], "frame": value,
              "restart_file": restart.name, "restart_sha256": file_digest(restart)}
    checkpoint = tmp_path/"checkpoint.json"
    checkpoint.write_text(json.dumps(record))
    _validate_restart(checkpoint, cfg)
    _compare_restart(value, record, cfg)
    changed = deepcopy(record)
    part = next(p for p in changed["frame"]["reactive_water"]["partitions"] if p["weight"] > 0)
    part[field] = [] if isinstance(part[field], list) else part[field]+1
    checkpoint.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="grouping, integer states or weights"):
        _validate_restart(checkpoint, cfg)
    with pytest.raises(ValueError, match="grouping, integer states or weights"):
        _compare_restart(value, changed, cfg)


def test_level_shift_is_a_frozen_cpu_ground_solver_control(tmp_path):
    from qmdpa.config import validate_qm
    from qmdpa.gpu_backends import validate_gpu_settings

    cfg = load(tmp_path)
    raw = document()
    raw["qm"]["scf_level_shift_hartree"] = .3
    with pytest.raises(ValueError, match="QM/provider settings differ"):
        load(tmp_path, raw)
    raw["buffer_reference"]["qm"]["scf_level_shift_hartree"] = .3
    changed = load(tmp_path, raw)
    assert cfg["physics_sha256"] != changed["physics_sha256"]
    assert changed["qm"]["scf_level_shift_hartree"] == .3
    for value in (True, -.1, "0.3", float("inf")):
        with pytest.raises(ValueError, match="scf_level_shift"):
            validate_qm({"scf_level_shift_hartree": value})
    with pytest.raises(ValueError, match="CPU PySCF"):
        validate_gpu_settings({"xc": "hf", "scf_level_shift_hartree": .3})
