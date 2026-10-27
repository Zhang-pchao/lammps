#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Selected-family route and restart admission without scientific imports."""

import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_molecular_adaptive import geometry, groups, switch
from test_selected_adaptive import controlled_provider, family, mixer  # noqa: F401 - Shared fixture.
from test_state_records import diagnostics

from qmdpa.config import file_digest
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.excited import _digest
from qmdpa.selected_adaptive import HAMILTONIAN, STATE_KIND
from qmdpa.selected_adaptive_runtime import attach_frame, compare_restart, validate_frame
from qmdpa.state_family import StateFamilyReference, prepare_family
from qmdpa.state_records import configured_reference


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "fixture")
    original = family()
    settings = replace(original.core.settings, tda_solver="dense")
    fam = StateFamilyReference(replace(original.core, settings=settings),
        tuple((name, replace(ref, settings=settings)) for name, ref in original.partitions))
    model, data, ref = (tmp_path/name for name in ("model.pt", "atoms.data", "family.json"))
    model.write_bytes(b"Model identity fixture, not evaluated")
    data.write_bytes(b"Native atom identity fixture, not parsed")
    ref.write_text(json.dumps(fam.to_record()))
    atoms = geometry()
    cfg = {
        "schema_version": 1, "hamiltonian": HAMILTONIAN,
        "provider": {"engine": "pyscf", "device": "cpu", "state": {"method": STATE_KIND}},
        "periodic": False, "model": str(model), "model_sha256": file_digest(model), "data_file": str(data),
        "type_symbols": ["H"], "full_region": {"atom_ids": list(atoms.atom_ids),
        "symbols": list(atoms.symbols), "molecules": [list(g.atom_ids) for g in groups()]},
        "groups": [asdict(g) for g in groups()], "core_group_ids": ["core"],
        "partitions": [{"partition_id": name, "qm_group_ids": ["core", name], "qm_charge": 0,
                        "qm_multiplicity": 1, "method_identity": "cpu-rhf-tda:"+fam.sha256,
                        "energy_reference": fam.sha256, "electronic_state": STATE_KIND}
                       for name, _ in fam.partitions],
        "switch": {"method": "quintic-molecular-occupancy", **asdict(switch())},
        "qm": asdict(settings), "full_charge_spin": [0, 1], "region_charge_spin": [0, 1],
        "family_reference": {"path": str(ref), "sha256": file_digest(ref)},
        "inference_profile": "torch-raw-deterministic-v1", "timestep_fs": .01, "steps": 2}
    return tmp_path/"run.json", cfg


def load(fixture, edit=None):
    path, raw = fixture
    cfg = copy.deepcopy(raw)
    if edit:
        edit(cfg)
    path.write_text(json.dumps(cfg))
    return load_run_config(path)


def test_unified_route_keeps_separate_identity_and_total(configuration):
    config = load(configuration)
    description = describe_run(config)
    assert description["coupling"] == {
        "surface": "selected", "embedding": "mechanical", "me_model": "molecular-common-core"}
    assert description["assembly"]["native_pair"] == "zero"
    assert description["state_family"]["electronic_propagation"] is False
    assert configured_reference(config) is None
    selected = load(configuration, lambda c: c.update(coupling=description["coupling"]))
    assert selected["physics_sha256"] == config["physics_sha256"]
    continued = load(configuration, lambda c: c.update(steps=100, sample_interval=5))
    assert continued["physics_sha256"] == config["physics_sha256"]
    changed = load(configuration, lambda c: c["switch"].update(outer_radius_angstrom=3.1))
    assert changed["physics_sha256"] != config["physics_sha256"]


@pytest.mark.parametrize("edit", [
    lambda c: c.update(periodic=True),
    lambda c: c["provider"].update(device="cuda"),
    lambda c: c["provider"]["state"].update(method="rhf-tda"),
    lambda c: c.update(full_charge_spin=[1, 2]),
    lambda c: c.update(region_charge_spin=[1, 2]),
    lambda c: c.update(inference_profile="uncontrolled"),
    lambda c: c["full_region"].update(molecules=[[1, 3], [2, 4], [5, 6], [7, 8]]),
    lambda c: c["full_region"]["symbols"].__setitem__(0, "He"),
    lambda c: c["qm"].update(conv_tol=1e-5),
    lambda c: c["qm"].update(tda_solver="iterative"),
    lambda c: c["family_reference"].update(sha256="0"*64),
    lambda c: c["family_reference"].update(extra="unknown"),
    lambda c: c["partitions"].pop(),
    lambda c: c["partitions"][0].update(energy_offset_ev=.01, offset_source="unqualified shift"),
    lambda c: c["partitions"][0].update(electronic_state="ground"),
    lambda c: c["partitions"][0].update(method_identity="untracked"),
    lambda c: c["partitions"][0].update(energy_reference="different"),
    lambda c: c.update(bias={}),
    lambda c: c.update(sampling={}),
    lambda c: c.update(thermostat={}),
    lambda c: c.update(execution={}),
])
def test_unadmitted_combinations_and_identity_changes_fail(configuration, edit):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        load(configuration, edit)


def fixture_frame(config):
    fam = StateFamilyReference.from_record(config["family_reference"])
    obj, atoms = mixer(fam), geometry()
    value = obj.evaluate(atoms)
    record = obj.last_state_record
    for name, previous in record["states"].items():
        record["states"][name] = {
            **diagnostics(fam.reference(name)), "family_sha256": fam.sha256, "partition_id": name,
            "common_core_reference_sha256": fam.core.sha256, "common_core": previous["common_core"]}
    obj._last_record = record
    frame = {"step": 3, "potential_eV": value.energy, "atoms": [
        {"id": atom, "x": xyz.tolist(), "f": force.tolist(), "image": [0, 0, 0]}
        for atom, xyz, force in zip(atoms.atom_ids, atoms.positions, value.forces)]}
    coupling = SimpleNamespace(failure=None, last_step=3, comm=SimpleNamespace(rank=0),
                               selected_family_observer=obj)
    attach_frame(frame, coupling, config)
    return frame, coupling


@pytest.mark.usefixtures("controlled_provider")
def test_frame_attachment_binds_state_and_is_detached(configuration):
    config = load(configuration)
    frame, coupling = fixture_frame(config)
    validate_frame(frame, config)
    validate_frame(json.loads(json.dumps(frame)), config)
    compare_restart(frame, json.loads(json.dumps(frame)), config)
    assert len(frame["selected_family"]["pair_correspondence"]) == 6
    coupling.selected_family_observer._last_record["states"].clear()
    assert len(frame["selected_family"]["states"]) == 3
    coupling.failure = "failed response"
    with pytest.raises(ValueError, match="successful"):
        attach_frame(frame, coupling, config)


@pytest.mark.parametrize("field", ["hash", "step", "family", "geometry", "weights", "active",
                                   "missing_pair", "duplicate_pair", "wrong_pair", "core_root",
                                   "own_root", "response", "forces", "energy", "switch_shape",
                                   "ownership", "images", "missing_state"])
@pytest.mark.usefixtures("controlled_provider")
def test_checkpoint_rejects_corrupt_or_incomplete_evidence(configuration, field):
    config = load(configuration)
    frame, _ = fixture_frame(config)
    r = frame["selected_family"]
    if field == "hash":
        r["sha256"] = "f"*64
    elif field == "step":
        r["step"] += 1
    elif field == "family":
        r["family_sha256"] = "e"*64
    elif field == "geometry":
        r["positions_angstrom"][0][0] += .01
    elif field == "weights":
        r["weights"][0] += .01
    elif field == "active":
        r["active_partition_ids"].pop()
    elif field == "missing_pair":
        r["pair_correspondence"].pop()
    elif field == "duplicate_pair":
        r["pair_correspondence"][1] = copy.deepcopy(r["pair_correspondence"][0])
    elif field == "wrong_pair":
        r["pair_correspondence"][0]["selected_root"] = 2
    elif field == "core_root":
        r["states"]["a"]["common_core"]["selected_root"] = 2
    elif field == "own_root":
        r["states"]["a"]["selected_root"] = 2
    elif field == "response":
        r["states"]["a"]["cphf_residual_after_refinement"] = .1
    elif field == "forces":
        r["forces_ev_angstrom"][0][0] += .01
    elif field == "energy":
        r["potential_ev"] += .01
    elif field == "switch_shape":
        r["switching_forces_ev_angstrom"].pop()
    elif field == "ownership":
        frame["atoms"][0]["id"] = frame["atoms"][1]["id"]
    elif field == "images":
        frame["atoms"][0]["image"] = [1, 0, 0]
    else:
        r["states"].pop("a")
    if field != "hash":
        r["sha256"] = _digest({k: v for k, v in r.items() if k != "sha256"})
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        validate_frame(frame, config)


@pytest.mark.usefixtures("controlled_provider")
def test_foreign_hamiltonian_cannot_carry_selected_family(configuration):
    frame, _ = fixture_frame(load(configuration))
    with pytest.raises(ValueError, match="Unexpected"):
        validate_frame(frame, {"hamiltonian": "finite-molecular-mechanical"})


def test_prepare_command_preserves_existing_reference(tmp_path, monkeypatch):
    from qmdpa import cli, state_family
    fam = family()
    atoms = geometry()
    request = {"schema_version": 1, "geometry": {"atom_ids": atoms.atom_ids, "symbols": atoms.symbols,
               "positions_angstrom": atoms.positions.tolist()}, "core_atom_ids": [1, 2],
               "partitions": [{"partition_id": name, "atom_ids": ref.qm_geometry.atom_ids}
                              for name, ref in fam.partitions],
               "settings": asdict(fam.core.settings), "initial_root": 1, "label": "fixture"}
    path, output = tmp_path/"request.json", tmp_path/"prepared.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(state_family, "build_state_family_reference", lambda *a, **kw: fam)
    assert cli.main(["prepare-state-family", str(path), "--output", str(output)]) == 0
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_family(path, output)
    assert output.read_bytes() == saved
