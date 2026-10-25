# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU reference collections preserve engine, portable files and failure identity."""

import copy
import json
import shutil
from dataclasses import replace
from unittest.mock import patch

import pytest
from test_excited_config import reference as cpu_reference
from test_gpu_selected_ee_config import reference
from test_state_inputs import request

from qmdpa.cli import main
from qmdpa.gpu_excited import GPUTDAReference
from qmdpa.state_inputs import prepare_state, prepare_states, resolve_provider_reference


def fixture(embedded=False):
    first = replace(reference(embedded), label="first")
    refs = {"first": first, "second": replace(first, label="second", seed_root=2,
            excitation_energy_ev=3.0, transition_density=first.transition_density.T)}
    data = request(first)
    del data["state"]
    data.update(engine="gpu4pyscf", device="cuda", states={"first": 1, "second": 2})
    return data, refs


@pytest.mark.parametrize("embedded", [False, True])
def test_gpu_command_uses_one_builder_and_portable_single_state_files(tmp_path, embedded, capsys):
    data, refs = fixture(embedded)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data))
    target = tmp_path / "set"
    with patch("qmdpa.multistate.build_gpu_tda_references", return_value=refs) as gpu, \
            patch("qmdpa.multistate.build_tda_references") as cpu, \
            patch("qmdpa.multistate.build_pbe0_tda_references") as pbe0:
        assert main(["prepare-states", str(path), "--output", str(target)]) == 0
    gpu.assert_called_once()
    cpu.assert_not_called()
    pbe0.assert_not_called()
    assert gpu.call_args.kwargs["states"] == data["states"]
    assert (gpu.call_args.kwargs["environment_geometry"] is not None) == embedded
    result = json.loads(capsys.readouterr().out)
    assert (result["engine"], result["device"]) == ("gpu4pyscf", "cuda")
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert "gpu4pyscf-cuda12x" in manifest["software"]["packages"]
    moved = tmp_path / "moved"
    shutil.copytree(target, moved)
    for name, entry in result["states"].items():
        leaf = moved / entry["directory"]
        inline = json.loads((leaf / "provider-inline.json").read_text())
        assert resolve_provider_reference(entry["provider"], moved / "config.json") == inline
        assert resolve_provider_reference(json.loads((leaf / "provider-file.json").read_text()),
                                          leaf / "config.json") == inline
        assert GPUTDAReference.from_record(inline["state"]["reference"]).sha256 == refs[name].sha256
        single = copy.deepcopy(data)
        del single["states"]
        single["state"] = {"initial_root": entry["initial_root"], "label": name}
        single_path = tmp_path / (name + ".json")
        single_path.write_text(json.dumps(single))
        output = tmp_path / (name + "-single")
        with patch("qmdpa.state_inputs._build_reference", return_value=refs[name]):
            prepare_state(single_path, output)
        for filename in ("reference.json", "provider-inline.json", "provider-file.json", "qm.json", "result.json"):
            assert (leaf / filename).read_bytes() == (output / filename).read_bytes()


@pytest.mark.parametrize("change", ["pbe0", "device", "settings_device", "iterative", "roots", "gaussian"])
def test_unsupported_gpu_collection_fails_before_building(tmp_path, change):
    data, _ = fixture(embedded=True)
    if change == "pbe0":
        data["method"] = "pbe0-tda"
    elif change == "device":
        data["device"] = "cpu"
    elif change == "settings_device":
        data["qm"]["device"] = "cpu"
    elif change == "iterative":
        data["qm"]["tda_solver"] = "iterative"
    elif change == "roots":
        data["states"]["second"] = 1
    else:
        data["charges"]["radii_angstrom"] = [1.0] * len(data["charges"]["atom_ids"])
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data))
    target = tmp_path / "out"
    with patch("qmdpa.multistate.build_gpu_tda_references") as build:
        with pytest.raises((TypeError, ValueError)):
            prepare_states(path, target)
    build.assert_not_called()
    assert json.loads((target / "manifest.json").read_text())["status"] == "failed"
    assert not (target / "result.json").exists() and not (target / "states").exists()


@pytest.mark.parametrize("change", ["foreign_reference", "failed_solve", "runtime_changed"])
def test_gpu_collection_cannot_accept_partial_or_foreign_results(tmp_path, change):
    data, refs = fixture()
    if change == "foreign_reference":
        refs["second"] = cpu_reference()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data))
    target = tmp_path / "out"
    effect = RuntimeError("SCF failure") if change == "failed_solve" else None
    identities = [{"cpu": 1}, {"gpu": 1}, {"gpu": 2}] if change == "runtime_changed" else [{"stable": 1}] * 4
    with patch("qmdpa.multistate.build_gpu_tda_references", return_value=refs, side_effect=effect), \
            patch("qmdpa.config.software_identity", side_effect=identities):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            prepare_states(path, target)
    assert json.loads((target / "manifest.json").read_text())["status"] == "failed"
    assert not (target / "states").exists() and not (target / "result.json").exists()
