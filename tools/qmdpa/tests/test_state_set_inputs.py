# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Portable shared-state preparation and complete-or-failed output collections."""

import copy
import json
import shutil
from dataclasses import replace
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, reference
from test_state_inputs import request

from qmdpa import state_inputs
from qmdpa.cli import main
from qmdpa.dispatch import load_run_config
from qmdpa.excited import TDAReference
from qmdpa.state_inputs import prepare_state, prepare_states, resolve_provider_reference

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case']


def fixture(embedded=False, names=("state a", "state b")):
    a = replace(reference(embedded=embedded), label=names[0])
    b = replace(
        a,
        label=names[1],
        seed_root=2,
        excitation_energy_ev=3.0,
        transition_density=a.transition_density.T,
    )
    value = request(a)
    del value["state"]
    value["states"] = {names[0]: 1, names[1]: 2}
    return value, {names[0]: a, names[1]: b}


def paths(tmp_path, request_data):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request_data))
    return path, tmp_path / "prepared"


@pytest.mark.parametrize("embedded", [False, True])
def test_cli_calls_shared_builder_once_and_leaf_formats_match_single(tmp_path, embedded, capsys):
    data, refs = fixture(embedded)
    path, output = paths(tmp_path, data)
    with patch("qmdpa.multistate.build_tda_references", return_value=refs) as build:
        assert main(["prepare-states", str(path), "--output", str(output)]) == 0
    build.assert_called_once()
    assert build.call_args.kwargs["states"] == data["states"]
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads((output / "result.json").read_text())
    assert result["preparation"] == "shared-scf-tda"
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
    assert not (output / ".states-pending").exists()
    for name, entry in result["states"].items():
        leaf = output / entry["directory"]
        pinned = TDAReference.from_record(json.loads((leaf / "reference.json").read_text()))
        assert pinned.sha256 == refs[name].sha256 == entry["reference_sha256"]
        inline = json.loads((leaf / "provider-inline.json").read_text())
        assert resolve_provider_reference(entry["provider"], output / "config.json") == inline
        assert (
            resolve_provider_reference(
                json.loads((leaf / "provider-file.json").read_text()), leaf / "config.json"
            )
            == inline
        )
        single = copy.deepcopy(data)
        del single["states"]
        single["state"] = {"initial_root": refs[name].seed_root, "label": name}
        single_path = tmp_path / (leaf.name + ".json")
        single_path.write_text(json.dumps(single))
        target = tmp_path / (leaf.name + "-single")
        with patch("qmdpa.state_inputs._build_reference", return_value=refs[name]):
            prepare_state(single_path, target)
        for filename in (
            "reference.json",
            "provider-inline.json",
            "provider-file.json",
            "qm.json",
            "result.json",
        ):
            assert (leaf / filename).read_bytes() == (target / filename).read_bytes()


def test_state_names_are_labels_not_filesystem_paths_and_collection_is_portable(
    tmp_path, monkeypatch
):
    data, refs = fixture(names=("../outside", "state/with: punctuation"))
    path, output = paths(tmp_path, data)
    with patch("qmdpa.multistate.build_tda_references", return_value=refs):
        result = prepare_states(path, output)
    assert [x["directory"] for x in result["states"].values()] == [
        "states/state-0001",
        "states/state-0002",
    ]
    assert not (tmp_path / "outside").exists()
    copied = tmp_path / "relocated"
    shutil.copytree(output, copied)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for name, entry in result["states"].items():
        resolved = resolve_provider_reference(entry["provider"], copied / "config.json")
        assert resolved["state"]["reference"] == refs[name].to_record()


@pytest.mark.parametrize(
    "change",
    [
        "singular",
        "missing",
        "one",
        "duplicate",
        "root_bool",
        "root_range",
        "empty_name",
        "unknown",
        "solver",
        "method",
        "engine",
        "gaussian",
        "field",
    ],
)
def test_bad_request_fails_before_shared_builder(tmp_path, change):
    data, _ = fixture(embedded=True)
    if change == "singular":
        data["state"] = {"initial_root": 1, "label": "ambiguous"}
    elif change == "missing":
        del data["states"]
    elif change == "one":
        data["states"] = {"a": 1}
    elif change == "duplicate":
        data["states"] = {"a": 1, "b": 1}
    elif change == "root_bool":
        data["states"] = {"a": 1, "b": True}
    elif change == "root_range":
        data["states"] = {"a": 1, "b": 99}
    elif change == "empty_name":
        data["states"] = {"a": 1, " ": 2}
    elif change == "unknown":
        data["automatic_fallback"] = True
    elif change == "solver":
        data["qm"]["tda_solver"] = "iterative"
    elif change == "method":
        data["method"] = "pbe0-tda"
    elif change == "engine":
        data.update(engine="gpu4pyscf", device="cuda")
    elif change == "gaussian":
        data["charges"]["radii_angstrom"] = [1, 1, 1]
    else:
        del data["environment_geometry"]
    path, output = paths(tmp_path, data)
    with patch("qmdpa.multistate.build_tda_references") as build:
        with pytest.raises((ValueError, TypeError)):
            prepare_states(path, output)
    build.assert_not_called()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert (output / "failure.json").exists() and not (output / "result.json").exists()
    assert not (output / "states").exists()


@pytest.mark.parametrize("raw", ['{"states":{"a":1,"a":2}}', '{"states":{"a":NaN,"b":2}}'])
def test_duplicate_and_nonfinite_json_are_rejected_before_calculation(tmp_path, raw):
    path = tmp_path / "request.json"
    path.write_text(raw)
    with patch("qmdpa.multistate.build_tda_references") as build:
        with pytest.raises(ValueError):
            prepare_states(path, tmp_path / "out")
    build.assert_not_called()


@pytest.mark.parametrize("change", ["missing", "root", "label", "settings", "geometry", "field"])
def test_wrong_builder_output_cannot_publish_a_collection(tmp_path, change):
    data, refs = fixture(embedded=True)
    bad = dict(refs)
    b = refs["state b"]
    if change == "missing":
        del bad["state b"]
    elif change == "root":
        bad["state b"] = replace(b, seed_root=1)
    elif change == "label":
        bad["state b"] = replace(b, label="wrong label")
    elif change == "settings":
        bad["state b"] = replace(b, settings=replace(b.settings, threads=3))
    elif change == "geometry":
        bad["state b"] = replace(
            b, qm_geometry=replace(b.qm_geometry, positions=b.qm_geometry.positions + 0.01)
        )
    else:
        bad["state b"] = replace(b, charges=replace(b.charges, source="Different field"))
    path, output = paths(tmp_path, data)
    with patch("qmdpa.multistate.build_tda_references", return_value=bad):
        with pytest.raises(ValueError, match="Prepared reference"):
            prepare_states(path, output)
    assert not (output / "states").exists() and not (output / "result.json").exists()


def test_solver_failure_is_recorded_without_completed_artifacts(tmp_path):
    data, _ = fixture()
    path, output = paths(tmp_path, data)
    with patch("qmdpa.multistate.build_tda_references", side_effect=RuntimeError("SCF failure")):
        with pytest.raises(RuntimeError, match="SCF failure"):
            prepare_states(path, output)
    assert not (output / "states").exists() and not (output / "result.json").exists()
    assert json.loads((output / "failure.json").read_text())["message"] == "SCF failure"


@pytest.mark.parametrize("stage", ["calculation", "files", "software"])
def test_input_or_source_change_cannot_publish_completed_states(tmp_path, stage):
    data, refs = fixture()
    path, output = paths(tmp_path, data)

    def build(*args, **kwargs):
        if stage == "calculation":
            path.write_text("{}")
        return refs

    writer = state_inputs._write_reference_files

    def write(directory, files):
        writer(directory, files)
        if stage == "files":
            path.write_text("{}")

    identities = [{"test": 1}, {"test": 2}] if stage == "software" else [{"test": 1}] * 3
    with (
        patch("qmdpa.multistate.build_tda_references", side_effect=build),
        patch("qmdpa.state_inputs._write_reference_files", side_effect=write),
        patch("qmdpa.config.software_identity", side_effect=identities),
    ):
        with pytest.raises(ValueError, match="changed"):
            prepare_states(path, output)
    assert not (output / "states").exists() and not (output / "result.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert (output / "request.json").read_text() == json.dumps(data)


def test_interrupted_file_writing_retains_staging_without_completed_collection(tmp_path):
    data, refs = fixture()
    path, output = paths(tmp_path, data)
    writer = state_inputs._write_reference_files

    def fail_second(directory, files):
        if directory.name == "state-0002":
            raise OSError("Injected write failure")
        writer(directory, files)

    with (
        patch("qmdpa.multistate.build_tda_references", return_value=refs),
        patch("qmdpa.state_inputs._write_reference_files", side_effect=fail_second),
    ):
        with pytest.raises(OSError):
            prepare_states(path, output)
    assert (output / ".states-pending/state-0001/reference.json").exists()
    assert not (output / "states").exists() and not (output / "result.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_existing_output_is_never_overwritten(tmp_path):
    data, _ = fixture()
    path, output = paths(tmp_path, data)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing")
    with patch("qmdpa.multistate.build_tda_references") as build:
        with pytest.raises(FileExistsError):
            prepare_states(path, output)
    build.assert_not_called()
    assert sentinel.read_text() == "existing"


def test_collection_provider_keeps_existing_me_and_additive_configuration(
    tmp_path, case, monkeypatch
):
    embedded, _, config = case
    data, refs = fixture(embedded)
    path, output = paths(tmp_path, data)
    with patch("qmdpa.multistate.build_tda_references", return_value=refs):
        result = prepare_states(path, output)
    monkeypatch.setattr("qmdpa.config.importlib.metadata.version", lambda name: "test")
    config["model"] = str(tmp_path / "model.pt")
    config["data_file"] = str(tmp_path / "atoms.data")
    for name, entry in result["states"].items():
        target = output / "config.json"
        config["provider"] = entry["provider"]
        target.write_text(json.dumps(config))
        from_file = load_run_config(target)
        config["provider"] = json.loads(
            (output / entry["directory"] / "provider-inline.json").read_text()
        )
        target.write_text(json.dumps(config))
        from_inline = load_run_config(target)
        assert from_file == from_inline
