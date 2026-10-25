# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Reference preparation, file integrity and unchanged native model identities."""

import copy
import hashlib
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, provider_request, reference
from test_gap_anchor_native import config as anchor_case
from test_pbe0_tda import pbe0_reference

from qmdpa.cli import main
from qmdpa.dispatch import load_run_config
from qmdpa.state_inputs import prepare_state, resolve_provider_reference, validate_preparation

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case', 'anchor_case']


def request(ref, method="rhf-tda"):
    record = ref.to_record()
    value = {
        "schema_version": 1,
        "method": method,
        "qm": asdict(ref.settings),
        "qm_geometry": record["qm_geometry"],
        "state": {"initial_root": 1, "label": "Small input"},
    }
    if ref.charges is not None:
        value.update(environment_geometry=record["environment_geometry"], charges=record["charges"])
    return value


def file_provider(tmp_path, provider, name="state.json"):
    path = tmp_path / name
    path.write_text(json.dumps(provider["state"]["reference"]))
    result = copy.deepcopy(provider)
    result["state"] = {
        "method": provider["state"]["method"],
        "reference_file": {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    return result


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("method", ["rhf-tda", "pbe0-tda"])
def test_preparation_roundtrip_and_cli_use_existing_builders(tmp_path, embedded, method, capsys):
    ref = reference(embedded=embedded) if method == "rhf-tda" else pbe0_reference(embedded)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request(ref, method)))
    output = tmp_path / "prepared"
    with patch("qmdpa.state_inputs._build_reference", return_value=ref) as build:
        assert main(["prepare-state", str(path), "--output", str(output)]) == 0
    build.assert_called_once()
    result = json.loads(capsys.readouterr().out)
    assert result["reference_sha256"] == ref.to_record()["sha256"]
    inline = json.loads((output / "provider-inline.json").read_text())
    declared = json.loads((output / "provider-file.json").read_text())
    assert resolve_provider_reference(declared, output / "future-config.json") == inline
    assert json.loads((output / "qm.json").read_text()) == asdict(ref.settings)
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        prepare_state(path, output)


@pytest.mark.parametrize(
    "change", ["root", "label", "solver", "unknown", "schema", "charges", "gaussian"]
)
def test_invalid_request_rejected_before_builder(tmp_path, change):
    data = request(reference(embedded=True))
    if change == "root":
        data["state"]["initial_root"] = True
    elif change == "label":
        data["state"]["label"] = " "
    elif change == "solver":
        del data["qm"]["tda_solver"]
    elif change == "unknown":
        data["unused"] = 1
    elif change == "schema":
        data["schema_version"] = True
    elif change == "charges":
        del data["charges"]
    else:
        data["charges"]["radii_angstrom"] = [0.5, 0.5, 0.5]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data))
    output = tmp_path / "out"
    with patch("qmdpa.state_inputs._build_reference") as build:
        with pytest.raises((ValueError, TypeError)):
            prepare_state(path, output)
    build.assert_not_called()
    assert (output / "failure.json").is_file() and not (output / "result.json").exists()


def test_solver_failure_or_input_mutation_cannot_publish_reference(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request(reference())))
    with patch("qmdpa.state_inputs._build_reference", side_effect=RuntimeError("SCF failed")):
        with pytest.raises(RuntimeError, match="SCF"):
            prepare_state(path, tmp_path / "failed")
    assert not (tmp_path / "failed/reference.json").exists()

    def mutate(*args):
        path.write_text("{}")
        return reference()

    with patch("qmdpa.state_inputs._build_reference", side_effect=mutate):
        with pytest.raises(ValueError, match="changed"):
            prepare_state(path, tmp_path / "mutated")
    assert not (tmp_path / "mutated/reference.json").exists()


def test_file_reference_rejects_tampering_ambiguity_and_duplicate_json(tmp_path):
    inline = provider_request(reference())
    spec = file_provider(tmp_path, inline)
    original = copy.deepcopy(spec)
    assert resolve_provider_reference(spec, tmp_path / "config.json") == inline
    assert spec == original
    bad = copy.deepcopy(spec)
    bad["state"]["reference"] = inline["state"]["reference"]
    with pytest.raises(ValueError, match="only"):
        resolve_provider_reference(bad, tmp_path / "c.json")
    (tmp_path / "state.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        resolve_provider_reference(spec, tmp_path / "c.json")
    raw = '{"method": "rhf-tda", "method": "rhf-tda"}'
    (tmp_path / "state.json").write_text(raw)
    spec["state"]["reference_file"]["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_provider_reference(spec, tmp_path / "c.json")


@pytest.mark.parametrize(
    "source",
    [
        None,
        {},
        {"path": "state.json"},
        {"path": "", "sha256": "0" * 64},
        {"path": "x", "sha256": "BAD"},
    ],
)
def test_malformed_file_sources_rejected(tmp_path, source):
    spec = {"state": {"method": "rhf-tda", "reference_file": source}}
    with pytest.raises(ValueError):
        resolve_provider_reference(spec, tmp_path / "c.json")


def test_me_and_additive_file_inputs_preserve_exact_physics(tmp_path, case, monkeypatch):
    _, ref, config = case
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr("qmdpa.config.importlib.metadata.version", lambda x: "test")
    inline = load_run_config(path)
    config["provider"] = file_provider(tmp_path, config["provider"])
    path.write_text(json.dumps(config))
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert load_run_config(path) == inline


def test_gap_anchor_file_input_preserves_exact_physics(tmp_path, anchor_case, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(anchor_case))
    monkeypatch.setattr("qmdpa.config.importlib.metadata.version", lambda x: "test")
    inline = load_run_config(path)
    anchor_case["provider"] = file_provider(tmp_path, anchor_case["provider"])
    path.write_text(json.dumps(anchor_case))
    assert load_run_config(path) == inline


def test_pbe0_preparation_requires_declared_grid_and_rejects_df():
    value = request(pbe0_reference(), "pbe0-tda")
    value["qm"].pop("grid_response")
    with pytest.raises(ValueError, match="grid"):
        validate_preparation(value)
    value = request(pbe0_reference(), "pbe0-tda")
    value["qm"]["density_fit"] = {"auxbasis": "weigend"}
    with pytest.raises(TypeError):
        validate_preparation(value)
