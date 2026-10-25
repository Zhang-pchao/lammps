# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Declared Hamiltonians cannot select another assembly or lose strict checks."""

import hashlib
import json
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import config as gap_config

from qmdpa import cli, dispatch
from qmdpa.config import load_config
from qmdpa.inference import RAW_PROFILE

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['gap_config']


def mechanical(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"test weights only")
    (tmp_path / "atoms.data").write_text("test data only")
    return {
        "schema_version": 1,
        "model": "model.pt",
        "model_sha256": hashlib.sha256(b"test weights only").hexdigest(),
        "data_file": "atoms.data",
        "type_symbols": ["H"],
        "periodic": False,
        "region": {"atom_ids": [1, 2], "symbols": ["H", "H"], "molecules": [[1, 2]]},
        "full_charge_spin": [0, 1],
        "region_charge_spin": [0, 1],
        "qm": {"xc": "pbe", "basis": "sto-3g"},
        "timestep_fs": 0.1,
        "steps": 2,
    }


def additive(tmp_path):
    value = mechanical(tmp_path)
    for key in ("region", "full_charge_spin", "region_charge_spin"):
        value.pop(key)
    value.update(
        hamiltonian="finite-electrostatic-additive",
        qm_region={"atom_ids": [1], "symbols": ["H"], "molecules": [[1]]},
        environment_region={"atom_ids": [2], "symbols": ["H"], "molecules": [[2]]},
        environment_charge_spin=[0, 2],
        embedding={"atom_ids": [2], "charges_e": [0.0], "source": "test fixed field"},
        cross={
            "form": "lj12-6",
            "source": "test explicit pair",
            "mixing": "none",
            "cutoff": None,
            "pairs": [
                {
                    "qm_symbol": "H",
                    "environment_symbol": "H",
                    "epsilon_ev": 0.0,
                    "sigma_angstrom": 1.0,
                }
            ],
        },
        inference_profile=RAW_PROFILE,
        qm={"basis": "sto-3g", "multiplicity": 2},
    )
    return value


def save(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path


def test_legacy_and_explicit_mechanical_are_identical(tmp_path):
    value = mechanical(tmp_path)
    path = save(tmp_path, value)
    implicit = dispatch.load_run_config(path)
    value["hamiltonian"] = dispatch.MECHANICAL
    explicit = dispatch.load_run_config(save(tmp_path, value))
    assert implicit == explicit == load_config(path)
    assert explicit["hamiltonian"] == dispatch.MECHANICAL


@pytest.mark.parametrize("kind", ["mechanical", "additive", "gap"])
def test_dispatch_reuses_existing_loader_and_correct_total_or_correction(
    tmp_path, gap_config, kind
):
    value = {"mechanical": mechanical, "additive": additive}.get(kind)
    value = value(tmp_path) if value else gap_config
    path = save(tmp_path, value)
    with patch("qmdpa.config.importlib.metadata.version", return_value="test"):
        result = dispatch.load_run_config(path)
        module, loader, _, _, _ = dispatch._route(result["hamiltonian"])
        assert result == dispatch._function(module, loader)(path)
    description = dispatch.describe_run(result)
    assert description["assembly"]["native_pair"] == ("deepmd" if kind == "mechanical" else "zero")
    assert description["physics_sha256"] == result["physics_sha256"]
    assert "reference" not in description["provider_contract"]["state"]


@pytest.mark.parametrize(
    "identifier",
    [None, True, {}, "unknown", "finite-reference-subtractive", "finite-two-partition-mechanical"],
)
def test_unsupported_model_never_falls_back(tmp_path, identifier):
    value = mechanical(tmp_path)
    value["hamiltonian"] = identifier
    with pytest.raises(ValueError, match="Unsupported"):
        dispatch.load_run_config(save(tmp_path, value))


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"hamiltonian":"fixed-region-mechanical","hamiltonian":"other"}',
        '{"steps":NaN}',
        '{"steps":Infinity}',
    ],
)
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, raw):
    path = tmp_path / "config.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        dispatch.load_run_config(path)


def test_ee_without_model_id_is_not_inferred_from_fields(tmp_path):
    value = additive(tmp_path)
    value.pop("hamiltonian")
    with pytest.raises(ValueError, match="Unknown"):
        dispatch.load_run_config(save(tmp_path, value))


def test_explicit_ee_cannot_enter_mechanical_api(tmp_path):
    value = mechanical(tmp_path)
    value["hamiltonian"] = "finite-electrostatic-gap-anchor"
    with pytest.raises(ValueError, match="Mechanical"):
        load_config(save(tmp_path, value))


def test_config_mutation_between_route_selection_and_loader_is_rejected(tmp_path):
    path = save(tmp_path, mechanical(tmp_path))

    def changed(_):
        path.write_text("{}")
        return {"hamiltonian": dispatch.MECHANICAL}

    with patch.object(dispatch, "_function", return_value=changed):
        with pytest.raises(ValueError, match="changed"):
            dispatch.load_run_config(path)


@pytest.mark.parametrize(
    "identifier,module,setup,factory",
    [
        (dispatch.MECHANICAL, "cli", "_mechanical_pair", "_mechanical_coupling"),
        ("finite-electrostatic-additive", "additive_cli", "_zero_pair", "_additive_coupling"),
        ("finite-electrostatic-gap-anchor", "gap_anchor_cli", "_zero_pair", "_gap_anchor_coupling"),
    ],
)
def test_assembly_delegates_only_to_the_declared_driver(identifier, module, setup, factory):
    config = {"hamiltonian": identifier}
    lmp, comm = object(), object()
    with (
        patch("qmdpa." + module + "." + setup) as pair,
        patch("qmdpa." + module + "." + factory) as coupling,
    ):
        dispatch._setup_pair(lmp, config)
        dispatch._make_coupling(lmp, comm, config)
        pair.assert_called_once_with(lmp, config)
        coupling.assert_called_once_with(lmp, comm, config)


def test_run_keeps_existing_collective_loader_and_checkpoint_lifecycle():
    with patch("qmdpa.cli._run") as run:
        dispatch.run("input.json", "output", restart="checkpoint.json")
    run.assert_called_once_with(
        "input.json",
        "output",
        restart="checkpoint.json",
        config_loader=dispatch.load_run_config,
        setup_pair=dispatch._setup_pair,
        make_coupling=dispatch._make_coupling,
    )


def test_cli_validate_runs_no_native_or_electronic_evaluator(tmp_path, capsys):
    path = save(tmp_path, mechanical(tmp_path))
    with patch("qmdpa.cli._run", side_effect=AssertionError("native execution")):
        assert cli.main(["validate", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "configuration-valid"
    assert result["assembly"]["external_quantity"] == "qm-minus-region-low-correction"


def test_cli_run_selects_configured_route_without_new_argument_grammar(tmp_path):
    path = tmp_path / "input.json"
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.json"
    with patch("qmdpa.dispatch.run") as run:
        assert (
            cli.main(["run", str(path), "--output", str(output), "--restart", str(checkpoint)]) == 0
        )
    run.assert_called_once_with(path, output, restart=checkpoint)


def test_invalid_validation_returns_error_and_no_success_summary(tmp_path, capsys):
    path = save(tmp_path, {"hamiltonian": "unknown"})
    assert cli.main(["validate", str(path)]) == 1
    streams = capsys.readouterr()
    assert streams.out == "" and "Unsupported" in streams.err
