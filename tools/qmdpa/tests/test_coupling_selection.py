# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Equivalent selectors preserve identities; incompatible physics never falls back."""

import copy
import json
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_additive_config import case as charge_case
from test_dispatch import additive, mechanical, save
from test_excited_config import case
from test_gap_anchor_native import config as gap_config
from test_pbe0_gap_native import converted as pbe0_gap
from test_pbe0_provider_state import convert_case

from qmdpa import dispatch

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['charge_case', 'case', 'gap_config']


def compare(tmp_path, value, choice):
    old = dispatch.load_run_config(save(tmp_path, value))
    selected = copy.deepcopy(value)
    selected.pop("hamiltonian", None)
    selected["coupling"] = choice
    path = save(tmp_path, selected)
    original = path.read_bytes()
    new = dispatch.load_run_config(path)
    assert path.read_bytes() == original
    assert new == old
    assert "coupling" not in new
    assert dispatch.describe_run(new)["coupling"] == choice
    # An explicit, agreeing model identifier is equally valid.
    selected["hamiltonian"] = old["hamiltonian"]
    assert dispatch.load_run_config(save(tmp_path, selected)) == old
    return new


@pytest.mark.parametrize("embedding", ["mechanical", "electrostatic"])
def test_ground_selector_exact_identity_and_original_relative_paths(
    tmp_path, embedding, monkeypatch
):
    nested = tmp_path / "inputs"
    nested.mkdir()
    value = (mechanical if embedding == "mechanical" else additive)(nested)
    choice = {"surface": "ground", "embedding": embedding}
    if embedding == "electrostatic":
        choice["ee_model"] = "additive"
    monkeypatch.chdir(tmp_path)
    result = compare(nested, value, choice)
    assert result["model"] == str(nested / "model.pt")
    assert result["data_file"] == str(nested / "atoms.data")


@pytest.mark.parametrize("pbe0", [False, True])
def test_selected_me_and_additive_preserve_full_state_and_identity(tmp_path, case, pbe0):
    embedded, _, value = convert_case(case) if pbe0 else case
    choice = {"surface": "selected", "embedding": "electrostatic" if embedded else "mechanical"}
    if embedded:
        choice["ee_model"] = "additive"
    result = compare(tmp_path, value, choice)
    terms = dispatch.describe_run(result)["energy_contract"]["total_terms"]
    assert terms[-1]["name"] == ("cross_lj" if embedded else "isolated_selected")
    assert any(t["name"] == "embedded_selected" for t in terms) == embedded


@pytest.mark.parametrize("pbe0", [False, True])
def test_gap_anchor_remains_total_with_me_ground_anchor(tmp_path, gap_config, pbe0):
    if pbe0:
        pbe0_gap(gap_config)
    result = compare(
        tmp_path,
        gap_config,
        {"surface": "selected", "embedding": "electrostatic", "ee_model": "gap-anchor"},
    )
    description = dispatch.describe_run(result)
    assert description["assembly"]["native_pair"] == "zero"
    terms = description["energy_contract"]["total_terms"]
    assert [(x["name"], x["coefficient"]) for x in terms] == [
        ("low_full", 1),
        ("low_qm", -1),
        ("isolated_ground", 1),
        ("embedded_gap", 1),
    ]
    assert terms[-1]["force_coordinates"] == "AB"


@pytest.mark.parametrize(
    "choice",
    [
        None,
        [],
        {},
        {"surface": "ground"},
        {"surface": "adiabatic", "embedding": "mechanical"},
        {"surface": "ground", "embedding": "unknown"},
        {"surface": "ground", "embedding": "mechanical", "ee_model": None},
        {"surface": "ground", "embedding": "mechanical", "lambda": 0.5},
        {"surface": "ground", "embedding": "electrostatic"},
        {"surface": "ground", "embedding": "electrostatic", "ee_model": "gap-anchor"},
        {"surface": "selected", "embedding": "electrostatic", "ee_model": "reference-subtractive"},
        {"surface": "ground", "embedding": "electrostatic", "ee_model": "LJ+D3"},
        {"surface": "selected", "embedding": "mechanical"},
    ],
)
def test_invalid_selection_rejects_before_loading_or_allocating(tmp_path, choice):
    value = mechanical(tmp_path)
    value["coupling"] = choice
    with patch.object(dispatch, "_function", side_effect=AssertionError("must not load")):
        with pytest.raises(ValueError):
            dispatch.load_run_config(save(tmp_path, value))


@pytest.mark.parametrize("conflict", ["hamiltonian", "provider"])
def test_explicit_model_and_state_conflicts_reject(tmp_path, case, conflict):
    embedded, _, value = case
    value["coupling"] = {
        "surface": "selected",
        "embedding": "electrostatic" if embedded else "mechanical",
    }
    if embedded:
        value["coupling"]["ee_model"] = "additive"
    if conflict == "hamiltonian":
        value["hamiltonian"] = "finite-electrostatic-gap-anchor"
    else:
        value["coupling"]["surface"] = "ground"
    with pytest.raises(ValueError, match="conflict"):
        dispatch.load_run_config(save(tmp_path, value))


def test_invalid_state_reference_cannot_be_hidden_by_selector(tmp_path, case):
    embedded, _, value = case
    value["coupling"] = {
        "surface": "selected",
        "embedding": "electrostatic" if embedded else "mechanical",
    }
    if embedded:
        value["coupling"]["ee_model"] = "additive"
    value["provider"]["state"]["reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        dispatch.load_run_config(save(tmp_path, value))


def test_selector_validation_keeps_file_mutation_guard(tmp_path):
    value = mechanical(tmp_path)
    value["coupling"] = {"surface": "ground", "embedding": "mechanical"}
    path = save(tmp_path, value)

    def changed(raw, location):
        assert location == path
        path.write_text("{}")
        return {"hamiltonian": dispatch.MECHANICAL}

    with patch.object(dispatch, "_function", return_value=changed):
        with pytest.raises(ValueError, match="changed"):
            dispatch.load_run_config(path)


def test_normalization_does_not_mutate_request():
    raw = {"coupling": {"surface": "ground", "embedding": "mechanical"}, "nested": [1]}
    before = copy.deepcopy(raw)
    result = dispatch._normalize_selection(raw)
    result["nested"].append(2)
    assert raw == before


def test_schema2_file_charge_identity_and_relative_base(tmp_path, charge_case, monkeypatch):
    import hashlib

    nested = tmp_path / "charge-inputs"
    nested.mkdir()
    raw = json.dumps(
        {
            "schema_version": 1,
            "units": "e",
            "source": "Test table",
            "charges": [
                {"atom_id": i, "charge_e": q} for i, q in zip([4, 5, 6], [-0.834, 0.417, 0.417])
            ],
        }
    ).encode()
    table = nested / "charges.json"
    table.write_bytes(raw)
    charge_case["embedding"] = {
        "charge_request": {
            "kind": "file",
            "path": "charge-inputs/charges.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source": "Test file selection",
            "units": "e",
            "expected_total_charge_e": 0,
        }
    }
    monkeypatch.chdir(nested)
    resolved = compare(
        tmp_path,
        charge_case,
        {"surface": "ground", "embedding": "electrostatic", "ee_model": "additive"},
    )
    assert resolved["embedding"]["charges_e"] == [-0.834, 0.417, 0.417]
    table.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SHA256|checksum|hash"):
        dispatch.load_run_config(tmp_path / "config.json")


def test_source_change_still_invalidates_selector_restart_identity(tmp_path):
    value = mechanical(tmp_path)
    value["coupling"] = {"surface": "ground", "embedding": "mechanical"}
    path = save(tmp_path, value)
    with patch("qmdpa.config.software_identity", return_value={"coupling_sha256": "first"}):
        first = dispatch.load_run_config(path)
    with patch("qmdpa.config.software_identity", return_value={"coupling_sha256": "second"}):
        second = dispatch.load_run_config(path)
    assert first["physics_sha256"] != second["physics_sha256"]
