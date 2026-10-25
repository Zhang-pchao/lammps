# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Schema-2 charge resolution and schema-1 compatibility at the driver boundary."""

import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from qmdpa.additive_config import load_additive_config
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.inference import RAW_PROFILE


@pytest.fixture
def case(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"test model identity only")
    (tmp_path / "atoms.data").write_bytes(b"test input identity only")
    return {
        "schema_version": 2,
        "hamiltonian": "finite-electrostatic-additive",
        "model": "model.pt",
        "model_sha256": hashlib.sha256(b"test model identity only").hexdigest(),
        "data_file": "atoms.data",
        "periodic": False,
        "type_symbols": ["O", "H"],
        "qm_region": {"atom_ids": [1], "symbols": ["H"], "molecules": [[1]]},
        "environment_region": {
            "atom_ids": [4, 5, 6],
            "symbols": ["O", "H", "H"],
            "molecules": [[4, 5, 6]],
        },
        "environment_charge_spin": [0, 1],
        "embedding": {
            "charge_request": {
                "kind": "water-preset",
                "preset": "TIP3P-original-1983",
                "units": "e",
                "source": "Explicit test charge template, not a fitted potential",
                "expected_total_charge_e": 0,
                "groups": [[4, 5, 6]],
            },
        },
        "cross": {
            "form": "lj12-6",
            "source": "Independent explicit test cross parameters",
            "mixing": "none",
            "cutoff": None,
            "pairs": [
                {
                    "qm_symbol": "H",
                    "environment_symbol": symbol,
                    "epsilon_ev": 0,
                    "sigma_angstrom": 1,
                }
                for symbol in ("O", "H")
            ],
        },
        "qm": {"xc": "pbe", "basis": "sto-3g", "multiplicity": 2},
        "inference_profile": RAW_PROFILE,
        "timestep_fs": 0.25,
        "steps": 2,
    }


def load(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with patch("qmdpa.additive_config.importlib.metadata.version", return_value="test"):
        return load_additive_config(path)


@pytest.mark.parametrize("radii", [None, [0.5, 0.4, 0.4]])
def test_schema2_resolves_once_to_existing_provider_fields(tmp_path, case, radii):
    if radii is not None:
        case["embedding"]["radii_angstrom"] = radii
    loaded = load(tmp_path, case)
    explicit = loaded["embedding"]
    provider_input = FixedEmbeddingCharges(**explicit)
    assert provider_input.atom_ids == (4, 5, 6)
    assert provider_input.charges_e.tolist() == [-0.834, 0.417, 0.417]
    assert loaded["charge_resolution"]["details"]["preset"] == "TIP3P-original-1983"
    if radii is None:
        assert provider_input.radii_angstrom is None
    else:
        assert provider_input.radii_angstrom.tolist() == radii
    assert "charge_request" not in explicit
    assert loaded["cross"] == case["cross"]
    assert loaded["qm"]["charge"] == 0
    assert loaded["qm"]["multiplicity"] == 2


@pytest.mark.parametrize("radii", [None, [0.5, 0.4, 0.4]])
def test_schema1_preserves_existing_explicit_point_and_gaussian_input(tmp_path, case, radii):
    case["schema_version"] = 1
    embedding = {"atom_ids": [4, 5, 6], "charges_e": [-0.834, 0.417, 0.417], "source": "Legacy"}
    if radii is not None:
        embedding["radii_angstrom"] = radii
    case["embedding"] = embedding
    loaded = load(tmp_path, case)
    assert loaded["embedding"] == embedding
    assert "charge_resolution" not in loaded
    case["embedding"]["atom_ids"] = [5, 4, 6]
    with pytest.raises(ValueError, match="IDs"):
        load(tmp_path, case)


def test_override_total_must_also_match_explicit_environment_electronic_charge(tmp_path, case):
    request = case["embedding"]["charge_request"]
    request["expected_total_charge_e"] = 1
    request["overrides"] = [{"atom_id": 4, "charge_e": 0.166, "source": "Ionic diagnostic"}]
    with pytest.raises(ValueError, match="match the DPA charge"):
        load(tmp_path, case)
    case["environment_charge_spin"] = [1, 1]
    loaded = load(tmp_path, case)
    assert loaded["charge_resolution"]["customized"] is True
    assert loaded["qm"]["charge"] == 0
    assert loaded["embedding"]["charges_e"] == [0.166, 0.417, 0.417]


def test_charge_source_parameters_and_override_provenance_affect_restart_identity(tmp_path, case):
    baseline = load(tmp_path, case)["physics_sha256"]
    for changed_key, value in (("preset", "SPC"), ("source", "Another declared source")):
        changed = copy.deepcopy(case)
        changed["embedding"]["charge_request"][changed_key] = value
        assert load(tmp_path, changed)["physics_sha256"] != baseline
    changed = copy.deepcopy(case)
    changed["embedding"]["charge_request"]["overrides"] = [
        {"atom_id": 4, "charge_e": -0.834, "source": "Explicitly retained but customized"}
    ]
    assert load(tmp_path, changed)["physics_sha256"] != baseline
    changed = copy.deepcopy(case)
    changed["steps"] = 10
    changed["sample_interval"] = 2
    assert load(tmp_path, changed)["physics_sha256"] == baseline


def test_external_file_is_content_pinned_and_relocation_does_not_change_restart(tmp_path, case):
    raw = json.dumps(
        {
            "schema_version": 1,
            "units": "e",
            "source": "Explicit imported charge table",
            "charges": [
                {"atom_id": 4, "charge_e": -0.834},
                {"atom_id": 5, "charge_e": 0.417},
                {"atom_id": 6, "charge_e": 0.417},
            ],
        }
    ).encode()
    first = tmp_path / "charges.json"
    first.write_bytes(raw)
    request = {
        "kind": "file",
        "path": first.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source": "Explicit file selection",
        "units": "e",
        "expected_total_charge_e": 0,
    }
    case["embedding"] = {"charge_request": request}
    original = load(tmp_path, case)
    assert original["charge_resolution"]["details"]["sha256"] == request["sha256"]
    other = tmp_path / "moved" / "another-name.json"
    other.parent.mkdir()
    other.write_bytes(raw)
    request["path"] = str(other)
    assert load(tmp_path, case)["physics_sha256"] == original["physics_sha256"]
    other.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="SHA256 differs"):
        load(tmp_path, case)
    request["sha256"] = hashlib.sha256(raw + b"\n").hexdigest()
    changed = load(tmp_path, case)
    assert changed["physics_sha256"] != original["physics_sha256"]
    assert changed["embedding"]["charges_e"] == original["embedding"]["charges_e"]


@pytest.mark.parametrize(
    "embedding",
    [
        {"atom_ids": [4, 5, 6], "charges_e": [-0.834, 0.417, 0.417], "source": "Legacy"},
        {"charge_request": {}, "source": "Mixed schema"},
        {"charge_request": {}, "radii_angstrom": [0.5], "charges_e": [0]},
        {},
    ],
)
def test_schema2_rejects_legacy_mixed_or_missing_charge_requests(tmp_path, case, embedding):
    case["embedding"] = embedding
    with pytest.raises(ValueError):
        load(tmp_path, case)


def test_schema1_rejects_new_keys_and_schema2_rejects_extra_source_fields(tmp_path, case):
    case["schema_version"] = 1
    with pytest.raises(ValueError):
        load(tmp_path, case)
    case["schema_version"] = 2
    case["embedding"]["charge_request"]["charges"] = []
    with pytest.raises(ValueError, match="unsupported"):
        load(tmp_path, case)
    del case["embedding"]["charge_request"]["charges"]
    case["charge_resolution"] = {}
    with pytest.raises(ValueError, match="unsupported"):
        load(tmp_path, case)


def test_bad_gaussian_radii_remain_rejected_after_charge_resolution(tmp_path, case):
    case["embedding"]["radii_angstrom"] = [0.5]
    with pytest.raises(ValueError, match="radius"):
        load(tmp_path, case)


@pytest.mark.parametrize("schema", [True, 0, 3, "2"])
def test_unrecognized_schema_versions_are_rejected(tmp_path, case, schema):
    case["schema_version"] = schema
    with pytest.raises(ValueError, match="schema_version"):
        load(tmp_path, case)


@pytest.mark.parametrize("duplicate", ["nested-charge", "embedding", "schema"])
def test_schema2_json_duplicate_keys_are_rejected_before_resolution(tmp_path, case, duplicate):
    raw = json.dumps(case)
    if duplicate == "nested-charge":
        raw = raw.replace(
            '"expected_total_charge_e": 0',
            '"expected_total_charge_e": 1, "expected_total_charge_e": 0',
            1,
        )
    elif duplicate == "embedding":
        raw = raw.replace('"embedding": {', '"embedding": {}, "embedding": {', 1)
    else:
        raw = raw.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2', 1)
    path = tmp_path / "config.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_additive_config(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_schema2_json_nonfinite_constants_are_rejected_before_resolution(tmp_path, case, constant):
    raw = json.dumps(case).replace(
        '"expected_total_charge_e": 0', f'"expected_total_charge_e": {constant}'
    )
    path = tmp_path / "config.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="Nonfinite JSON constant"):
        load_additive_config(path)


def test_schema1_retains_legacy_json_key_parsing(tmp_path, case):
    case["schema_version"] = 1
    case["embedding"] = {
        "atom_ids": [4, 5, 6],
        "charges_e": [-0.834, 0.417, 0.417],
        "source": "Legacy",
    }
    expected = load(tmp_path, case)["physics_sha256"]
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(case).replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
        )
    )
    with patch("qmdpa.additive_config.importlib.metadata.version", return_value="test"):
        assert load_additive_config(path)["physics_sha256"] == expected


def test_schema2_both_total_checks_use_accurate_summation(tmp_path, case):
    # Arithmetic stress input only, not a physical charge model.
    case["embedding"] = {
        "charge_request": {
            "kind": "per-atom",
            "units": "e",
            "source": "Cancellation-sensitive arithmetic control",
            "expected_total_charge_e": 1,
            "charges": [
                {"atom_id": atom, "charge_e": value}
                for atom, value in zip((4, 5, 6), (1e16, 1, -1e16))
            ],
        }
    }
    case["environment_charge_spin"] = [1, 1]
    assert load(tmp_path, case)["charge_resolution"]["total_charge_e"] == 1
