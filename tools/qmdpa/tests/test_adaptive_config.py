# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Finite adaptive declarations, preflight validation and restart identity."""

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from qmdpa.adaptive_config import HAMILTONIAN, load_adaptive_config
from qmdpa.config import file_digest
from qmdpa.inference import RAW_PROFILE
from qmdpa.runtime import _validate_restart


def document():
    return {
        "schema_version": 1,
        "hamiltonian": HAMILTONIAN,
        "data_file": "water.data",
        "model": "model.pt",
        "model_sha256": hashlib.sha256(b"unevaluated model fixture").hexdigest(),
        "type_symbols": ["O", "H"],
        "periodic": False,
        "full_region": {
            "atom_ids": list(range(1, 10)),
            "symbols": ["O", "H", "H"] * 3,
            "molecules": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        },
        "groups": [
            {"group_id": name, "atom_ids": ids, "symbols": ["O", "H", "H"]}
            for name, ids in (("core", [1, 2, 3]), ("a", [4, 5, 6]), ("b", [7, 8, 9]))
        ],
        "partitions": [
            {
                "partition_id": name,
                "qm_group_ids": ["core", candidate],
                "qm_charge": 0,
                "qm_multiplicity": 1,
                "method_identity": "PBE/def2-SVP ground-state control",
                "energy_reference": "same pinned QM and DPA reference",
                "electronic_state": "ground",
            }
            for name, candidate in (("A", "a"), ("B", "b"))
        ],
        "switch": {
            "method": "quintic-distance-exchange",
            "center_atom_id": 1,
            "first_atom_id": 4,
            "second_atom_id": 7,
            "half_width_angstrom": 0.4,
        },
        "full_charge_spin": [0, 1],
        "region_charge_spin": [0, 1],
        "provider": {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}},
        "qm": {
            "xc": "pbe",
            "basis": "def2-svp",
            "charge": 0,
            "multiplicity": 1,
            "grid_response": True,
            "threads": 1,
        },
        "inference_profile": RAW_PROFILE,
        "timestep_fs": 0.25,
        "steps": 10,
    }


def load(tmp_path, config=None):
    (tmp_path / "model.pt").write_bytes(b"unevaluated model fixture")
    (tmp_path / "water.data").write_bytes(b"unevaluated data fixture")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document() if config is None else config))
    with patch("qmdpa.adaptive_config.importlib.metadata.version", return_value="fixture"):
        return load_adaptive_config(path)


def test_normalized_complete_groups_state_and_paths(tmp_path):
    value = load(tmp_path)
    assert value["model"] == str(tmp_path / "model.pt")
    assert value["data_file"] == str(tmp_path / "water.data")
    assert value["data_sha256"] == file_digest(tmp_path / "water.data")
    assert value["sample_interval"] == 1
    assert [group["group_id"] for group in value["groups"]] == ["a", "b", "core"]
    assert value["partitions"][0]["qm_group_ids"] == ["a", "core"]
    assert value["partitions"][0]["energy_offset_ev"] == 0.0
    assert value["partitions"][0]["offset_source"] is None
    assert value["provider_contract"]["force_coordinates"] == ["qm"]
    assert value["adaptive_contract"]["force_scope"] == "all-atoms"
    assert value["adaptive_contract"]["native_pair"] == "zero"


def test_canonical_reordering_and_budget_preserve_restart_identity(tmp_path):
    original = load(tmp_path)["physics_sha256"]
    config = document()
    config.update(steps=100, sample_interval=5, temperature_K=20, velocity_seed=421)
    config["groups"] = config["groups"][::-1]
    for group in config["groups"]:
        group["atom_ids"].reverse()
        group["symbols"].reverse()
    config["full_region"]["atom_ids"].reverse()
    config["full_region"]["symbols"].reverse()
    config["full_region"]["molecules"].reverse()
    config["partitions"][0].update(energy_offset_ev=0, offset_source=None)
    assert load(tmp_path, config)["physics_sha256"] == original


@pytest.mark.parametrize(
    "change", ["width", "anchor", "method", "reference", "offset", "qm", "timestep", "groups"]
)
def test_each_physical_policy_changes_identity(tmp_path, change):
    first = load(tmp_path)["physics_sha256"]
    config = document()
    if change == "width":
        config["switch"]["half_width_angstrom"] = 0.5
    elif change == "anchor":
        config["switch"]["first_atom_id"] = 5
    elif change == "method":
        for p in config["partitions"]:
            p["method_identity"] = "independently qualified alternate setting"
    elif change == "reference":
        for p in config["partitions"]:
            p["energy_reference"] = "different declared reference"
    elif change == "offset":
        config["partitions"][0].update(
            energy_offset_ev=0.1, offset_source="Declared offset control"
        )
    elif change == "qm":
        config["qm"]["conv_tol"] = 1e-12
    elif change == "timestep":
        config["timestep_fs"] = 0.125
    else:
        config["groups"][1]["group_id"] = "candidate-a"
        config["partitions"][0]["qm_group_ids"] = ["core", "candidate-a"]
    assert load(tmp_path, config)["physics_sha256"] != first


def test_restart_rejects_switch_offset_and_group_changes(tmp_path):
    original = load(tmp_path)
    restart = tmp_path / "restart.bin"
    restart.write_bytes(b"accepted native restart fixture")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "physics_sha256": original["physics_sha256"],
                "restart_file": restart.name,
                "restart_sha256": file_digest(restart),
                "frame": {},
            }
        )
    )
    assert _validate_restart(checkpoint, original)["restart_path"] == str(restart)
    changed = document()
    changed["switch"]["half_width_angstrom"] = 0.2
    with pytest.raises(ValueError, match="physics configuration differs"):
        _validate_restart(checkpoint, load(tmp_path, changed))
    restart.write_bytes(b"corrupted binary")
    with pytest.raises(ValueError, match="binary checksum"):
        _validate_restart(checkpoint, original)


def test_source_dependency_and_external_bytes_are_bound(tmp_path):
    first = load(tmp_path)["physics_sha256"]
    path = tmp_path / "config.json"
    with patch(
        "qmdpa.adaptive_config.importlib.metadata.version", return_value="different-version"
    ):
        assert load_adaptive_config(path)["physics_sha256"] != first
    with (
        patch(
            "qmdpa.adaptive_config.software_identity",
            return_value={"coupling_sha256": "changed-source", "packages": {}},
        ),
        patch("qmdpa.adaptive_config.importlib.metadata.version", return_value="fixture"),
    ):
        assert load_adaptive_config(path)["physics_sha256"] != first
    (tmp_path / "water.data").write_bytes(b"changed input coordinates")
    with patch("qmdpa.adaptive_config.importlib.metadata.version", return_value="fixture"):
        assert load_adaptive_config(path)["physics_sha256"] != first
    (tmp_path / "model.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="Model SHA256"):
        load_adaptive_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("periodic", True),
        ("schema_version", True),
        ("hamiltonian", "finite-electrostatic-additive"),
        ("inference_profile", "aot"),
        ("region_charge_spin", [0, 3]),
        ("full_charge_spin", [1, 1]),
        ("full_charge_spin", [False, 1]),
        ("unknown", 1),
        ("steps", True),
        ("sample_interval", 0),
        ("timestep_fs", True),
        ("timestep_fs", float("nan")),
        ("temperature_K", 20),
        ("velocity_seed", 42),
        ("type_symbols", "OH"),
        ("provider", None),
    ],
)
def test_unsupported_modes_and_bad_numerics_fail_before_loading(tmp_path, field, value):
    config = document()
    config[field] = value
    with (
        patch("qmdpa.backends.DeepMDBackend", side_effect=AssertionError("must not construct")),
        patch("qmdpa.providers.make_qm_provider", side_effect=AssertionError("must not construct")),
        pytest.raises((TypeError, ValueError)),
    ):
        load(tmp_path, config)


@pytest.mark.parametrize(
    "change",
    [
        "gpu",
        "engine",
        "excited",
        "qm-state",
        "grid-response",
        "nlc",
        "missing-q",
        "partition-state",
        "different-method",
        "different-reference",
        "unbound-offset",
        "overlap",
        "coverage",
        "composition",
        "cut-group",
        "wrong-exchange",
        "same-anchor",
        "wrong-switch",
        "extra-partition",
    ],
)
def test_unsupported_states_and_partition_families_are_rejected(tmp_path, change):
    c = document()
    if change == "gpu":
        c["provider"]["device"] = "gpu"
    elif change == "engine":
        c["provider"]["engine"] = "abacus"
    elif change == "excited":
        c["provider"]["state"]["method"] = "tda"
    elif change == "qm-state":
        c["qm"]["charge"] = 1
    elif change == "grid-response":
        c["qm"]["grid_response"] = False
    elif change == "nlc":
        c["qm"]["nlc_grid_level"] = 4
    elif change == "missing-q":
        del c["qm"]["charge"]
    elif change == "partition-state":
        c["partitions"][0]["electronic_state"] = "excited"
    elif change == "different-method":
        c["partitions"][0]["method_identity"] = "different"
    elif change == "different-reference":
        c["partitions"][0]["energy_reference"] = "different"
    elif change == "unbound-offset":
        c["partitions"][0]["energy_offset_ev"] = 0.1
    elif change == "overlap":
        c["groups"][1]["atom_ids"][0] = 1
    elif change == "coverage":
        c["full_region"]["symbols"][0] = "N"
    elif change == "composition":
        c["groups"][2]["symbols"][2] = "O"
    elif change == "cut-group":
        c["full_region"]["molecules"] = [[1, 2, 4], [3, 5, 6], [7, 8, 9]]
    elif change == "wrong-exchange":
        c["partitions"][1]["qm_group_ids"] = ["core", "a"]
    elif change == "same-anchor":
        c["switch"]["first_atom_id"] = 1
    elif change == "wrong-switch":
        c["switch"]["method"] = "nearest-region"
    else:
        c["partitions"].append(copy.deepcopy(c["partitions"][0]))
    with pytest.raises((TypeError, ValueError)):
        load(tmp_path, c)


def test_strict_json_and_nonraw_model_are_rejected(tmp_path):
    load(tmp_path)
    path = tmp_path / "config.json"
    for content in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}'):
        path.write_text(content)
        with pytest.raises(ValueError):
            load_adaptive_config(path)
    config = document()
    config["model"] = "model.pt2"
    (tmp_path / "model.pt2").write_bytes(b"unevaluated model fixture")
    with pytest.raises(ValueError, match="original .pt"):
        load(tmp_path, config)


def test_missing_optional_mpi_identity_and_required_torch(tmp_path):
    load(tmp_path)

    def version(name):
        if name == "mpi4py":
            raise PackageNotFoundError(name)
        return "fixture"

    with patch("qmdpa.adaptive_config.importlib.metadata.version", side_effect=version):
        result = load_adaptive_config(tmp_path / "config.json")
    assert result["software"]["packages"]["mpi4py"] is None
    with (
        patch(
            "qmdpa.adaptive_config.importlib.metadata.version",
            side_effect=PackageNotFoundError("torch"),
        ),
        pytest.raises(PackageNotFoundError),
    ):
        load_adaptive_config(tmp_path / "config.json")


@pytest.mark.parametrize("field,value", [("atom_ids", "123456789"), ("symbols", "OHHOHHOHH")])
def test_full_region_requires_explicit_lists(tmp_path, field, value):
    config = document()
    config["full_region"][field] = value
    with pytest.raises(ValueError, match="nonempty list"):
        load(tmp_path, config)


def test_new_registered_states_remain_rejected_by_adaptive_route(tmp_path):
    provider = {"engine": "pyscf", "device": "cpu", "state": {"method": "rhf-tda"}}
    with (
        patch("qmdpa.adaptive_config.validate_provider", return_value=provider),
        pytest.raises(ValueError, match="ground electronic state"),
    ):
        load(tmp_path)
