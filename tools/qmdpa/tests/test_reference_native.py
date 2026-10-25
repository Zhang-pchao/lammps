# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Reference model admission, Hamiltonian selection and restart contracts."""

import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.inference import RAW_PROFILE
from qmdpa.reference_config import HAMILTONIAN, load_reference_config


@pytest.fixture
def request_config(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"pinned test model")
    (tmp_path / "atoms.data").write_bytes(b"pinned test geometry")
    q = {"atom_ids": [1, 2, 3], "symbols": ["O", "H", "H"], "molecules": [[1, 2, 3]]}
    e = {"atom_ids": [4, 5, 6], "symbols": ["O", "H", "H"], "molecules": [[4, 5, 6]]}
    config = {
        "schema_version": 1,
        "hamiltonian": HAMILTONIAN,
        "model": "model.pt",
        "model_sha256": hashlib.sha256(b"pinned test model").hexdigest(),
        "data_file": "atoms.data",
        "type_symbols": ["O", "H"],
        "periodic": False,
        "qm_region": q,
        "environment_region": e,
        "full_charge_spin": [0, 1],
        "region_charge_spin": [0, 1],
        "qm": {"xc": "hf", "basis": "sto-3g", "charge": 0, "multiplicity": 1},
        "embedding": {
            "atom_ids": [4, 5, 6],
            "charges_e": [-0.8, 0.4, 0.4],
            "source": "declared environment test charges",
        },
        "reference": {
            "form": "fixed-point-charge-cross-coulomb",
            "qm_charges": {
                "atom_ids": [1, 2, 3],
                "charges_e": [-0.8, 0.4, 0.4],
                "source": "declared QM reference charges",
            },
        },
        "inference_profile": RAW_PROFILE,
        "timestep_fs": 0.1,
        "steps": 2,
    }
    path = tmp_path / "config.json"
    return config, path


def load(pair, loader=load_reference_config):
    config, path = pair
    path.write_text(json.dumps(config))
    return loader(path)


def test_reference_total_assembly_and_selection(request_config):
    direct = load(request_config)
    description = describe_run(direct)
    assert direct["provider_contract"]["force_coordinates"] == ["qm", "environment"]
    assert description["assembly"] == {
        "native_pair": "zero",
        "external_quantity": "complete-total-energy-and-forces",
    }
    assert [
        (x["name"], x["coefficient"]) for x in description["energy_contract"]["total_terms"]
    ] == [("low_full", 1), ("low_qm", -1), ("embedded_ground", 1), ("reference_cross_coulomb", -1)]
    value, path = request_config
    value.pop("hamiltonian")
    value["coupling"] = {
        "surface": "ground",
        "embedding": "electrostatic",
        "ee_model": "reference-subtractive",
    }
    selected = load((value, path), load_run_config)
    assert selected["physics_sha256"] == direct["physics_sha256"]
    assert describe_run(selected)["reference"] == value["reference"]


def test_budget_does_not_change_reference_identity(request_config):
    original = load(request_config)["physics_sha256"]
    request_config[0].update(steps=15, sample_interval=3)
    assert load(request_config)["physics_sha256"] == original


@pytest.mark.parametrize(
    "change",
    [
        "reference_values",
        "embedding_values",
        "reference_source",
        "embedding_source",
        "time_step",
        "model_bytes",
        "data_bytes",
    ],
)
def test_reference_identity_covers_inputs(request_config, change):
    config, path = request_config
    original = load(request_config)["physics_sha256"]
    if change == "reference_values":
        config["reference"]["qm_charges"]["charges_e"] = [-0.6, 0.3, 0.3]
    elif change == "embedding_values":
        config["embedding"]["charges_e"] = [-0.6, 0.3, 0.3]
    elif change == "reference_source":
        config["reference"]["qm_charges"]["source"] = "another declared reference"
    elif change == "embedding_source":
        config["embedding"]["source"] = "another declared field"
    elif change == "time_step":
        config["timestep_fs"] = 0.05
    elif change == "model_bytes":
        (path.parent / "model.pt").write_bytes(b"new pinned model")
        config["model_sha256"] = hashlib.sha256(b"new pinned model").hexdigest()
    else:
        (path.parent / "atoms.data").write_bytes(b"changed geometry")
    assert load(request_config)["physics_sha256"] != original


@pytest.mark.parametrize(
    "change",
    [
        "periodic",
        "wrong_model",
        "wrong_profile",
        "wrong_hash",
        "missing_state",
        "full_charge",
        "reference_charge",
        "overlap",
        "reference_order",
        "embedding_order",
        "reference_form",
        "extra_lj",
        "radii",
        "extra_reference",
        "unknown_symbol",
        "gpu",
        "selected",
        "seed_only",
        "temperature_only",
        "duplicate_group",
        "wrong_hamiltonian",
    ],
)
def test_reference_rejects_incompatible_inputs(request_config, change):
    c, p = request_config
    if change == "periodic":
        c["periodic"] = True
    elif change == "wrong_model":
        c["model"] = "model.pt2"
    elif change == "wrong_profile":
        c["inference_profile"] = "automatic"
    elif change == "wrong_hash":
        c["model_sha256"] = "0" * 64
    elif change == "missing_state":
        c.pop("full_charge_spin")
    elif change == "full_charge":
        c["full_charge_spin"] = [1, 1]
    elif change == "reference_charge":
        c["reference"]["qm_charges"]["charges_e"][0] = 0
    elif change == "overlap":
        c["environment_region"] = copy.deepcopy(c["qm_region"])
    elif change == "reference_order":
        c["reference"]["qm_charges"]["atom_ids"] = [2, 1, 3]
    elif change == "embedding_order":
        c["embedding"]["atom_ids"] = [5, 4, 6]
    elif change == "reference_form":
        c["reference"]["form"] = "gap-anchor"
    elif change == "extra_lj":
        c["cross"] = {"form": "lj12-6"}
    elif change == "radii":
        c["embedding"]["radii_angstrom"] = [0.1] * 3
    elif change == "extra_reference":
        c["reference"]["environment_charges"] = c["embedding"]
    elif change == "unknown_symbol":
        c["type_symbols"] = ["H"]
    elif change in ("gpu", "selected"):
        c["provider"] = {
            "engine": "gpu4pyscf" if change == "gpu" else "pyscf",
            "device": "cuda" if change == "gpu" else "cpu",
            "state": {"method": "ground" if change == "gpu" else "rhf-tda"},
        }
    elif change == "seed_only":
        c["velocity_seed"] = 17
    elif change == "temperature_only":
        c["temperature_K"] = 300
    elif change == "duplicate_group":
        c["environment_region"]["molecules"] = [[4, 4, 6]]
    else:
        c["hamiltonian"] = "finite-electrostatic-additive"
    with pytest.raises((ValueError, TypeError, KeyError)):
        load((c, p))


def test_reference_rejects_ambiguous_json(request_config):
    config, path = request_config
    raw = json.dumps(config).replace('"steps": 2', '"steps": 2, "steps": 3')
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_reference_config(path)


def test_reference_reuses_owner_lifecycle():
    from qmdpa import reference_cli
    from qmdpa.reference_config import load_reference_config

    with patch("qmdpa.cli._run", return_value="accepted") as run:
        assert reference_cli.run("config.json", "output", restart="checkpoint") == "accepted"
    assert run.call_args.kwargs == {
        "restart": "checkpoint",
        "config_loader": load_reference_config,
        "setup_pair": reference_cli._zero_pair,
        "make_coupling": reference_cli._reference_coupling,
    }
