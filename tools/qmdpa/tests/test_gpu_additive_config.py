# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU additive selection, point-charge limits and dependency/restart identity."""

import builtins
from unittest.mock import patch

import pytest
import test_additive_config

# Register shared pytest fixtures consumed through test function arguments.
from test_additive_config import case

from qmdpa import dispatch
from qmdpa.additive_config import load_additive_config

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case']


def select(case):
    case["provider"] = {"engine": "gpu4pyscf", "device": "cuda", "state": {"method": "ground"}}
    case["qm_region"] = {"atom_ids": [1, 2], "symbols": ["H", "H"], "molecules": [[1, 2]]}
    case["qm"].update(multiplicity=1, density_fit={"auxbasis": "weigend"})
    return case


def test_point_charge_contract_and_lazy_unified_selector(tmp_path, case):
    request = select(case)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf", "cupy", "torch")):
            raise AssertionError("Unexpected scientific import: " + name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        normalized = test_additive_config.load(tmp_path, request)
        request.pop("hamiltonian")
        request["coupling"] = {
            "surface": "ground",
            "embedding": "electrostatic",
            "ee_model": "additive",
        }
        import json

        path = tmp_path / "chosen.json"
        path.write_text(json.dumps(request))
        with patch("qmdpa.config.importlib.metadata.version", return_value="test"):
            chosen = dispatch.load_run_config(path)
    assert chosen == normalized
    assert chosen["provider_contract"]["force_coordinates"] == ["qm", "environment"]
    assert chosen["provider_contract"]["charge_model"] == "point"
    description = dispatch.describe_run(chosen)
    assert description["energy_contract"]["native_baseline"] == "zero"
    assert [x["name"] for x in description["energy_contract"]["total_terms"]] == [
        "low_environment",
        "embedded_ground",
        "cross_lj",
    ]


@pytest.mark.parametrize("damage", ["gaussian", "dipole", "no-fit", "open-shell", "selected"])
def test_unsupported_gpu_field_settings_fail_before_construction(tmp_path, case, damage):
    request = select(case)
    if damage == "gaussian":
        request["embedding"]["radii_angstrom"] = [0.5, 0.4, 0.4]
    elif damage == "dipole":
        request["qm"]["include_qm_dipole"] = True
    elif damage == "no-fit":
        request["qm"].pop("density_fit")
    elif damage == "open-shell":
        request["qm"]["multiplicity"] = 2
    else:
        request["provider"]["state"]["method"] = "pbe0-tda"
    with (
        patch("qmdpa.gpu_backends.GPU4PySCFElectrostaticBackend") as backend,
        pytest.raises(ValueError),
    ):
        test_additive_config.load(tmp_path, request)
    backend.assert_not_called()


def test_gpu_charge_and_device_changes_are_bound_but_run_budget_is_not(tmp_path, case):
    request = select(case)
    gpu = test_additive_config.load(tmp_path, request)
    request["steps"] += 1
    assert test_additive_config.load(tmp_path, request)["physics_sha256"] == gpu["physics_sha256"]
    request["embedding"]["charge_request"]["source"] = "Different explicitly supplied source"
    assert test_additive_config.load(tmp_path, request)["physics_sha256"] != gpu["physics_sha256"]
    request["provider"] = {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}
    cpu = test_additive_config.load(tmp_path, request)
    assert cpu["physics_sha256"] != gpu["physics_sha256"]
    assert "gpu4pyscf-cuda12x" not in cpu["software"]["packages"]


def test_gpu_dependency_versions_change_additive_restart_identity(tmp_path, case):
    request = select(case)
    import json

    path = tmp_path / "input.json"
    path.write_text(json.dumps(request))
    with patch("qmdpa.config.importlib.metadata.version", return_value="baseline"):
        first = load_additive_config(path)
    with patch(
        "qmdpa.config.importlib.metadata.version",
        side_effect=lambda name: "changed" if name == "gpu4pyscf-cuda12x" else "baseline",
    ):
        second = load_additive_config(path)
    assert first["physics_sha256"] != second["physics_sha256"]
