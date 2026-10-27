# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Metal occupations, reciprocal sampling and force-consistent energy contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_abacus_provider import request
from test_dispatch import save

from qmdpa.abacus import validate_abacus_settings
from qmdpa.abacus_worker import _calculator_arguments, _evaluation_response
from qmdpa.core import Geometry
from qmdpa.dispatch import describe_run, load_run_config

METAL = {"kmesh": [2, 2, 2], "smearing_method": "fd",
         "smearing_sigma": 0.01, "nbands": 56}


def settings(tmp_path):
    return {**request(tmp_path)["qm"], **METAL}


@pytest.mark.parametrize("key,value", [
    ("kmesh", [1, 2]), ("kmesh", [1, 0, 1]), ("kmesh", [True, 1, 1]),
    ("kmesh", [1.0, 1, 1]), ("kmesh", "222"), ("smearing_method", "gaussian"),
    ("smearing_method", []), ("smearing_sigma", 0), ("smearing_sigma", -0.1),
    ("smearing_sigma", True), ("smearing_sigma", float("nan")),
    ("smearing_sigma", float("inf")), ("nbands", 0), ("nbands", 1.0),
    ("nbands", True),
])
def test_invalid_metal_controls_fail_before_backend(tmp_path, key, value):
    config = settings(tmp_path)
    config[key] = value
    with pytest.raises(ValueError):
        validate_abacus_settings(config, tmp_path)


@pytest.mark.parametrize("missing", ["smearing_sigma", "nbands"])
def test_fd_requires_explicit_temperature_and_band_budget(tmp_path, missing):
    config = settings(tmp_path)
    del config[missing]
    with pytest.raises(ValueError, match="require explicit"):
        validate_abacus_settings(config, tmp_path)


def test_fixed_occupations_reject_unused_sigma(tmp_path):
    config = settings(tmp_path)
    config["smearing_method"] = "fixed"
    with pytest.raises(ValueError, match="requires Fermi-Dirac"):
        validate_abacus_settings(config, tmp_path)


def test_legacy_settings_do_not_acquire_new_defaults(tmp_path):
    resolved = validate_abacus_settings(request(tmp_path)["qm"], tmp_path)
    assert not set(METAL).intersection(resolved)


@pytest.mark.parametrize("mode", ["fileio", "socket"])
def test_native_controls_pass_unchanged_in_both_modes(tmp_path, mode):
    config = settings(tmp_path)
    config["execution_mode"] = mode
    resolved = validate_abacus_settings(config, tmp_path)
    directory = tmp_path / mode
    directory.mkdir()
    with patch.dict("sys.modules", {"abacuslite": SimpleNamespace(
            AbacusProfile=lambda **kw: kw)}):
        checked, arguments = _calculator_arguments(
            {"settings": resolved, "directory": str(directory)})
    assert checked == resolved
    assert arguments["kpts"] == {
        "mode": "mp-sampling", "gamma-centered": True,
        "nk": [2, 2, 2], "kshift": [0, 0, 0]}
    assert {key: arguments["inp"][key] for key in METAL if key != "kmesh"} == {
        key: value for key, value in METAL.items() if key != "kmesh"}


def test_metal_energy_uses_free_energy_and_never_sigma_zero():
    geometry = Geometry((1,), ("Cu",), [[0, 0, 0]])
    calculator = SimpleNamespace(results={"energy": -1.0, "free_energy": -1.4,
                                          "forces": np.zeros((1, 3))})
    value = _evaluation_response(geometry, calculator, METAL)
    assert value["energy_ev"] == -1.4
    assert value["energy_definition"] == "pbe-electronic-free-energy-plus-ions"
    assert value["smearing_sigma_ry"] == 0.01
    assert _evaluation_response(geometry, calculator)["energy_ev"] == -1.0
    del calculator.results["free_energy"]
    with pytest.raises(KeyError):
        _evaluation_response(geometry, calculator, METAL)


@pytest.mark.parametrize("energy", [float("nan"), float("inf")])
def test_nonfinite_free_energy_rejected(energy):
    geometry = Geometry((1,), ("Cu",), [[0, 0, 0]])
    with pytest.raises(ValueError):
        _evaluation_response(geometry, SimpleNamespace(results={
            "energy": -1, "free_energy": energy, "forces": np.zeros((1, 3))}), METAL)


@pytest.mark.parametrize("key,value", [
    ("kmesh", [3, 2, 2]), ("smearing_sigma", 0.02), ("nbands", 60)])
def test_new_controls_change_restart_identity_and_description(tmp_path, key, value):
    config = request(tmp_path)
    config["qm"].update(METAL)
    first = load_run_config(save(tmp_path, config))
    info = describe_run(first)["provider_contract"]
    assert info["energy"] == "pseudopotential-pbe-electronic-free-energy"
    assert info["electronic_controls"] == METAL
    config["qm"][key] = value
    second = load_run_config(save(tmp_path, config))
    assert first["physics_sha256"] != second["physics_sha256"]


def test_non_gamma_fixed_occupation_description(tmp_path):
    config = request(tmp_path)
    config["qm"]["kmesh"] = [2, 2, 2]
    info = describe_run(load_run_config(save(tmp_path, config)))["provider_contract"]
    assert info["electronic_model"] == "pw-pbe-kmesh-fixed-occupation"
    assert info["energy"] == "pseudopotential-pbe-total"


def test_periodic_ee_does_not_silently_admit_metal_controls(tmp_path):
    config = settings(tmp_path)
    config["periodic_field"] = {}
    with pytest.raises(ValueError, match="does not admit metal"):
        validate_abacus_settings(config, tmp_path)
