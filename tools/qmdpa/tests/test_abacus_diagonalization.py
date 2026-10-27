# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit inner-solver accuracy, legacy omission and restart identity."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_abacus_provider import request
from test_dispatch import save

from qmdpa.abacus import AbacusBackend, validate_abacus_settings
from qmdpa.abacus_worker import _calculator_arguments
from qmdpa.dispatch import describe_run, load_run_config


@pytest.mark.parametrize("value", [
    True, False, None, "1e-13", -1e-13, 0.010000001,
    float("nan"), float("inf"), -float("inf"),
])
def test_invalid_floor_rejected_before_execution(tmp_path, value):
    config = request(tmp_path)["qm"]
    config["pw_diag_thr_floor"] = value
    with pytest.raises(ValueError, match="pw_diag_thr_floor"):
        validate_abacus_settings(config, tmp_path)


@pytest.mark.parametrize("mode", ["fileio", "socket"])
@pytest.mark.parametrize("value", [0, 1e-13, 0.01])
def test_floor_reaches_worker_and_recorded_settings(tmp_path, mode, value):
    config = request(tmp_path)["qm"]
    config.update(execution_mode=mode, pw_diag_thr_floor=value)
    settings = validate_abacus_settings(config, tmp_path)
    assert type(settings["pw_diag_thr_floor"]) is float
    directory = tmp_path / "worker"
    directory.mkdir()
    with patch.dict("sys.modules", {"abacuslite": SimpleNamespace(
            AbacusProfile=lambda **kw: kw)}):
        checked, arguments = _calculator_arguments(
            {"settings": settings, "directory": str(directory)})
    assert checked == settings
    assert arguments["inp"]["pw_diag_thr_floor"] == float(value)
    assert arguments["inp"]["scf_thr"] == settings["scf_thr"]
    backend_settings = {key: val for key, val in settings.items() if key != "engine"}
    backend = AbacusBackend(**backend_settings, directory=tmp_path / "backend")
    recorded = json.loads((backend.directory / "settings.json").read_text())
    assert recorded == settings
    backend.close_checked()


def test_omitted_floor_preserves_legacy_settings_and_native_input(tmp_path):
    settings = validate_abacus_settings(request(tmp_path)["qm"], tmp_path)
    assert "pw_diag_thr_floor" not in settings
    directory = tmp_path / "worker"
    directory.mkdir()
    with patch.dict("sys.modules", {"abacuslite": SimpleNamespace(
            AbacusProfile=lambda **kw: kw)}):
        checked, arguments = _calculator_arguments(
            {"settings": settings, "directory": str(directory)})
    assert checked == settings
    assert "pw_diag_thr_floor" not in arguments["inp"]
    backend_settings = {key: value for key, value in settings.items() if key != "engine"}
    backend = AbacusBackend(**backend_settings, directory=tmp_path / "backend")
    assert json.loads((backend.directory / "settings.json").read_text()) == settings
    backend.close_checked()


def test_explicit_floor_changes_restart_identity_and_description(tmp_path):
    config = request(tmp_path)
    legacy = load_run_config(save(tmp_path, config))
    config["qm"]["pw_diag_thr_floor"] = 0.0
    zero = load_run_config(save(tmp_path, config))
    config["qm"]["pw_diag_thr_floor"] = 1e-13
    bounded = load_run_config(save(tmp_path, config))
    assert len({item["physics_sha256"] for item in (legacy, zero, bounded)}) == 3
    assert describe_run(bounded)["provider_contract"]["electronic_controls"] == {
        "pw_diag_thr_floor": 1e-13}
    assert "electronic_controls" not in describe_run(legacy)["provider_contract"]


@pytest.mark.parametrize("value", [0.0, 1e-13])
def test_floor_does_not_admit_unqualified_periodic_embedding(tmp_path, value):
    config = request(tmp_path)["qm"]
    config.update(pw_diag_thr_floor=value, periodic_field={})
    with pytest.raises(ValueError, match="Periodic Gaussian embedding.*pw_diag_thr_floor"):
        validate_abacus_settings(config, tmp_path)
