# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Declared ABACUS execution files and rejection before accepting results."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_abacus import FAKE_WORKER
from test_abacus_provider import request
from test_dispatch import save

from qmdpa import abacus, dispatch
from qmdpa.abacus_worker import _calculator_arguments
from qmdpa.core import EvaluationError, Geometry
from qmdpa.providers import make_qm_provider


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path):
    config = request(tmp_path)
    files = {}
    for label in ("native_binary", "blas"):
        path = tmp_path / (label + ".fixture")
        path.write_bytes(b"Synthetic execution dependency; never loaded: " + label.encode())
        files[label] = {"path": path.name, "sha256": digest(path)}
    config["qm"]["runtime_files"] = files
    return config


def geometry():
    return Geometry((1,), ("H",), [[0.1, 0.2, 0.3]])


def backend(case, tmp_path, mode="fileio"):
    settings = abacus.validate_abacus_settings(case["qm"], tmp_path)
    settings.pop("engine")
    settings["execution_mode"] = mode
    return abacus.AbacusBackend(**settings, directory=tmp_path / "runtime")


def fake_command(tmp_path):
    (tmp_path / "fake-interface.py").write_text("# Synthetic worker identity\n")
    return [sys.executable, "-u", "-c", FAKE_WORKER, "success", str(tmp_path / "worker.jsonl")]


def check_failure(provider, tmp_path, successful=0):
    assert provider.statistics["successful_evaluations"] == successful
    assert provider.statistics["failed_evaluations"] == 1
    assert provider._process is None
    failure = json.loads((tmp_path / "runtime/failure.json").read_text())
    assert "runtime file" in failure["error"] or "execution-file" in failure["error"]
    with pytest.raises(EvaluationError, match="ABACUS evaluation failed"):
        provider.evaluate(geometry())
    assert provider.statistics["failed_evaluations"] == 1
    provider.close_checked()


def test_relative_paths_hash_case_and_inputs_are_preserved(case, tmp_path, monkeypatch):
    case["qm"]["runtime_files"]["blas"]["sha256"] = (
        case["qm"]["runtime_files"]["blas"]["sha256"].upper()
    )
    original = copy.deepcopy(case)
    path = save(tmp_path, case)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = dispatch.load_run_config(path)
    assert case == original
    for label, item in original["qm"]["runtime_files"].items():
        assert loaded["qm"]["runtime_files"][label] == {
            "path": str(tmp_path / item["path"]), "sha256": item["sha256"].lower()
        }
    with patch("qmdpa.abacus.AbacusBackend") as engine:
        make_qm_provider(loaded["qm"], provider=loaded["provider"], directory=tmp_path / "new")
    assert engine.call_args.kwargs["runtime_files"] == loaded["qm"]["runtime_files"]


def test_omission_preserves_normalized_legacy_shape(case, tmp_path):
    case["qm"].pop("runtime_files")
    legacy = dispatch.load_run_config(save(tmp_path, case))
    assert "runtime_files" not in legacy["qm"]
    provider = backend(case, tmp_path)
    try:
        assert "runtime_files" not in provider._settings
        assert "runtime_files" not in json.loads((provider.directory / "settings.json").read_text())
    finally:
        provider.close_checked()


@pytest.mark.parametrize("value", [None, {}, [], "", False, 1])
def test_invalid_file_sets_fail_before_backend(case, tmp_path, value):
    case["qm"]["runtime_files"] = value
    with patch("qmdpa.abacus.AbacusBackend") as engine:
        with pytest.raises(ValueError, match="runtime_files"):
            dispatch.load_run_config(save(tmp_path, case))
    engine.assert_not_called()


@pytest.mark.parametrize("label", ["", "space label", ".hidden", "../lib", "a/b", "\u00e9", 3])
def test_invalid_names_are_rejected(case, tmp_path, label):
    case["qm"]["runtime_files"] = {label: case["qm"]["runtime_files"]["blas"]}
    with pytest.raises(ValueError, match="named runtime file"):
        abacus.validate_abacus_settings(case["qm"], tmp_path)


@pytest.mark.parametrize("kind", ["missing", "extra", "nonmapping", "path", "hash", "wrong-hash",
                                 "missing-file", "directory"])
def test_invalid_file_identities_are_rejected(case, tmp_path, kind):
    item = case["qm"]["runtime_files"]["blas"]
    if kind == "missing":
        item.pop("sha256")
    elif kind == "extra":
        item["version"] = "1"
    elif kind == "nonmapping":
        case["qm"]["runtime_files"]["blas"] = []
    elif kind == "path":
        item["path"] = ""
    elif kind == "hash":
        item["sha256"] = "abc"
    elif kind == "wrong-hash":
        item["sha256"] = "0" * 64
    elif kind == "missing-file":
        (tmp_path / item["path"]).unlink()
    else:
        item["path"] = str(tmp_path)
    with pytest.raises(ValueError):
        abacus.validate_abacus_settings(case["qm"], tmp_path)


def test_duplicate_symlink_targets_are_rejected(case, tmp_path):
    item = case["qm"]["runtime_files"]["blas"]
    link = tmp_path / "alias"
    link.symlink_to(item["path"])
    case["qm"]["runtime_files"]["another"] = {**item, "path": link.name}
    with pytest.raises(ValueError, match="duplicate resolved paths"):
        abacus.validate_abacus_settings(case["qm"], tmp_path)


@pytest.mark.parametrize("change", ["contents", "path", "label", "omission"])
def test_changed_dependencies_change_restart_identity(case, tmp_path, change):
    before = dispatch.load_run_config(save(tmp_path, case))["physics_sha256"]
    item = case["qm"]["runtime_files"]["blas"]
    if change == "contents":
        path = tmp_path / item["path"]
        path.write_bytes(b"Different declared runtime")
        item["sha256"] = digest(path)
    elif change == "path":
        path = tmp_path / "another.fixture"
        path.write_bytes((tmp_path / item["path"]).read_bytes())
        item["path"] = path.name
    elif change == "label":
        case["qm"]["runtime_files"]["renamed"] = case["qm"]["runtime_files"].pop("blas")
    else:
        case["qm"].pop("runtime_files")
    after = dispatch.load_run_config(save(tmp_path, case))["physics_sha256"]
    assert before != after


@pytest.mark.parametrize("mode", ["socket", "fileio"])
def test_unchanged_files_preserve_worker_results(case, tmp_path, mode):
    provider = backend(case, tmp_path, mode)
    try:
        with patch("qmdpa.abacus._worker_command", return_value=fake_command(tmp_path)):
            first = provider.evaluate(geometry())
            second = provider.evaluate(geometry())
        np.testing.assert_array_equal(first.forces, -2 * geometry().positions)
        np.testing.assert_array_equal(first.forces, second.forces)
        assert first.energy == second.energy
        assert provider.statistics["successful_evaluations"] == 2
        assert provider.statistics["failed_evaluations"] == 0
        rows = [json.loads(line) for line in (tmp_path / "worker.jsonl").read_text().splitlines()]
        initial = [r["request"] for r in rows if r["request"]["op"] == "initialize"]
        assert initial and all(r["settings"]["runtime_files"] == provider._settings["runtime_files"]
                               for r in initial)
    finally:
        provider.close_checked()


@pytest.mark.parametrize("mode", ["socket", "fileio"])
def test_precheck_refuses_start_and_permanently_fails(case, tmp_path, mode):
    provider = backend(case, tmp_path, mode)
    (tmp_path / case["qm"]["runtime_files"]["blas"]["path"]).write_bytes(b"Changed before call")
    with patch.object(provider, "_start") as start:
        with pytest.raises(EvaluationError, match="runtime file blas"):
            provider.evaluate(geometry())
        start.assert_not_called()
    assert not (tmp_path / "runtime/worker.json").exists()
    check_failure(provider, tmp_path)


@pytest.mark.parametrize("mode", ["socket", "fileio"])
def test_postcheck_refuses_result_and_cleans_up(case, tmp_path, mode):
    provider = backend(case, tmp_path, mode)
    exchange = provider._exchange

    def mutate_after_evaluation(request, deadline):
        response = exchange(request, deadline)
        if request["op"] == "evaluate":
            (tmp_path / case["qm"]["runtime_files"]["blas"]["path"]).write_bytes(
                b"Changed while evaluating"
            )
        return response

    with (patch("qmdpa.abacus._worker_command", return_value=fake_command(tmp_path)),
          patch.object(provider, "_exchange", side_effect=mutate_after_evaluation)):
        with pytest.raises(EvaluationError, match="runtime file blas"):
            provider.evaluate(geometry())
    rows = [json.loads(line) for line in (tmp_path / "worker.jsonl").read_text().splitlines()]
    assert sum(r["request"]["op"] == "evaluate" for r in rows) == 1
    check_failure(provider, tmp_path)


def test_change_after_success_never_returns_stale_forces(case, tmp_path):
    provider = backend(case, tmp_path, "socket")
    with patch("qmdpa.abacus._worker_command", return_value=fake_command(tmp_path)):
        provider.evaluate(geometry())
    (tmp_path / case["qm"]["runtime_files"]["blas"]["path"]).unlink()
    with pytest.raises(EvaluationError, match="runtime file blas"):
        provider.evaluate(geometry())
    check_failure(provider, tmp_path, successful=1)
    rows = [json.loads(line) for line in (tmp_path / "worker.jsonl").read_text().splitlines()]
    assert sum(r["request"]["op"] == "evaluate" for r in rows) == 1


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_identity_checks_share_request_deadline(case, tmp_path, stage):
    provider = backend(case, tmp_path)
    clock = [10.0]
    checks = [0]
    original = provider._check_runtime_files

    def check():
        original()
        checks[0] += 1
        if checks[0] == (1 if stage == "pre" else 2):
            clock[0] += provider._settings["timeout_seconds"]

    response = {"status": "ok", "scf_converged": True, "atom_ids": [1], "symbols": ["H"],
                "energy_ev": 0.14, "forces_ev_per_angstrom": [[-0.2, -0.4, -0.6]]}
    with (patch("qmdpa.abacus.time.monotonic", side_effect=lambda: clock[0]),
          patch.object(provider, "_check_runtime_files", side_effect=check),
          patch.object(provider, "_start") as start,
          patch.object(provider, "_exchange", return_value=response) as exchange,
          patch.object(provider, "_finish_fileio") as finish):
        with pytest.raises(EvaluationError, match="execution-file verification"):
            provider.evaluate(geometry())
        if stage == "pre":
            start.assert_not_called()
            exchange.assert_not_called()
            finish.assert_not_called()
        else:
            start.assert_called_once()
            exchange.assert_called_once()
            finish.assert_called_once()
    check_failure(provider, tmp_path)


@pytest.mark.parametrize("changed", [False, True])
def test_worker_validates_files_before_constructing_calculator(case, tmp_path, changed):
    settings = abacus.validate_abacus_settings(case["qm"], tmp_path)
    if changed:
        (tmp_path / case["qm"]["runtime_files"]["blas"]["path"]).write_bytes(b"Changed")
    directory = tmp_path / "worker"
    directory.mkdir()
    profile = Mock(return_value=object())
    with patch.dict(sys.modules, {"abacuslite": SimpleNamespace(AbacusProfile=profile)}):
        if changed:
            with pytest.raises(ValueError, match="runtime file blas"):
                _calculator_arguments({"settings": settings, "directory": directory})
            profile.assert_not_called()
            assert not (directory / "pseudopotentials").exists()
        else:
            resolved, arguments = _calculator_arguments(
                {"settings": settings, "directory": directory}
            )
            assert resolved["runtime_files"] == settings["runtime_files"]
            assert "runtime_files" not in arguments
            assert "runtime_files" not in arguments["inp"]
            profile.assert_called_once()
