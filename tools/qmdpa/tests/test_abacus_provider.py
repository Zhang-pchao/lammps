# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit ABACUS method, identities and owner resource lifecycle."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from test_cli import accepted_frame
from test_dispatch import mechanical, save

from qmdpa import cli, dispatch
from qmdpa.config import software_identity, validate_qm
from qmdpa.core import EvaluationError
from qmdpa.providers import make_qm_provider, provider_contract, validate_provider
from qmdpa.runtime import ProviderResources

PROVIDER = {"engine": "abacus", "device": "cpu", "state": {"method": "ground"}}


def request(tmp_path):
    config = mechanical(tmp_path)
    executable = tmp_path / "abacus"
    executable.write_text("#!/bin/sh\nexit 99\n")
    executable.chmod(0o700)
    pseudo = tmp_path / "H.upf"
    pseudo.write_text("Synthetic input identity; never executed")
    config["provider"] = copy.deepcopy(PROVIDER)
    config["qm"] = {
        "executable": executable.name,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "pseudopotentials": {
            "H": {"path": pseudo.name, "sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest()}
        },
        "supercell_angstrom": [12.0, 12.0, 12.0],
        "coordinate_offset_angstrom": [6.0, 6.0, 6.0],
        "coordinate_gauge": "arithmetic-centroid",
        "execution_mode": "fileio",
    }
    return config


def test_explicit_registry_contract_and_file_paths_remain_model_specific(tmp_path, monkeypatch):
    config = request(tmp_path)
    config["coupling"] = {"surface": "ground", "embedding": "mechanical"}
    path = save(tmp_path, config)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = dispatch.load_run_config(path)
    assert loaded["provider"] == PROVIDER
    assert loaded["qm"]["executable"] == str(tmp_path / "abacus")
    assert loaded["qm"]["pseudopotentials"]["H"]["path"] == str(tmp_path / "H.upf")
    info = dispatch.describe_run(loaded)
    assert info["assembly"]["native_pair"] == "deepmd"
    assert info["assembly"]["external_quantity"] == "qm-minus-region-low-correction"
    assert info["provider_contract"]["qm_boundary"] == "finite-periodic-supercell"
    assert info["provider_contract"]["energy"] == "pseudopotential-pbe-total"
    assert info["energy_contract"]["total_terms"][-1] == {
        "name": "supercell_ground",
        "coefficient": 1,
        "force_coordinates": "A",
    }
    assert set(loaded["software"]["backend_package_sources"]) == {"ase", "abacuslite"}
    assert provider_contract(PROVIDER)["history"] == "none"
    assert "backend_package_sources" not in software_identity()


@pytest.mark.parametrize(
    "change", ["socket", "offset", "periodic", "charge", "spin", "pseudo", "engine", "directory"]
)
def test_native_input_rejects_unimplemented_conditions_before_backend(tmp_path, change):
    config = request(tmp_path)
    if change == "socket":
        config["qm"]["execution_mode"] = "socket"
    elif change == "offset":
        config["qm"]["coordinate_gauge"] = "fixed-offset"
    elif change == "periodic":
        config["periodic"] = True
    elif change == "charge":
        config["qm"]["charge"] = 1
    elif change == "spin":
        config["qm"]["multiplicity"] = 3
    elif change == "pseudo":
        config["qm"]["pseudopotentials"] = {"O": config["qm"]["pseudopotentials"]["H"]}
    elif change == "engine":
        config["qm"]["engine"] = "pyscf"
    else:
        config["qm"]["directory"] = "existing-output"
    with patch("qmdpa.abacus.AbacusBackend") as engine:
        with pytest.raises(ValueError):
            dispatch.load_run_config(save(tmp_path, config))
    engine.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [("device", "cuda"), ("state", {"method": "rhf-tda"}), ("state", {"method": "pbe0-tda"})],
)
def test_no_selected_or_gpu_substitution(key, value):
    with pytest.raises(ValueError):
        validate_provider({**PROVIDER, key: value})


def test_no_electrostatic_or_adaptive_admission(tmp_path):
    with pytest.raises(ValueError):
        validate_provider(PROVIDER, electrostatic=True)
    with pytest.raises(ValueError):
        make_qm_provider({}, provider=PROVIDER, charges=object())
    with pytest.raises(ValueError, match="runtime directory"):
        make_qm_provider({}, provider=PROVIDER)
    with pytest.raises(ValueError, match="only for ABACUS"):
        make_qm_provider({}, directory=tmp_path)


def test_factory_uses_resolved_inputs_and_the_fresh_directory_only(tmp_path):
    config = request(tmp_path)
    qm = validate_qm(config["qm"], provider=PROVIDER, base_dir=tmp_path)
    target = tmp_path / "runtime"
    with (
        patch("qmdpa.abacus.AbacusBackend") as engine,
        patch("qmdpa.backends.PySCFBackend") as wrong,
    ):
        assert make_qm_provider(qm, provider=PROVIDER, directory=target) is engine.return_value
    kwargs = engine.call_args.kwargs
    assert kwargs["directory"] == target and kwargs["execution_mode"] == "fileio"
    assert "engine" not in kwargs
    wrong.assert_not_called()


@pytest.mark.parametrize(
    "key,value", [("ecutwfc", 80.0), ("scf_thr", 1e-12), ("supercell_angstrom", [14.0, 14.0, 14.0])]
)
def test_actual_model_controls_change_restart_identity(tmp_path, key, value):
    config = request(tmp_path)
    before = dispatch.load_run_config(save(tmp_path, config))["physics_sha256"]
    config["qm"][key] = value
    assert dispatch.load_run_config(save(tmp_path, config))["physics_sha256"] != before


def test_checked_close_cannot_hide_cleanup_errors_when_diagnostic_writing_fails(tmp_path):
    config = request(tmp_path)
    qm = validate_qm(config["qm"], provider=PROVIDER, base_dir=tmp_path)
    provider = make_qm_provider(qm, provider=PROVIDER, directory=tmp_path / "run")
    with (
        patch.object(provider, "_terminate_group", return_value=["Injected cleanup failure"]),
        patch.object(provider, "_diagnostic"),
    ):
        with pytest.raises(EvaluationError, match="Injected cleanup failure"):
            provider.close_checked()
    assert not (provider.directory / "cleanup.json").exists()
    with pytest.raises(EvaluationError, match="Injected cleanup failure"):
        provider.close_checked()


def test_resource_collection_closes_all_once_and_retains_errors(tmp_path):
    order = []

    def close_a():
        order.append("a")
        raise RuntimeError("a cleanup failed")

    a = SimpleNamespace(close_checked=close_a, statistics={"calls": 1})
    b = SimpleNamespace(close_checked=lambda: order.append("b"), statistics={"calls": 2})
    resources = ProviderResources(tmp_path / "run")
    resources.register(a)
    resources.register(b)
    assert len(resources.close()) == 1 and order == ["b", "a"]
    assert resources.close() == ["RuntimeError: a cleanup failed"] and order == ["b", "a"]
    assert resources.statistics()["closed"] is True
    with pytest.raises(RuntimeError):
        resources.register(b)


@pytest.mark.parametrize("mode", ["success", "construct", "callback", "cleanup", "combined"])
def test_shared_driver_only_completes_after_checked_cleanup(tmp_path, mode):
    output = tmp_path / "output"
    closed = []

    def cleanup():
        closed.append("provider")
        assert not (output / "result.json").exists()
        if mode in ("cleanup", "combined"):
            raise EvaluationError("Deliberate close failure")

    provider = SimpleNamespace(close_checked=cleanup, statistics={"successful_evaluations": 1})
    lmp = SimpleNamespace(
        get_mpi_comm=lambda: None,
        extract_setting=lambda name: 1,
        commands_list=lambda commands: None,
        command=lambda command: None,
        extract_global=lambda name: 0,
        close=lambda: closed.append("lammps"),
    )
    coupling = SimpleNamespace(statistics=lambda: {})

    def run(steps):
        if mode in ("callback", "combined"):
            raise EvaluationError("Original SCF failure")

    coupling.run = run

    def factory(*args, provider_resources):
        provider_resources.register(provider)
        if mode == "construct":
            raise ValueError("Later constructor failed")
        return coupling

    config = {
        "provider": PROVIDER,
        "periodic": False,
        "data_file": "unused",
        "timestep_fs": 0.1,
        "steps": 0,
        "sample_interval": 1,
    }
    with (
        patch.dict("sys.modules", {"lammps": SimpleNamespace(lammps=lambda **kwargs: lmp)}),
        patch("qmdpa.cli._frame", return_value=accepted_frame()),
        patch("qmdpa.cli._attach_state"),
    ):
        if mode == "success":
            cli._run(
                "input",
                output,
                restart=None,
                config_loader=lambda _: config,
                setup_pair=lambda *args: None,
                make_coupling=factory,
            )
        else:
            with pytest.raises((ValueError, EvaluationError, RuntimeError)):
                cli._run(
                    "input",
                    output,
                    restart=None,
                    config_loader=lambda _: config,
                    setup_pair=lambda *args: None,
                    make_coupling=factory,
                )
    assert closed == ["provider", "lammps"]
    if mode == "success":
        assert json.loads((output / "result.json").read_text())["metrics"]["qm_provider"]["closed"]
    else:
        assert not (output / "result.json").exists()
        failure = json.loads((output / "failure.json").read_text())
        if mode in ("callback", "combined"):
            assert "Original SCF failure" in failure["error"]
        if mode in ("cleanup", "combined"):
            assert failure["metrics"]["qm_provider"]["cleanup_errors"]


def test_mechanical_constructor_registers_before_later_dpa_failure(tmp_path):
    resources = ProviderResources(tmp_path / "runtime")
    provider = SimpleNamespace(close_checked=Mock(), statistics={})
    config = {"provider": PROVIDER, "qm": {}, "model": "unused"}
    with (
        patch("qmdpa.cli.make_qm_provider", return_value=provider),
        patch("qmdpa.cli.DeepMDBackend", side_effect=RuntimeError("DPA setup failed")),
    ):
        with pytest.raises(RuntimeError, match="DPA setup failed"):
            cli._mechanical_coupling(None, None, config, provider_resources=resources)
    assert resources.close() == []
    provider.close_checked.assert_called_once()


def test_unified_factory_transmits_resources_only_to_abacus_mechanical(tmp_path):
    resources = ProviderResources(tmp_path)
    config = {"hamiltonian": dispatch.MECHANICAL, "provider": PROVIDER}
    with patch("qmdpa.cli._mechanical_coupling") as factory:
        dispatch._make_coupling(None, None, config, provider_resources=resources)
    factory.assert_called_once_with(None, None, config, provider_resources=resources)
    config["provider"] = {**PROVIDER, "engine": "pyscf"}
    with pytest.raises(ValueError):
        dispatch._make_coupling(None, None, config, provider_resources=resources)
