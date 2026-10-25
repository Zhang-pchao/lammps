# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU selection, early rejection, lazy factory and restart identity contracts."""

import builtins
from copy import deepcopy
from unittest.mock import patch

import pytest
import test_adaptive_config
import test_cli
import test_fragment_ground_native

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import config

from qmdpa import dispatch
from qmdpa.providers import make_qm_provider, provider_contract, validate_provider

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']

GPU = {"engine": "gpu4pyscf", "device": "cuda", "state": {"method": "ground"}}
FITTING = {"auxbasis": "def2-universal-jkfit"}


@pytest.fixture
def case():
    case = test_cli.ConfigTests()
    case.setUp()
    case.config["provider"] = deepcopy(GPU)
    case.config["qm"]["density_fit"] = deepcopy(FITTING)
    try:
        yield case
    finally:
        case.doCleanups()


def test_supported_request_is_detached_and_factory_uses_only_gpu():
    request = deepcopy(GPU)
    normalized = validate_provider(request)
    assert normalized == request
    normalized["state"]["method"] = "mutated"
    assert request == GPU
    contract = provider_contract(request)
    assert contract["engine"] == "gpu4pyscf" and contract["device"] == "cuda"
    assert contract["execution"] == "single-visible-cuda-device"
    assert contract["cpu_fallback"] == "forbidden"
    assert contract["force_coordinates"] == ["qm"]
    options = {"xc": "pbe", "density_fit": FITTING}
    with (
        patch("qmdpa.gpu_backends.GPU4PySCFBackend") as gpu,
        patch("qmdpa.backends.PySCFBackend") as cpu,
    ):
        assert make_qm_provider(options, provider=request) is gpu.return_value
        gpu.assert_called_once_with(**options)
        cpu.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"device": "cpu"},
        {"engine": "pyscf"},
        {"state": {"method": "rhf-tda"}},
        {"state": {"method": "pbe0-tda"}},
        {"reuse_identical_geometry": False},
        {"matrix_assembly": "operator"},
        {"state": {"method": "ground", "root": 0}},
    ],
)
def test_unsupported_combinations_never_construct_gpu(change):
    with patch("qmdpa.gpu_backends.GPU4PySCFBackend") as gpu, pytest.raises(ValueError):
        make_qm_provider({}, provider={**GPU, **change})
    gpu.assert_not_called()


def test_invalid_ee_charges_are_rejected_before_any_backend():
    with (
        patch("qmdpa.gpu_backends.GPU4PySCFBackend") as gpu,
        pytest.raises(TypeError, match="FixedEmbeddingCharges"),
    ):
        make_qm_provider({}, provider=GPU, charges=object())
    gpu.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"xc": "wb97m-v"},
        {"density_fit": None},
        {"grid_response": False},
        {"nlc_grid_level": 1},
        {"multiplicity": 2},
    ],
)
def test_configuration_rejects_unsupported_qm_settings(case, change):
    case.config["qm"].update(change)
    with patch("qmdpa.gpu_backends.GPU4PySCFBackend") as gpu, pytest.raises(ValueError):
        case.load()
    gpu.assert_not_called()


def test_periodic_ground_native_keeps_the_quantum_region_finite(case):
    case.config["periodic"] = True
    loaded = case.load()
    assert loaded["periodic"] is True
    assert loaded["provider_contract"]["qm_boundary"] == "isolated-molecule"
    assert loaded["provider_contract"]["environment"] == "none"


def test_lazy_validation_selector_equivalence_and_exact_assembly(case):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf", "cupy", "torch")):
            raise AssertionError("Unexpected engine import: " + name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=checked):
        old = case.load()
        case.config["coupling"] = {"surface": "ground", "embedding": "mechanical"}
        import json

        case.path.write_text(json.dumps(case.config))
        selected = dispatch.load_run_config(case.path)
    assert selected == old
    description = dispatch.describe_run(selected)
    assert description["energy_contract"]["native_baseline"] == "low_full"
    assert selected["provider"] == GPU


@pytest.mark.parametrize(
    "package",
    [
        "gpu4pyscf-cuda12x",
        "gpu4pyscf-libxc-cuda12x",
        "cupy-cuda12x",
        "cutensor-cu12",
        "torch",
        "mpi4py",
    ],
)
def test_gpu_dependency_changes_restart_identity(case, package):
    with patch("qmdpa.config.importlib.metadata.version", return_value="original"):
        baseline = case.load()
    with patch(
        "qmdpa.config.importlib.metadata.version",
        side_effect=lambda name: "changed" if name == package else "original",
    ):
        modified = case.load()
    assert baseline["physics_sha256"] != modified["physics_sha256"]
    assert baseline["software"]["packages"][package] == "original"


def test_device_auxbasis_and_budget_identity(case):
    gpu = case.load()
    case.config["steps"] += 1
    assert case.load()["physics_sha256"] == gpu["physics_sha256"]
    case.config["qm"]["density_fit"]["auxbasis"] = "weigend"
    assert case.load()["physics_sha256"] != gpu["physics_sha256"]
    case.config["qm"]["density_fit"] = deepcopy(FITTING)
    case.config.pop("provider")
    cpu = case.load()
    assert cpu["physics_sha256"] != gpu["physics_sha256"]
    assert set(cpu["software"]["packages"]) == {"numpy", "pyscf", "deepmd-kit"}


def test_adaptive_rejects_gpu_before_ignoring_provider(tmp_path):
    request = test_adaptive_config.document()
    request["provider"] = deepcopy(GPU)
    request["qm"]["density_fit"] = deepcopy(FITTING)
    with pytest.raises(ValueError, match="CPU PySCF"):
        test_adaptive_config.load(tmp_path, request)


def test_matched_fragments_reject_gpu_before_cpu_fallback(tmp_path, config):
    request = test_fragment_ground_native.ground_request(config)
    request["provider"] = deepcopy(GPU)
    request["qm"]["density_fit"] = deepcopy(FITTING)
    with pytest.raises(ValueError, match="CPU PySCF"):
        test_fragment_ground_native.load(tmp_path, request)
