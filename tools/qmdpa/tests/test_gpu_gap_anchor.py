# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU anchor keeps matched terms, full-total assembly and runtime identity."""

import builtins
import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_excited_config import reference as cpu_reference

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import NativeDriver, config, load
from test_gpu_selected_config import spec
from test_gpu_selected_ee_config import reference
from test_state_inputs import file_provider

from qmdpa import dispatch, gap_anchor_cli
from qmdpa.core import EvaluationError
from qmdpa.fragment_response_config import FRAGMENT_GAP_HAMILTONIAN
from qmdpa.gap_anchor import make_gpu_rhf_gap_anchor
from qmdpa.gap_anchor_config import load_gap_anchor_config
from qmdpa.interaction_delta_config import CORRECTED_GAP_HAMILTONIAN

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def convert(config):
    ref = reference()
    config["qm"] = asdict(ref.settings)
    config["provider"] = {**spec(ref), "reuse_identical_geometry": True}
    return ref


def test_gpu_factory_constructs_all_matched_terms_from_one_reference():
    ref = reference()
    with (
        patch("qmdpa.gpu_backends.GPU4PySCFBackend") as ground,
        patch("qmdpa.gpu_backends.GPU4PySCFElectrostaticBackend") as embedded,
        patch("qmdpa.gpu_excited.GPU4PySCFTDAElectrostaticBackend") as selected,
        patch("qmdpa.backends.PySCFBackend") as cpu,
    ):
        embedded.return_value.charges = selected.return_value.charges = ref.charges
        calc = make_gpu_rhf_gap_anchor(object(), object(), ref, reuse_identical_geometry=True)
        settings = {
            name: getattr(ref.settings, name)
            for name in (
                "xc",
                "basis",
                "charge",
                "multiplicity",
                "conv_tol",
                "conv_tol_grad",
                "max_cycle",
                "threads",
            )
        }
        ground.assert_called_once_with(**settings)
        embedded.assert_called_once_with(ref.charges, **settings)
        selected.assert_called_once_with(
            ref.settings, ref, ref.charges, reuse_identical_geometry=True
        )
        cpu.assert_not_called()
        assert calc.qm_atom_ids == ref.qm_geometry.atom_ids
        assert calc.environment_atom_ids == ref.environment_geometry.atom_ids


@pytest.mark.parametrize("ref", [cpu_reference(embedded=True), reference(False)])
def test_gpu_factory_rejects_unmatched_reference_before_engine(ref):
    with patch("qmdpa.gpu_backends.GPU4PySCFBackend") as ground:
        with pytest.raises(ValueError, match="embedded GPU"):
            make_gpu_rhf_gap_anchor(object(), object(), ref)
    ground.assert_not_called()


def test_plain_gpu_config_is_lazy_total_and_file_equivalent(tmp_path, config):
    cpu = load(tmp_path, config)
    convert(config)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "cupy", "gpu4pyscf")):
            raise AssertionError("Unexpected optional import: " + name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        resolved = load(tmp_path, config)
    assert resolved["assembly"]["ground_method"] == "rhf"
    assert resolved["assembly"]["pair_style"] == "zero"
    assert resolved["assembly"]["external_quantity"] == "total-energy-and-all-forces"
    assert resolved["physics_sha256"] != cpu["physics_sha256"]
    assert "gpu4pyscf-cuda12x" in resolved["software"]["packages"]
    config["provider"] = file_provider(tmp_path, config["provider"])
    assert load(tmp_path, config) == resolved
    selected = deepcopy(config)
    selected.pop("hamiltonian")
    selected["coupling"] = {
        "surface": "selected",
        "embedding": "electrostatic",
        "ee_model": "gap-anchor",
    }
    path = tmp_path / "selector.json"
    path.write_text(json.dumps(selected))
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        assert dispatch.load_run_config(path) == resolved


@pytest.mark.parametrize("package", ["numpy", "gpu4pyscf-cuda12x", "cupy-cuda12x", "torch"])
def test_gpu_runtime_change_alters_anchor_restart_identity(tmp_path, config, package):
    convert(config)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    with patch("qmdpa.config.importlib.metadata.version", return_value="original"):
        first = load_gap_anchor_config(path)
    with patch(
        "qmdpa.config.importlib.metadata.version",
        side_effect=lambda n: "changed" if n == package else "original",
    ):
        second = load_gap_anchor_config(path)
    assert first["physics_sha256"] != second["physics_sha256"]


@pytest.mark.parametrize("kind", ["cross", "fragment", "ground-reuse", "charge"])
def test_gpu_expansion_is_rejected_before_allocating_backends(tmp_path, config, kind):
    convert(config)
    if kind == "cross":
        config["hamiltonian"] = CORRECTED_GAP_HAMILTONIAN
        config["interaction_delta"] = {}
    elif kind == "fragment":
        config["hamiltonian"] = FRAGMENT_GAP_HAMILTONIAN
        config["fragment_response"] = {}
    elif kind == "charge":
        config["qm"]["charge"] = 2
    else:
        config["reuse_ground_terms"] = True
    with patch("qmdpa.backends.DeepMDBackend") as low, pytest.raises(ValueError):
        load(tmp_path, config)
    low.assert_not_called()


def test_native_gpu_route_preserves_reference_and_live_field(config):
    ref = convert(config)
    backend = SimpleNamespace(reference=ref, charges=ref.charges)
    calc = SimpleNamespace(embedded_selected_backend=backend)
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.gap_anchor.make_gpu_rhf_gap_anchor", return_value=calc) as gpu,
        patch("qmdpa.gap_anchor.make_rhf_gap_anchor") as cpu,
    ):
        bridge = gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    assert low.call_count == 2
    cpu.assert_not_called()
    assert gpu.call_args.args[2].to_record() == ref.to_record()
    assert gpu.call_args.kwargs == {"reuse_identical_geometry": True}
    assert bridge.state_observer.reference.sha256 == ref.sha256
    assert bridge.state_observer.charges is ref.charges


def test_failed_gpu_factory_never_falls_back_to_cpu(config):
    convert(config)
    with (
        patch("qmdpa.backends.DeepMDBackend"),
        patch(
            "qmdpa.gap_anchor.make_gpu_rhf_gap_anchor",
            side_effect=EvaluationError("GPU anchor failure"),
        ),
        patch("qmdpa.gap_anchor.make_rhf_gap_anchor") as cpu,
    ):
        with pytest.raises(RuntimeError, match="GPU anchor failure"):
            gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    cpu.assert_not_called()


def test_direct_gpu_fragment_dispatch_is_rejected_before_low_backend(config):
    convert(config)
    config["hamiltonian"] = CORRECTED_GAP_HAMILTONIAN
    config["interaction_delta"] = {}
    with patch("qmdpa.backends.DeepMDBackend") as low, pytest.raises(ValueError, match="plain RHF"):
        gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    low.assert_not_called()
