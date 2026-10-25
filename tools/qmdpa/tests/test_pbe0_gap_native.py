# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Method-specific gap-anchor native dispatch, identity and failure boundaries."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_excited_config import reference

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import NativeDriver, config, load
from test_pbe0_provider_state import convert_case

from qmdpa import gap_anchor_cli
from qmdpa.core import EvaluationError
from qmdpa.state_records import configured_reference

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def converted(config):
    return convert_case((True, reference(embedded=True), config))[1]


def test_pbe0_uses_complete_total_and_distinct_physical_identity(tmp_path, config):
    legacy = load(tmp_path, config)
    pinned = converted(config)
    result = load(tmp_path, config)
    assert result["assembly"] == {
        "pair_style": "zero",
        "external_quantity": "total-energy-and-all-forces",
        "model_format": "pt2",
        "ground_method": "rks-xc-pbe0",
    }
    assert legacy["assembly"]["ground_method"] == "rks-xc-hf"
    assert result["physics_sha256"] != legacy["physics_sha256"]
    assert configured_reference(result).to_record() == pinned.to_record()
    assert result["provider_contract"]["force_coordinates"] == ["qm", "environment"]
    assert result["provider_contract"]["energy"] == "total-pbe0-plus-tda-excitation"
    config.update(steps=40, sample_interval=2)
    assert load(tmp_path, config)["physics_sha256"] == result["physics_sha256"]


@pytest.mark.parametrize("change", ["method", "reference", "grid", "response", "basis", "field"])
def test_unmatched_pbe0_inputs_fail_before_backend_allocation(tmp_path, config, change):
    old = copy.deepcopy(config["provider"]["state"]["reference"])
    converted(config)
    if change == "method":
        config["provider"]["state"]["method"] = "rhf-tda"
    elif change == "reference":
        config["provider"]["state"]["reference"] = old
    elif change == "grid":
        del config["qm"]["grid_level"]
    elif change == "response":
        config["qm"]["grid_response"] = False
    elif change == "basis":
        config["qm"]["basis"] = "sto-3g"
    else:
        config["embedding"]["source"] = "different field"
    with patch("qmdpa.backends.DeepMDBackend") as low, pytest.raises((ValueError, EvaluationError)):
        load(tmp_path, config)
    low.assert_not_called()


def test_pbe0_dispatch_keeps_matched_reference_and_live_field(config):
    pinned = converted(config)
    backend = SimpleNamespace(reference=pinned, charges=pinned.charges)
    calculator = SimpleNamespace(embedded_selected_backend=backend)
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.gap_anchor.make_pbe0_gap_anchor", return_value=calculator) as factory,
        patch("qmdpa.gap_anchor.make_rhf_gap_anchor") as foreign,
    ):
        bridge = gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    assert low.call_count == 2
    foreign.assert_not_called()
    args = factory.call_args
    assert args.args[2].to_record() == pinned.to_record()
    assert args.kwargs == {"reuse_identical_geometry": True}
    assert bridge.state_observer is calculator.embedded_selected_backend
    assert bridge.state_observer.charges is backend.charges
    assert bridge.state_observer.reference.sha256 == pinned.sha256


def test_failed_pbe0_builder_never_uses_rhf_fallback(config):
    converted(config)
    with (
        patch("qmdpa.backends.DeepMDBackend"),
        patch("qmdpa.gap_anchor.make_pbe0_gap_anchor", side_effect=EvaluationError("PBE0 failure")),
        patch("qmdpa.gap_anchor.make_rhf_gap_anchor") as foreign,
        pytest.raises(RuntimeError, match="PBE0 failure"),
    ):
        gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    foreign.assert_not_called()


def test_unsupported_direct_dispatch_rejects_before_low_backend(config):
    config["provider"]["state"]["method"] = "ground"
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        pytest.raises(RuntimeError, match="explicit"),
    ):
        gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    low.assert_not_called()


def test_pbe0_native_preflight_imports_no_quantum_engine(tmp_path, config):
    import builtins

    converted(config)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf")):
            raise AssertionError("Unexpected engine import: " + name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        assert load(tmp_path, config)["assembly"]["ground_method"] == "rks-xc-pbe0"
