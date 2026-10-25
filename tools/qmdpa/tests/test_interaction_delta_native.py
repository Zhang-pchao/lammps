# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit corrected-model selection, complete fragments and native ownership."""

import builtins
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_dispatch import save

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import NativeDriver, config, load
from test_pbe0_gap_native import converted

from qmdpa import dispatch, gap_anchor_cli
from qmdpa.interaction_delta import CorrectedCalculator
from qmdpa.interaction_delta_config import CORRECTED_GAP_HAMILTONIAN, INTERACTION_MODEL

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def select(config):
    pinned = converted(config)
    config["hamiltonian"] = CORRECTED_GAP_HAMILTONIAN
    config["interaction_delta"] = {
        "model": INTERACTION_MODEL,
        "ground_method": "matched-pbe0",
        "core_region": copy.deepcopy(config["qm_region"]),
        "fragment_region": copy.deepcopy(config["environment_region"]),
        "charge_spin": {name: [0, 1] for name in ("pair", "core", "fragment")},
    }
    return pinned


def test_distinct_identity_complete_total_and_selector(tmp_path, config):
    converted(config)
    original = load(tmp_path, config)
    # select() converts only once; use a saved converted request below.
    config["hamiltonian"] = CORRECTED_GAP_HAMILTONIAN
    config["interaction_delta"] = {
        "model": INTERACTION_MODEL,
        "ground_method": "matched-pbe0",
        "core_region": copy.deepcopy(config["qm_region"]),
        "fragment_region": copy.deepcopy(config["environment_region"]),
        "charge_spin": {k: [0, 1] for k in ("pair", "core", "fragment")},
    }
    corrected = load(tmp_path, config)
    assert corrected["physics_sha256"] != original["physics_sha256"]
    assert corrected["assembly"]["pair_style"] == "zero"
    assert corrected["assembly"]["external_quantity"] == "total-energy-and-all-forces"
    assert corrected["assembly"]["interaction_delta"] == INTERACTION_MODEL
    chosen = copy.deepcopy(config)
    chosen.pop("hamiltonian")
    chosen["coupling"] = {
        "surface": "selected",
        "embedding": "electrostatic",
        "ee_model": "gap-anchor-cross-delta",
    }
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        normalized = dispatch.load_run_config(save(tmp_path, chosen))
    assert normalized == corrected
    description = dispatch.describe_run(corrected)
    assert description["interaction_delta"] == corrected["interaction_delta"]
    assert description["energy_contract"]["total_terms"][-1] == {
        "name": "cross_interaction_delta",
        "coefficient": 1,
        "force_coordinates": "CB",
    }
    config.update(steps=2, sample_interval=1)
    assert load(tmp_path, config)["physics_sha256"] == corrected["physics_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "unannounced",
        "model",
        "method",
        "charge",
        "boolean",
        "unknown",
        "outside",
        "symbol",
        "split-molecule",
        "invented-molecules",
    ],
)
def test_invalid_correction_rejects_before_backend(tmp_path, config, change):
    select(config)
    d = config["interaction_delta"]
    if change == "missing":
        config.pop("interaction_delta")
    elif change == "unannounced":
        config["hamiltonian"] = "finite-electrostatic-gap-anchor"
    elif change == "model":
        d["model"] = "add-dispersion"
    elif change == "method":
        d["ground_method"] = "hf"
    elif change == "charge":
        d["charge_spin"]["core"] = [1, 2]
    elif change == "boolean":
        d["charge_spin"]["core"] = [False, 1]
    elif change == "unknown":
        d["fitted_scale"] = 1.0
    elif change == "outside":
        d["core_region"] = copy.deepcopy(config["environment_region"])
    elif change == "symbol":
        d["core_region"]["symbols"][0] = "C"
    elif change == "split-molecule":
        d["core_region"] = {
            "atom_ids": d["core_region"]["atom_ids"][:1],
            "symbols": d["core_region"]["symbols"][:1],
            "molecules": [d["core_region"]["atom_ids"][:1]],
        }
    else:
        d["core_region"]["molecules"] = [[i] for i in d["core_region"]["atom_ids"]]
    with patch("qmdpa.backends.DeepMDBackend") as low, pytest.raises(ValueError):
        load(tmp_path, config)
    low.assert_not_called()


def test_pbe0_only_and_no_ground_native_fallback(tmp_path, config):
    original = copy.deepcopy(config)
    select(config)
    config["provider"] = original["provider"]
    config["qm"] = original["qm"]
    with pytest.raises(ValueError, match="PBE0"):
        load(tmp_path, config)
    choice = {
        "coupling": {
            "surface": "ground",
            "embedding": "electrostatic",
            "ee_model": "gap-anchor-cross-delta",
        }
    }
    with pytest.raises(ValueError, match="selected PBE0"):
        dispatch._normalize_selection(choice)


def test_no_quantum_import_during_validation(tmp_path, config):
    select(config)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf")):
            raise AssertionError("Unexpected engine import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        assert load(tmp_path, config)["hamiltonian"] == CORRECTED_GAP_HAMILTONIAN


def test_native_total_wrapper_preserves_state_observer(config):
    pinned = select(config)
    backend = SimpleNamespace(reference=pinned, charges=pinned.charges)
    baseline = SimpleNamespace(embedded_selected_backend=backend, evaluate=lambda g: None)
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.gap_anchor.make_pbe0_gap_anchor", return_value=baseline),
        patch("qmdpa.pbe0_tda.PySCFPBE0GroundBackend") as high,
    ):
        bridge = gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    assert isinstance(bridge.calculator, CorrectedCalculator)
    assert bridge.calculator.baseline is baseline
    assert bridge.state_observer is baseline.embedded_selected_backend
    assert bridge.state_observer.reference.sha256 == pinned.sha256
    assert bridge.calculator.correction.core_atom_ids == tuple(config["qm_region"]["atom_ids"])
    high.assert_called_once_with(pinned.settings)
    assert low.call_count == 2


def test_direct_factory_cannot_omit_correction(config):
    select(config)
    config.pop("interaction_delta")
    with patch("qmdpa.backends.DeepMDBackend") as low, pytest.raises(ValueError, match="together"):
        gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    low.assert_not_called()
