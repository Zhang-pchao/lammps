# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit small PBE0 inputs preserve complete response and legacy identities."""

import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, load
from test_gap_anchor_native import config as anchor_case
from test_gap_anchor_native import load as load_anchor
from test_pbe0_provider_state import convert_case, pbe0_diagnostics
from test_pbe0_tda import pbe0_reference
from test_state_inputs import file_provider, request

from qmdpa import pbe0_tda as pbe0
from qmdpa import state_inputs, state_records
from qmdpa.core import EvaluationError

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case', 'anchor_case']


def small_reference(legacy):
    settings = replace(legacy.settings, basis="sto-3g", grid_level=1)
    runtime = {**legacy.runtime, "source_sha256": pbe0._source_hashes_for(settings)}
    return replace(legacy, settings=settings, runtime_json=json.dumps(runtime))


def test_explicit_development_pair_keeps_legacy_defaults_and_runtime_identity():
    legacy = pbe0_reference()
    record = legacy.to_record()
    assert pbe0.PBE0TDASettings() == legacy.settings
    assert pbe0._source_hashes_for(legacy.settings) == pbe0._SOURCE_SHA256
    assert pbe0.PBE0TDAReference.from_record(record).to_record() == record
    small = small_reference(legacy)
    assert small.sha256 != legacy.sha256
    assert small.settings.grid_response is True
    assert "gto/basis/sto-3g.dat" in small.runtime["source_sha256"]
    assert pbe0.PBE0TDAReference.from_record(small.to_record()).sha256 == small.sha256
    with pytest.raises(EvaluationError):
        replace(small, runtime_json=json.dumps(legacy.runtime))
    with pytest.raises(ValueError):
        replace(legacy, runtime_json=json.dumps({**legacy.runtime, "unknown": 1}))


@pytest.mark.parametrize(
    "change",
    [
        {"basis": "sto-3g"},
        {"grid_level": 1},
        {"basis": "sto-3g", "grid_level": 2},
        {"basis": "sto-3g", "grid_level": True},
        {"basis": "sto-3g", "grid_level": 1, "grid_response": False},
        {"basis": "sto-3g", "grid_level": 1, "response_residual_tol": 1e-6},
    ],
)
def test_no_implicit_pair_fallback_or_response_weakening(change):
    with pytest.raises((ValueError, TypeError)):
        pbe0.PBE0TDASettings(**change)


def test_small_pair_configuration_files_and_restart_identity(tmp_path, case):
    converted = convert_case(case)
    embedded, old, config = converted
    old_loaded = load(tmp_path, converted)
    new = small_reference(old)
    config["qm"] = asdict(new.settings)
    config["provider"]["state"]["reference"] = new.to_record()
    small = load(tmp_path, converted)
    assert small["physics_sha256"] != old_loaded["physics_sha256"]
    config["provider"] = file_provider(tmp_path, config["provider"])
    assert load(tmp_path, converted) == small
    assert small["provider_contract"]["force_coordinates"] == (
        ["qm", "environment"] if embedded else ["qm"]
    )


@pytest.mark.parametrize("embedded", [False, True])
def test_small_pair_preparation_keeps_requested_response(tmp_path, embedded):
    ref = small_reference(pbe0_reference(embedded))
    req = request(ref, "pbe0-tda")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req))
    out = tmp_path / "prepared"
    with patch("qmdpa.state_inputs._build_reference", return_value=ref):
        state_inputs.prepare_state(path, out)
    prepared = json.loads((out / "qm.json").read_text())
    assert prepared["basis"] == "sto-3g" and prepared["grid_level"] == 1
    assert prepared["grid_response"] is True
    record = json.loads((out / "reference.json").read_text())
    assert record == ref.to_record()


def test_small_pair_state_record_requires_its_own_grid():
    ref = small_reference(pbe0_reference(True))
    diagnostic = pbe0_diagnostics(ref)
    record = {
        "schema_version": 1,
        "method": "pbe0-tda",
        "reference_sha256": ref.sha256,
        "geometry_sha256": state_records.geometry_digest(ref.qm_geometry, ref.environment_geometry),
        "selected_root": 1,
        "provider_energy_ev": -8.0,
        "diagnostics": diagnostic,
    }
    with pytest.raises(ValueError):
        state_records.validate_state_record(record, ref)  # Fixture still declares grid4.
    diagnostic["grid_level"] = 1
    state_records.validate_state_record(record, ref)


def test_small_pair_matched_ground_and_plain_anchor_preserve_identity(tmp_path, anchor_case):
    from test_pbe0_gap_native import converted

    legacy = converted(anchor_case)
    before = load_anchor(tmp_path, anchor_case)
    ref = small_reference(legacy)
    settings = pbe0._pbe0_ground_settings(ref.settings)
    assert settings["basis"] == "sto-3g" and settings["grid_level"] == 1
    anchor_case["qm"] = asdict(ref.settings)
    anchor_case["provider"]["state"]["reference"] = ref.to_record()
    anchor_case["reuse_ground_terms"] = True
    loaded = load_anchor(tmp_path, anchor_case)
    assert loaded["assembly"]["pair_style"] == "zero"
    assert loaded["assembly"]["ground_method"] == "rks-xc-pbe0"
    assert loaded["physics_sha256"] != before["physics_sha256"]
    assert loaded["provider"]["state"]["reference"] == ref.to_record()


def test_small_pair_keeps_fragment_model_validation(tmp_path, anchor_case):
    from test_pbe0_gap_native import converted

    from qmdpa.fragment_ground import _ground_settings
    from qmdpa.interaction_delta_config import CORRECTED_GAP_HAMILTONIAN

    ref = small_reference(converted(anchor_case))
    anchor_case["qm"] = asdict(ref.settings)
    anchor_case["provider"]["state"]["reference"] = ref.to_record()
    anchor_case["hamiltonian"] = CORRECTED_GAP_HAMILTONIAN
    anchor_case["interaction_delta"] = {}
    with pytest.raises(ValueError, match="Interaction correction requires"):
        load_anchor(tmp_path, anchor_case)
    assert _ground_settings(asdict(ref.settings)) == ref.settings
