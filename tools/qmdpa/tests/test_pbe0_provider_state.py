# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""PBE0 dispatch and frame guards preserve a distinct, explicit state contract."""

import json
from dataclasses import asdict
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, load
from test_pbe0_tda import pbe0_reference
from test_state_records import Backend, diagnostics, fixture, frame

from qmdpa import excited, pbe0_tda
from qmdpa.config import validate_qm
from qmdpa.core import EvaluationError
from qmdpa.gap_anchor import make_rhf_gap_anchor
from qmdpa.providers import make_qm_provider, validate_provider
from qmdpa.state_records import (
    RecordingTDAProvider,
    configured_reference,
    validate_frame_state,
    validate_state_record,
)

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case']


def request(reference):
    return {
        "engine": "pyscf",
        "device": "cpu",
        "reuse_identical_geometry": True,
        "state": {"method": "pbe0-tda", "reference": reference.to_record()},
    }


def convert_case(case):
    embedded, old, config = case
    pinned = pbe0_tda.PBE0TDAReference(
        old.qm_geometry,
        pbe0_tda.PBE0TDASettings(),
        old.transition_density,
        old.seed_root,
        old.excitation_energy_ev,
        old.reference_gap_ev,
        old.label,
        json.dumps(pbe0_reference().runtime),
        old.environment_geometry,
        old.charges,
    )
    config["provider"] = request(pinned)
    config["qm"] = asdict(pinned.settings)
    return embedded, pinned, config


def pbe0_diagnostics(pinned):
    # The independently declared expected response has no RHF refinement fields.
    rhf, _, _, _ = fixture()
    result = diagnostics(rhf)
    for key in list(result):
        if key.startswith("cphf_"):
            del result[key]
    _, chosen = excited.select_tda_state([1.0, 0.0, 0.0], [2.0, 3.0, 4.0], pinned.settings)
    result.update(chosen)
    result.update(
        reference_sha256=pinned.sha256,
        method="cpu-pbe0-tda-singlet",
        energy_definition="E_PBE0+omega",
        tda_matrix_residuals_hartree=[1e-14] * 3,
        tda_eigensystem_residuals_hartree=[1e-14] * 3,
        tda_eigensystem_residual_max_hartree=1e-14,
        xc="pbe0",
        grid_level=4,
        grid_prune="nwchem_prune",
        grid_points=100,
        rks_grid_response=True,
        density_fitting=False,
        nlc=False,
        cphf_residual_max=1e-9,
        cphf_solver="official-unrefined-residual-checked",
        cphf_refinement_count=0,
        grid_response_model="restricted-pbe0-tda-xc-lagrangian-v1",
        quadrature_response_points=96,
        weight_response=True,
        coordinate_response=True,
        grid_response_force_assembly=True,
        grid_response_gradient_hartree_bohr=[[0.01, 0.0, 0.0]] * len(pinned.qm_geometry.atom_ids),
    )
    return result


def observer(embedded=False):
    pinned = pbe0_reference(embedded)
    backend = Backend.__new__(Backend)
    backend.reference = pinned
    backend.calls = []
    backend.failure = False
    backend.diagnostic = pbe0_diagnostics(pinned)
    return pinned, backend, RecordingTDAProvider(backend)


def test_both_configuration_routes_preserve_method_settings_and_physics_identity(tmp_path, case):
    legacy = load(tmp_path, case)
    converted = convert_case(case)
    embedded, pinned, config = converted
    result = load(tmp_path, converted)
    assert result["qm"] == asdict(pinned.settings)
    assert result["provider"] == request(pinned)
    assert result["provider_contract"]["energy"] == "total-pbe0-plus-tda-excitation"
    assert configured_reference(result).to_record() == pinned.to_record()
    assert result["physics_sha256"] != legacy["physics_sha256"]
    baseline = load(tmp_path, case)  # Same explicit PBE0 input has a stable identity.
    assert result["physics_sha256"] == baseline["physics_sha256"]
    assert result["provider_contract"]["force_coordinates"] == (
        ["qm", "environment"] if embedded else ["qm"]
    )


@pytest.mark.parametrize("name", ["tda_solver", "grid_level", "grid_prune", "grid_response"])
def test_implicit_pbe0_grid_or_solver_is_rejected_before_runtime(case, name):
    embedded, pinned, config = convert_case(case)
    del config["qm"][name]
    with (
        patch("qmdpa.pbe0_tda._load_runtime") as runtime,
        pytest.raises(ValueError, match="explicit"),
    ):
        make_qm_provider(config["qm"], provider=config["provider"], charges=pinned.charges)
    runtime.assert_not_called()


@pytest.mark.parametrize("changed", ["method", "reference", "grid", "source", "settings"])
def test_method_and_runtime_swaps_cannot_inherit_qualification(case, changed):
    embedded, old, config = case
    _, pinned, config = convert_case(case)
    if changed == "method":
        config["provider"]["state"]["method"] = "rhf-tda"
    elif changed == "reference":
        config["provider"]["state"]["reference"] = old.to_record()
    elif changed == "grid":
        del config["provider"]["state"]["reference"]["settings"]["grid_response"]
    elif changed == "source":
        config["provider"]["state"]["reference"]["runtime"] = old.runtime
    else:
        config["qm"]["threads"] = 2
    with (
        patch("qmdpa.pbe0_tda._load_runtime") as runtime,
        pytest.raises((ValueError, EvaluationError)),
    ):
        make_qm_provider(config["qm"], provider=config["provider"], charges=pinned.charges)
    runtime.assert_not_called()


def test_pbe0_factory_constructs_only_requested_method_without_evaluating(case):
    embedded, pinned, config = convert_case(case)
    target = "PySCFPBE0TDAElectrostaticBackend" if embedded else "PySCFPBE0TDABackend"
    with (
        patch("qmdpa.pbe0_tda." + target) as backend,
        patch("qmdpa.state_records.RecordingTDAProvider") as wrap,
    ):
        result = make_qm_provider(config["qm"], provider=config["provider"], charges=pinned.charges)
    assert result is wrap.return_value
    assert backend.call_args.args[0] == pinned.settings
    assert backend.call_args.args[1].to_record() == pinned.to_record()
    assert backend.call_args.kwargs == {"reuse_identical_geometry": True}
    backend.return_value.evaluate_surface.assert_not_called()


@pytest.mark.parametrize("embedded", [False, True])
def test_pbe0_same_evaluation_state_records_reload_and_clear_on_failure(embedded):
    pinned, backend, recording = observer(embedded)
    recording.evaluate(pinned.qm_geometry, pinned.environment_geometry)
    snapshot = frame(pinned)
    snapshot["electronic_state"] = recording.accepted_record(snapshot)
    config = {"periodic": False, "provider": request(pinned)}
    validate_frame_state(snapshot, config)
    assert snapshot["electronic_state"]["method"] == "pbe0-tda"
    assert len(backend.calls) == 1
    backend.failure = True
    with pytest.raises(EvaluationError):
        recording.evaluate(pinned.qm_geometry, pinned.environment_geometry)
    with pytest.raises(EvaluationError, match="No successful"):
        recording.accepted_record(snapshot)


@pytest.mark.parametrize(
    "key,value",
    [
        ("method", "cpu-rhf-tda-singlet"),
        ("energy_definition", "E_RHF+omega"),
        ("grid_response_force_assembly", False),
        ("rks_grid_response", False),
        ("grid_response_model", "unknown"),
        ("weight_response", 1),
        ("coordinate_response", False),
        ("grid_level", 5),
        ("grid_prune", "none"),
        ("density_fitting", True),
        ("nlc", True),
        ("cphf_refinement_count", True),
        ("cphf_solver", "normalized-defect-correction"),
        ("cphf_residual_max", 2e-7),
        ("density_trace", 0.0),
        ("grid_points", True),
        ("quadrature_response_points", 0),
        ("grid_response_gradient_hartree_bohr", [[0.0, 0.0, 0.0]]),
        ("grid_response_gradient_hartree_bohr", [[float("nan")] * 3] * 2),
        ("cphf_refinement_method", "normalized-defect-correction"),
    ],
)
def test_foreign_or_incomplete_response_cannot_be_written(key, value):
    pinned, backend, recording = observer()
    backend.diagnostic[key] = value
    with pytest.raises((ValueError, EvaluationError)):
        recording.evaluate(pinned.qm_geometry)
    with pytest.raises(EvaluationError, match="No successful"):
        recording.accepted_record(frame(pinned))


def test_pbe0_record_cannot_be_relabelled_or_used_as_rhf():
    pinned, backend, recording = observer()
    recording.evaluate(pinned.qm_geometry)
    record = recording.accepted_record(frame(pinned))
    record["method"] = "rhf-tda"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_state_record(record, pinned)
    with pytest.raises(ValueError, match="frozen embedded"):
        make_rhf_gap_anchor(object(), object(), pbe0_reference(True))


def test_pbe0_registry_does_not_import_quantum_engine(case):
    import builtins

    _, pinned, _ = convert_case(case)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf")):
            raise AssertionError("Unexpected engine import: " + name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        assert validate_provider(
            request(pinned), electrostatic=pinned.charges is not None
        ) == request(pinned)
        assert validate_qm(
            asdict(pinned.settings),
            provider=request(pinned),
            electrostatic=pinned.charges is not None,
        ) == asdict(pinned.settings)
