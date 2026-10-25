# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU selected finite-field dispatch, preparation and complete state records."""

import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, load
from test_excited_config import reference as cpu_reference
from test_gap_anchor_native import config as anchor_case
from test_gap_anchor_native import load as load_anchor
from test_gpu_excited import record as gpu_runtime
from test_gpu_selected_config import gpu_record, spec
from test_state_inputs import file_provider, request

from qmdpa import dispatch, providers, state_inputs, state_records
from qmdpa.core import EvaluationError
from qmdpa.gpu_excited import _EMBEDDING_SOURCES, GPUTDAReference, GPUTDASettings

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case', 'anchor_case']


def reference(embedded=True):
    old = cpu_reference(embedded=embedded)
    runtime = gpu_runtime()
    if embedded:
        runtime["embedding_source_sha256"] = dict(_EMBEDDING_SOURCES)
    return GPUTDAReference(
        old.qm_geometry,
        GPUTDASettings(**{**asdict(old.settings), "device": "cuda"}),
        old.transition_density,
        old.seed_root,
        old.excitation_energy_ev,
        old.reference_gap_ev,
        old.label,
        json.dumps(runtime),
        old.environment_geometry,
        old.charges,
    )


def test_embedded_factory_preserves_full_field_and_energy_contract():
    ref = reference()
    normalized = providers.validate_provider(spec(ref), electrostatic=True)
    contract = providers.provider_contract(normalized, electrostatic=True)
    assert contract["force_coordinates"] == ["qm", "environment"]
    assert contract["energy"] == "total-rhf-plus-tda-excitation"
    assert contract["charge_model"] == "point" and contract["cpu_fallback"] == "forbidden"
    with patch("qmdpa.gpu_excited.GPU4PySCFTDAElectrostaticBackend", autospec=True) as backend:
        backend.return_value.reference = ref
        result = providers.make_qm_provider(
            asdict(ref.settings), provider=spec(ref), charges=ref.charges
        )
        assert result.backend is backend.return_value
        args = backend.call_args.args
        assert args[0] == ref.settings and args[1].to_record() == ref.to_record()
        assert args[2] is ref.charges


def test_gpu_fixed_modes_and_portable_files(tmp_path, case):
    embedded, _, data = case
    ref = reference(embedded)
    data["provider"] = spec(ref)
    data["qm"] = asdict(ref.settings)
    normalized = load(tmp_path, case)
    assert state_records.configured_reference(normalized).sha256 == ref.sha256
    data["provider"] = file_provider(tmp_path, data["provider"])
    assert load(tmp_path, case) == normalized
    coupling = {"surface": "selected", "embedding": "electrostatic" if embedded else "mechanical"}
    if embedded:
        coupling["ee_model"] = "additive"
    path = tmp_path / "unified.json"
    path.write_text(json.dumps({**data, "coupling": coupling}))
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        assert dispatch.load_run_config(path) == normalized
    if embedded:
        assert normalized["hamiltonian"] == "finite-electrostatic-additive"
        assert normalized["provider_contract"]["force_coordinates"] == ["qm", "environment"]
        data["embedding"]["source"] += " changed"
        with pytest.raises(ValueError, match="charges"):
            load(tmp_path, case)


def test_prepare_field_reaches_builder_and_serializes_identity(tmp_path):
    ref = reference()
    req = request(ref)
    req.update(engine="gpu4pyscf", device="cuda")
    resolved = state_inputs.validate_preparation(req)
    with patch("qmdpa.gpu_excited.build_gpu_tda_reference", return_value=ref) as build:
        assert state_inputs._build_reference(*resolved) is ref
        kwargs = build.call_args.kwargs
        assert kwargs["environment_geometry"].atom_ids == ref.environment_geometry.atom_ids
        assert kwargs["charges"].source == ref.charges.source
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req))
    out = tmp_path / "prepared"
    with patch("qmdpa.state_inputs._build_reference", return_value=ref):
        state_inputs.prepare_state(path, out)
    inline = json.loads((out / "provider-inline.json").read_text())
    assert inline == spec(ref)
    contract = json.loads((out / "result.json").read_text())["provider_contract"]
    assert contract["force_coordinates"] == ["qm", "environment"]
    declared = json.loads((out / "provider-file.json").read_text())
    assert state_inputs.resolve_provider_reference(declared, out / "config.json") == inline
    req["charges"]["radii_angstrom"] = [0.5] * len(ref.charges.atom_ids)
    with pytest.raises(ValueError, match="point charges"):
        state_inputs.validate_preparation(req)


def embedded_record():
    ref = reference()
    ref = replace(ref, settings=replace(ref.settings, nstates=2))
    _, record = gpu_record()
    record["reference_sha256"] = ref.sha256
    record["geometry_sha256"] = state_records.geometry_digest(
        ref.qm_geometry, ref.environment_geometry
    )
    d = record["diagnostics"]
    d.update(
        reference_sha256=ref.sha256,
        density_trace=2.0,
        density_trace_error=0.0,
        density_trace_tolerance=ref.settings.density_trace_tol,
        relaxed_density="official-total-rhf-tda",
        density_export="base-zero-response-bookkeeping",
    )
    return ref, record


def test_complete_embedded_gpu_state_record():
    ref, record = embedded_record()
    state_records.validate_state_record(record, ref)


@pytest.mark.parametrize(
    "key,value",
    [
        ("density_trace", -1.0),
        ("density_trace_error", 1e-3),
        ("density_trace_error", -1.0),
        ("density_trace_tolerance", 1e-3),
        ("relaxed_density", "ground-only"),
        ("density_export", "unverified"),
        ("density_trace", None),
    ],
)
def test_incomplete_environment_density_rejected(key, value):
    ref, record = embedded_record()
    state_records.validate_state_record(record, ref)
    record["diagnostics"][key] = value
    with pytest.raises((ValueError, EvaluationError)):
        state_records.validate_state_record(record, ref)


def test_gpu_fragment_correction_still_requires_cpu(tmp_path, anchor_case):
    from qmdpa.interaction_delta_config import CORRECTED_GAP_HAMILTONIAN

    ref = reference()
    anchor_case["provider"] = spec(ref)
    anchor_case["qm"] = asdict(ref.settings)
    anchor_case["hamiltonian"] = CORRECTED_GAP_HAMILTONIAN
    anchor_case["interaction_delta"] = {}
    with pytest.raises(ValueError, match="matched CPU"):
        load_anchor(tmp_path, anchor_case)
