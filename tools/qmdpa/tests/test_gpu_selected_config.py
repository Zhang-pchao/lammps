# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Selected GPU preparation, fixed ME dispatch and state identity."""

import builtins
import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, load
from test_gpu_excited import reference as api_reference
from test_state_inputs import file_provider, request
from test_state_records import diagnostics

from qmdpa import dispatch, providers, state_inputs, state_records
from qmdpa.core import EvaluationError

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case']


def reference():
    ref = api_reference()
    return replace(ref, settings=replace(ref.settings, tda_solver="dense"))


def spec(ref):
    return {
        "engine": "gpu4pyscf",
        "device": "cuda",
        "state": {"method": "rhf-tda", "reference": ref.to_record()},
    }


def test_lazy_gpu_reference_selection_and_factory():
    ref = reference()
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.startswith(("cupy", "gpu4pyscf", "pyscf")):
            raise AssertionError("Unexpected scientific runtime import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=checked):
        assert providers.validate_provider(spec(ref)) == spec(ref)
        contract = providers.provider_contract(spec(ref))
        assert contract["device"] == "cuda" and contract["force_coordinates"] == ["qm"]
        assert contract["energy"] == "total-rhf-plus-tda-excitation"
        assert providers.reference_engine(ref) == "gpu4pyscf"
        assert providers.reference_method(ref) == "rhf-tda"
    with patch("qmdpa.gpu_excited.GPU4PySCFTDABackend", autospec=True) as backend:
        backend.return_value.reference = ref
        result = providers.make_qm_provider(asdict(ref.settings), provider=spec(ref))
        assert result.backend is backend.return_value
        assert backend.call_count == 1 and backend.call_args.kwargs == {}
        assert backend.call_args.args[0] == ref.settings
        assert backend.call_args.args[1].to_record() == ref.to_record()


@pytest.mark.parametrize("change", ["device", "engine", "method", "iterative", "ee"])
def test_unsupported_request_never_enters_gpu(change):
    ref = reference()
    data = spec(ref)
    if change == "device":
        data["device"] = "cpu"
    elif change == "engine":
        data["engine"] = "pyscf"
    elif change == "method":
        data["state"]["method"] = "pbe0-tda"
    elif change == "iterative":
        data = spec(replace(ref, settings=replace(ref.settings, tda_solver="iterative")))
    with patch("qmdpa.gpu_excited.GPU4PySCFTDABackend") as backend:
        with pytest.raises(ValueError):
            providers.validate_provider(data, electrostatic=change == "ee")
    backend.assert_not_called()


def test_gpu_me_config_and_hashed_files_match(tmp_path, case):
    embedded, old, data = case
    if embedded:
        with pytest.raises(ValueError):
            providers.validate_provider(spec(reference()), electrostatic=True)
        return
    ref = replace(reference(), qm_geometry=old.qm_geometry)
    data["provider"] = spec(ref)
    data["qm"] = asdict(ref.settings)
    normalized = load(tmp_path, case)
    assert normalized["provider"]["engine"] == "gpu4pyscf"
    assert state_records.configured_reference(normalized).sha256 == ref.sha256
    data["provider"] = file_provider(tmp_path, data["provider"])
    assert load(tmp_path, case) == normalized
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        path = tmp_path / "configuration.json"
        path.write_text(
            json.dumps({**data, "coupling": {"surface": "selected", "embedding": "mechanical"}})
        )
        selected = dispatch.load_run_config(path)
    assert selected == normalized
    data["periodic"] = True
    with pytest.raises(ValueError, match="nonperiodic"):
        load(tmp_path, case)


def test_gpu_prepare_state_retains_identity_and_runtime_packages(tmp_path):
    ref = reference()
    req = request(ref)
    req.update(engine="gpu4pyscf", device="cuda")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req))
    out = tmp_path / "out"
    with patch("qmdpa.state_inputs._build_reference", return_value=ref):
        state_inputs.prepare_state(path, out)
    inline = json.loads((out / "provider-inline.json").read_text())
    assert inline == spec(ref)
    declared = json.loads((out / "provider-file.json").read_text())
    assert state_inputs.resolve_provider_reference(declared, out / "config.json") == inline
    versions = json.loads((out / "manifest.json").read_text())["software"]["packages"]
    assert "gpu4pyscf-cuda12x" in versions
    req["device"] = "cpu"
    with pytest.raises(ValueError):
        state_inputs.validate_preparation(req)


def gpu_record():
    ref = reference()
    d = diagnostics(ref)
    d.pop("density_trace")
    d.update(
        method="gpu-rhf-tda-singlet",
        engine="gpu4pyscf",
        device="cuda",
        cpu_fallback=False,
        cphf_solver="official-gpu-normalized-defect-correction",
        cphf_refinement_count=d["cphf_refinement_iterations"],
        cphf_residual_max=d["cphf_residual_history"][-1],
    )
    record = {
        "schema_version": 1,
        "method": "rhf-tda",
        "reference_sha256": ref.sha256,
        "geometry_sha256": state_records.geometry_digest(ref.qm_geometry),
        "selected_root": 1,
        "provider_energy_ev": -8.0,
        "diagnostics": d,
    }
    return ref, record


def test_gpu_state_record_replays_without_inventing_relaxed_environment_density():
    ref, record = gpu_record()
    state_records.validate_state_record(record, ref)
    assert "density_trace" not in record["diagnostics"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("engine", "pyscf"),
        ("device", "cpu"),
        ("cpu_fallback", True),
        ("cphf_solver", "unchecked"),
        ("cphf_refinement_count", 0),
        ("cphf_residual_max", 1e-4),
        ("method", "cpu-rhf-tda-singlet"),
    ],
)
def test_foreign_or_bad_gpu_record_is_rejected(key, value):
    ref, record = gpu_record()
    record["diagnostics"][key] = value
    with pytest.raises((ValueError, EvaluationError)):
        state_records.validate_state_record(record, ref)
