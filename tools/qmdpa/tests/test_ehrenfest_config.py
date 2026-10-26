"""Coherent-state route admission without importing an electronic backend."""
import copy
import json

import numpy as np
import pytest

from qmdpa.casscf_pair import (
    _RUNTIME_MODULES,
    NAC_SOURCE_SHA256,
    CASSCFPairReference,
    CASSCFPairSettings,
    CASWavefunction,
)
from qmdpa.config import file_digest
from qmdpa.core import Geometry
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.ehrenfest_mechanical import HAMILTONIAN


@pytest.fixture
def configuration(tmp_path):
    model, data, ref = (tmp_path / name for name in ("fixture.pt", "fixture.data", "pair.json"))
    model.write_bytes(b"Model identity fixture; no inference in configuration tests")
    data.write_bytes(b"Native data identity fixture")
    # An abstract orthonormal singlet reference suffices for schema tests.
    # The real provider independently verifies the AO metric and source runtime.
    geometry = Geometry((1, 2), ("Li", "H"), [[0., 0, 0], [1.5, 0, 0]])
    hashes = {name: "0"*64 for name in _RUNTIME_MODULES}
    hashes["nac.sacasscf"] = NAC_SOURCE_SHA256
    reference = CASSCFPairReference(
        CASSCFPairSettings(), CASWavefunction(geometry, 1, np.eye(3), [[[1., 0], [0, 0]], [[0, 0], [0, 1.]]]),
        json.dumps({"pyscf_version": "2.14.0", "source_sha256": hashes}))
    ref.write_text(json.dumps(reference.record()))
    cfg = {"schema_version": 1, "hamiltonian": HAMILTONIAN,
           "provider": {"engine": "pyscf", "device": "cpu", "state": {"method": "sa-casscf22"}},
           "periodic": False, "model": str(model), "model_sha256": file_digest(model),
           "data_file": str(data), "type_symbols": ["H", "Li"],
           "full_region": {"atom_ids": [1, 2, 9], "symbols": ["Li", "H", "H"], "molecules": [[1, 2], [9]]},
           "qm_region": {"atom_ids": [1, 2], "symbols": ["Li", "H"], "molecules": [[1, 2]]},
           "qm": {}, "full_charge_spin": [0, 1], "region_charge_spin": [0, 1],
           "pair_reference": {"path": str(ref), "sha256": file_digest(ref)},
           "initial_amplitudes": [[1., 0.], [0., 0.]],
           "inference_profile": "torch-raw-deterministic-v1", "timestep_fs": 0.01, "steps": 2}
    path = tmp_path / "run.json"
    return path, cfg


def load(fixture, edit=None):
    path, original = fixture
    cfg = copy.deepcopy(original)
    if edit:
        edit(cfg)
    path.write_text(json.dumps(cfg))
    return load_run_config(path)


def test_unified_route_has_its_own_model_and_complete_total(configuration):
    cfg = load(configuration)
    description = describe_run(cfg)
    assert description["coupling"] == {"surface": "two-state", "embedding": "mechanical", "me_model": "ehrenfest"}
    assert description["assembly"]["native_pair"] == "zero"
    assert description["electronic_dynamics"]["algorithm"] == "two-state-ehrenfest"
    assert cfg["qm"]["mc_grad_tol"] == 1e-9
    selected = load(configuration, lambda c: c.update(coupling=description["coupling"]))
    assert selected["physics_sha256"] == cfg["physics_sha256"]
    continued = load(configuration, lambda c: c.update(steps=7, sample_interval=2))
    assert continued["physics_sha256"] == cfg["physics_sha256"]
    changed = load(configuration, lambda c: c.update(initial_amplitudes=[[0., 0.], [1., 0.]]))
    assert changed["physics_sha256"] != cfg["physics_sha256"]


@pytest.mark.parametrize("edit", [
    lambda c: c.update(periodic=True),
    lambda c: c["provider"].update(device="cuda"),
    lambda c: c["provider"]["state"].update(method="rhf-tda"),
    lambda c: c.update(full_charge_spin=[1, 2]),
    lambda c: c.update(initial_amplitudes=[[1, 0], [1, 0]]),
    lambda c: c.update(inference_profile="uncontrolled"),
    lambda c: c["full_region"].update(molecules=[[1, 9], [2]]),
    lambda c: c["full_region"]["symbols"].__setitem__(0, "H"),
    lambda c: c["qm"].update(mc_tol=1e-5),
    lambda c: c["pair_reference"].update(sha256="0"*64),
    lambda c: c.update(bias={}),
    lambda c: c.update(sampling={}),
    lambda c: c.update(thermostat={}),
    lambda c: c.update(execution={}),
])
def test_unsupported_combinations_and_inconsistent_identity_fail(configuration, edit):
    with pytest.raises((ValueError, RuntimeError)):
        load(configuration, edit)
