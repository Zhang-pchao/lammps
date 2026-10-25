# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""PBE0 state sets retain separate methods, response and file identities."""

import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from test_excited import T2, charges, environment, qm, reference
from test_pbe0_development import small_reference
from test_pbe0_tda import install_solver, pbe0_reference
from test_state_inputs import request

from qmdpa import multistate
from qmdpa import pbe0_tda as pbe0
from qmdpa.core import EvaluationError
from qmdpa.state_inputs import prepare_states, resolve_provider_reference


def references(embedded=False, small=False):
    first = pbe0_reference(embedded)
    if small:
        first = small_reference(first)
    first = replace(first, label="first")
    return {"first": first, "second": replace(first, label="second", seed_root=2,
            transition_density=T2, excitation_energy_ev=3.0)}


@pytest.mark.parametrize("embedded", [False, True])
def test_preparation_uses_one_pbe0_solve_and_method_specific_references(monkeypatch, embedded):
    calls = install_solver(monkeypatch)
    kw = {"environment_geometry": environment(), "charges": charges()} if embedded else {}
    with patch("qmdpa.multistate._solve", side_effect=AssertionError("RHF solve requested")):
        refs = multistate.build_pbe0_tda_references(
            qm(), pbe0.PBE0TDASettings(), states={"first": 1, "second": 2}, **kw)
    assert len(calls) == 1
    for name, ref in refs.items():
        assert type(ref) is pbe0.PBE0TDAReference and ref.label == name
        assert pbe0.PBE0TDAReference.from_record(ref.to_record()).sha256 == ref.sha256
        assert ref.settings.grid_response is True


@pytest.mark.parametrize("embedded", [False, True])
def test_state_specific_pbe0_response_uses_the_same_common_solution(monkeypatch, embedded):
    calls = install_solver(monkeypatch)
    response = pbe0.PySCFPBE0TDABackend._surface_gradient
    seen = []
    def measured(self, solved, index, *args):
        seen.append((id(solved), index))
        return response(self, solved, index, *args)
    monkeypatch.setattr(pbe0.PySCFPBE0TDABackend, "_surface_gradient", measured)
    obj = multistate.PySCFPBE0TDAStateSet(references(embedded))
    result = obj.evaluate_surfaces(qm(), environment() if embedded else None)
    assert len(calls) == 1 and [x[1] for x in seen] == [0, 1]
    assert seen[0][0] == seen[1][0]
    for name, surface in result.items():
        assert surface.diagnostics["method"] == "cpu-pbe0-tda-singlet"
        assert surface.diagnostics["energy_definition"] == "E_PBE0+omega"
        assert surface.diagnostics["reference_sha256"] == obj.references[name].sha256
        np.testing.assert_allclose(surface.qm_forces_ev_per_angstrom, -1.0)
        if embedded:
            np.testing.assert_allclose(surface.environment_forces_ev_per_angstrom, -15.0)
    noforce = obj.evaluate_surfaces(qm(), environment() if embedded else None, forces=False)
    assert len(calls) == 2 and len(seen) == 2
    assert all(x.qm_forces_ev_per_angstrom is None and x.environment_forces_ev_per_angstrom is None
               for x in noforce.values())


@pytest.mark.parametrize("change", ["method", "grid", "field"])
def test_mixed_reference_sets_are_rejected_before_pbe0_runtime(monkeypatch, change):
    refs = references(True)
    if change == "method":
        refs["second"] = reference(embedded=True)
    elif change == "grid":
        refs["second"] = small_reference(refs["second"])
    else:
        refs["second"] = replace(refs["second"], charges=replace(charges(), source="Other field"))
    with patch("qmdpa.pbe0_tda._load_runtime") as runtime:
        with pytest.raises((TypeError, ValueError)):
            multistate.PySCFPBE0TDAStateSet(refs)
    runtime.assert_not_called()


def test_collision_and_failed_second_response_never_return_a_partial_set(monkeypatch):
    calls = install_solver(monkeypatch)
    refs = references()
    colliding = {"first": refs["first"], "alias": replace(refs["first"], label="alias")}
    with patch.object(pbe0.PySCFPBE0TDABackend, "_surface_gradient") as response:
        with pytest.raises(EvaluationError, match="same candidate state"):
            multistate.PySCFPBE0TDAStateSet(colliding).evaluate_surfaces(qm())
    response.assert_not_called()
    original = pbe0.PySCFPBE0TDABackend._surface_gradient
    def fail_second(self, solved, index, *args):
        if index == 1:
            raise EvaluationError("Missing second-state grid response")
        return original(self, solved, index, *args)
    obj = multistate.PySCFPBE0TDAStateSet(refs)
    with patch.object(pbe0.PySCFPBE0TDABackend, "_surface_gradient", fail_second):
        with pytest.raises(EvaluationError, match="second-state grid response"):
            obj.evaluate_surfaces(qm())
    assert len(obj.evaluate_surfaces(qm())) == 2 and len(calls) == 3


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("small", [False, True])
def test_prepare_states_dispatches_explicit_pbe0_and_preserves_portable_files(tmp_path, embedded, small):
    refs = references(embedded, small)
    data = request(refs["first"], "pbe0-tda")
    del data["state"]
    data["states"] = {"first": 1, "second": 2}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data))
    target = tmp_path / "out"
    with patch("qmdpa.multistate.build_pbe0_tda_references", return_value=refs) as build, \
            patch("qmdpa.multistate.build_tda_references") as rhf:
        result = prepare_states(path, target)
    build.assert_called_once()
    rhf.assert_not_called()
    for name, entry in result["states"].items():
        provider = resolve_provider_reference(entry["provider"], target / "config.json")
        assert provider["state"]["method"] == "pbe0-tda"
        assert provider["state"]["reference"] == refs[name].to_record()


@pytest.mark.parametrize("change", ["missing_grid", "mixed_result", "gpu", "unknown"])
def test_preparation_cannot_default_or_substitute_a_method(tmp_path, change):
    refs = references(small=True)
    data = request(refs["first"], "pbe0-tda")
    del data["state"]
    data["states"] = {"first": 1, "second": 2}
    if change == "missing_grid":
        del data["qm"]["grid_response"]
    elif change == "gpu":
        data.update(engine="gpu4pyscf", device="cuda")
    elif change == "unknown":
        data["method"] = "arbitrary-td-dft"
    else:
        refs["second"] = reference()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(data))
    target = tmp_path / "out"
    with patch("qmdpa.multistate.build_pbe0_tda_references", return_value=refs) as build:
        with pytest.raises((TypeError, ValueError)):
            prepare_states(path, target)
    assert build.call_count == (1 if change == "mixed_result" else 0)
    assert not (target / "states").exists() and not (target / "result.json").exists()
