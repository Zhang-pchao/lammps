# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""One common solve with distinct fixed states and all-or-error results."""

from dataclasses import replace

import numpy as np
import pytest
from test_excited import (
    T2,
    charges,
    environment,
    install_synthetic_solver,
    qm,
    reference,
    settings,
)

from qmdpa import excited, multistate
from qmdpa.core import EvaluationError, Geometry
from qmdpa.pbe0_tda import PBE0TDASettings


def setup(monkeypatch, *, embedded=False, permute=False):
    _, gradients = install_synthetic_solver(monkeypatch, permute=permute)
    original = excited._solve
    calls = []

    def measured(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(excited, "_solve", measured)
    monkeypatch.setattr(multistate, "_solve", measured)
    monkeypatch.setattr(multistate, "_load_runtime", excited._load_runtime)
    a = reference(embedded=embedded, configuration=settings(tda_solver="dense"))
    b = replace(
        a, transition_density=T2, seed_root=2, excitation_energy_ev=3.0, label="second fixed state"
    )
    return {"first": a, "second": b}, calls, gradients


@pytest.mark.parametrize("embedded", [False, True])
def test_preparation_shares_one_solve_and_roundtrips_existing_references(monkeypatch, embedded):
    _, calls, gradients = setup(monkeypatch)
    kwargs = {"environment_geometry": environment(), "charges": charges()} if embedded else {}
    refs = multistate.build_tda_references(
        qm(), settings(tda_solver="dense"), states={"first": 1, "second": 2}, **kwargs
    )
    assert len(calls) == 1 and gradients == []
    for name, ref in refs.items():
        assert ref.label == name
        assert excited.TDAReference.from_record(ref.to_record()).sha256 == ref.sha256
    assert [x.seed_root for x in refs.values()] == [1, 2]
    with pytest.raises(TypeError):
        refs["third"] = refs["first"]


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("permute", [False, True])
def test_complete_surfaces_share_one_solve_and_track_fixed_names(monkeypatch, embedded, permute):
    refs, calls, gradients = setup(monkeypatch, embedded=embedded, permute=permute)
    obj = multistate.PySCFTDAStateSet(refs)
    refs.clear()
    args = (qm(), environment()) if embedded else (qm(),)
    results = obj.evaluate_surfaces(*args)
    assert len(calls) == 1 and gradients == ([1, 0] if permute else [0, 1])
    assert list(results) == ["first", "second"]
    assert results["second"].energy_ev - results["first"].energy_ev == pytest.approx(1.0)
    for name, result in results.items():
        assert result.diagnostics["reference_sha256"] == obj.references[name].sha256
        np.testing.assert_array_equal(result.qm_forces_ev_per_angstrom, -np.ones((2, 3)))
        if embedded:
            np.testing.assert_array_equal(
                result.environment_forces_ev_per_angstrom, -np.full((2, 3), 15.0)
            )
        else:
            assert result.environment_forces_ev_per_angstrom is None
        with pytest.raises(ValueError):
            result.qm_forces_ev_per_angstrom.flags.writeable = True
    with pytest.raises(TypeError):
        results["third"] = results["first"]
    with pytest.raises(TypeError):
        obj.references["third"] = obj.references["first"]
    obj.evaluate_surfaces(*args)
    assert len(calls) == 2  # No implicit last-result cache.


@pytest.mark.parametrize("embedded", [False, True])
def test_energy_only_skips_each_response_and_never_reports_zero_forces(monkeypatch, embedded):
    refs, calls, gradients = setup(monkeypatch, embedded=embedded)
    result = multistate.PySCFTDAStateSet(refs).evaluate_surfaces(
        qm(), environment() if embedded else None, forces=False
    )
    assert len(calls) == 1 and gradients == []
    for surface in result.values():
        assert surface.qm_forces_ev_per_angstrom is None
        assert surface.environment_forces_ev_per_angstrom is None
        assert surface.diagnostics["forces_evaluated"] is False


def test_collision_is_detected_before_any_gradient(monkeypatch):
    refs, calls, gradients = setup(monkeypatch)
    refs["second"] = replace(refs["first"], label="same state with another name")
    obj = multistate.PySCFTDAStateSet(refs)
    with pytest.raises(EvaluationError, match="same candidate state"):
        obj.evaluate_surfaces(qm())
    assert len(calls) == 1 and gradients == []


def test_ambiguous_state_does_not_return_first_valid_state(monkeypatch):
    refs, _, gradients = setup(monkeypatch)
    refs["second"] = replace(refs["second"], transition_density=np.ones((2, 2)))
    obj = multistate.PySCFTDAStateSet(refs)
    with pytest.raises(EvaluationError):
        obj.evaluate_surfaces(qm())
    assert gradients == []


def test_failed_later_response_leaves_next_call_cold(monkeypatch):
    refs, calls, gradients = setup(monkeypatch)
    obj = multistate.PySCFTDAStateSet(refs)
    original = excited._total_gradient

    def failure(solved, index, *args):
        if index == 1:
            raise EvaluationError("Deliberate second-state response failure")
        return original(solved, index, *args)

    monkeypatch.setattr(excited, "_total_gradient", failure)
    with pytest.raises(EvaluationError, match="second-state"):
        obj.evaluate_surfaces(qm())
    assert gradients == [0]
    monkeypatch.setattr(excited, "_total_gradient", original)
    result = obj.evaluate_surfaces(qm())
    assert len(calls) == 2 and len(result) == 2 and gradients == [0, 0, 1]


@pytest.mark.parametrize("change", ["settings", "qm_ids", "field", "mode", "duplicate"])
def test_incompatible_reference_sets_fail_before_quantum_calls(monkeypatch, change):
    refs, calls, gradients = setup(monkeypatch, embedded=True)
    b = refs["second"]
    if change == "settings":
        b = replace(b, settings=replace(b.settings, basis="sto-3g"))
    elif change == "qm_ids":
        b = replace(b, qm_geometry=Geometry([3, 4], qm().symbols, qm().positions))
    elif change == "field":
        b = replace(b, charges=replace(b.charges, source="Different field source"))
    elif change == "mode":
        b = replace(b, environment_geometry=None, charges=None)
    else:
        b = refs["first"]
    refs["second"] = b
    with pytest.raises(ValueError):
        multistate.PySCFTDAStateSet(refs)
    assert calls == [] and gradients == []


@pytest.mark.parametrize(
    "bad", [{}, {"a": 1}, {"a": 1, "b": 1}, {"a": True, "b": 2}, {"a": 1, "b": 3}, {"a": 1, " ": 2}]
)
def test_bad_preparation_roots_and_names_fail_before_solve(monkeypatch, bad):
    _, calls, _ = setup(monkeypatch)
    with pytest.raises(ValueError):
        multistate.build_tda_references(qm(), settings(tda_solver="dense"), states=bad)
    assert calls == []


@pytest.mark.parametrize("configuration", [settings(), PBE0TDASettings()])
def test_other_solvers_and_methods_are_not_implicitly_admitted(monkeypatch, configuration):
    _, calls, _ = setup(monkeypatch)
    with pytest.raises(ValueError, match="CPU RHF dense"):
        multistate.build_tda_references(qm(), configuration, states={"a": 1, "b": 2})
    assert calls == []


def test_gaussian_fields_and_foreign_reference_types_are_not_admitted(monkeypatch):
    refs, calls, _ = setup(monkeypatch, embedded=True)
    gaussian = replace(charges(), radii_angstrom=[1.0, 1.0])
    with pytest.raises(ValueError, match="point charges"):
        multistate.build_tda_references(
            qm(),
            refs["first"].settings,
            states={"a": 1, "b": 2},
            environment_geometry=environment(),
            charges=gaussian,
        )
    refs = {name: replace(ref, charges=gaussian) for name, ref in refs.items()}
    with pytest.raises(ValueError, match="point charges"):
        multistate.PySCFTDAStateSet(refs)
    with pytest.raises(TypeError, match="TDAReference"):
        multistate.PySCFTDAStateSet({"a": object(), "b": object()})
    assert calls == []


@pytest.mark.parametrize("change", ["identity", "periodic", "force_flag", "runtime", "settings"])
def test_invalid_request_is_rejected_before_shared_solve(monkeypatch, change):
    refs, calls, _ = setup(monkeypatch)
    obj = multistate.PySCFTDAStateSet(refs)
    geo = qm()
    kwargs = {}
    if change == "identity":
        geo = Geometry([2, 1], geo.symbols, geo.positions)
    elif change == "periodic":
        geo = replace(geo, cell=np.eye(3) * 10)
    elif change == "force_flag":
        kwargs["forces"] = 1
    elif change == "runtime":
        obj._providers[1][1]._runtime.record["pyscf_version"] = "changed"
    else:
        obj._providers[1][1].settings = replace(refs["first"].settings, basis="sto-3g")
    with pytest.raises((ValueError, EvaluationError)):
        obj.evaluate_surfaces(geo, **kwargs)
    assert calls == []
