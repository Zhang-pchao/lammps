# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Exact-input result reuse is bounded, detached, and cleared after failure."""

from dataclasses import replace

import numpy as np
import pytest
from test_excited import charges, environment, install_synthetic_solver, qm, reference, settings

from qmdpa import excited
from qmdpa.core import EvaluationError, Geometry
from qmdpa.embedding import FixedEmbeddingCharges


def provider(monkeypatch, *, embedded=False, enabled=True):
    install_synthetic_solver(monkeypatch)
    solve = excited._solve
    calls = []

    def measured(*args, **kwargs):
        calls.append(args)
        return solve(*args, **kwargs)

    monkeypatch.setattr(excited, "_solve", measured)
    if embedded:
        obj = excited.PySCFTDAElectrostaticBackend(
            settings(), reference(embedded=True), charges(), reuse_identical_geometry=enabled
        )
    else:
        obj = excited.PySCFTDABackend(settings(), reference(), reuse_identical_geometry=enabled)
    return obj, calls


@pytest.mark.parametrize("embedded", [False, True])
def test_only_exact_consecutive_inputs_reuse(monkeypatch, embedded):
    obj, calls = provider(monkeypatch, embedded=embedded)
    args = (qm(), environment()) if embedded else (qm(),)
    a = obj.evaluate_surface(*args)
    b = obj.evaluate_surface(*args)
    assert len(calls) == 1 and a is not b
    assert a.energy_ev == b.energy_ev and dict(a.diagnostics) == dict(b.diagnostics)
    np.testing.assert_array_equal(a.qm_forces_ev_per_angstrom, b.qm_forces_ev_per_angstrom)
    changed = Geometry(qm().atom_ids, qm().symbols, qm().positions + 1e-12)
    obj.evaluate_surface(changed, *args[1:])
    obj.evaluate_surface(*args)
    assert len(calls) == 3
    assert obj.reuse_statistics() == {
        "enabled": True,
        "requests": 4,
        "hits": 1,
        "calculations": 3,
        "failures": 0,
    }


def test_environment_movement_and_force_request_are_part_of_key(monkeypatch):
    obj, calls = provider(monkeypatch, embedded=True)
    obj.evaluate_surface(qm(), environment(), forces=False)
    obj.evaluate_surface(qm(), environment(), forces=True)
    moved = Geometry(environment().atom_ids, environment().symbols, environment().positions + 1e-12)
    obj.evaluate_surface(qm(), moved)
    obj.evaluate_surface(qm(), moved)
    assert len(calls) == 3
    obj.evaluate_surface(qm(), moved, forces=False)
    assert len(calls) == 4


def test_default_disabled_and_explicit_clear_are_cold(monkeypatch):
    obj, calls = provider(monkeypatch, enabled=False)
    obj.evaluate_surface(qm())
    obj.evaluate_surface(qm())
    assert len(calls) == 2 and obj.reuse_statistics()["hits"] == 0
    enabled, calls = provider(monkeypatch)
    enabled.evaluate_surface(qm())
    enabled.clear_reuse()
    enabled.evaluate_surface(qm())
    assert len(calls) == 2
    default = excited.PySCFTDABackend(settings(), reference())
    assert default.reuse_statistics()["enabled"] is False


def test_returned_nested_diagnostics_cannot_poison_saved_result(monkeypatch):
    obj, calls = provider(monkeypatch)
    original = excited._solve

    def nested(*args, **kwargs):
        solved = original(*args, **kwargs)
        solved.tda_diagnostics["nested_test"] = {"values": [1.0]}
        return solved

    monkeypatch.setattr(excited, "_solve", nested)
    a = obj.evaluate_surface(qm())
    a.diagnostics["nested_test"]["values"][0] = 0.0
    b = obj.evaluate_surface(qm())
    assert b.diagnostics["nested_test"]["values"][0] == 1.0
    b.diagnostics["nested_test"]["values"][0] = 0.2
    c = obj.evaluate_surface(qm())
    assert c.diagnostics["nested_test"]["values"][0] == 1.0 and len(calls) == 1
    with pytest.raises(ValueError):
        c.qm_forces_ev_per_angstrom.flags.writeable = True
    stats = obj.reuse_statistics()
    stats["hits"] = 99
    assert obj.reuse_statistics()["hits"] == 2


@pytest.mark.parametrize("kind", ["identity", "periodic", "force_flag", "solver"])
def test_any_failed_request_clears_previous_success(monkeypatch, kind):
    obj, calls = provider(monkeypatch)
    obj.evaluate_surface(qm())
    if kind == "identity":
        def operation():
            return obj.evaluate_surface(Geometry((2, 1), qm().symbols, qm().positions))
    elif kind == "periodic":
        def operation():
            return obj.evaluate_surface(
                    Geometry(qm().atom_ids, qm().symbols, qm().positions, cell=np.eye(3) * 10)
                )
    elif kind == "force_flag":
        def operation():
            return obj.evaluate_surface(qm(), forces=1)
    else:
        original = excited._solve

        def broken(*args, **kwargs):
            raise EvaluationError("Deliberate solver failure")

        monkeypatch.setattr(excited, "_solve", broken)
        def operation():
            return obj.evaluate_surface(
                    Geometry(qm().atom_ids, qm().symbols, qm().positions + 0.1)
                )
    with pytest.raises((ValueError, EvaluationError)):
        operation()
    if kind == "solver":
        monkeypatch.setattr(excited, "_solve", original)
    obj.evaluate_surface(qm())
    assert len(calls) == 2 and obj.reuse_statistics()["hits"] == 0
    assert obj.reuse_statistics()["failures"] == 1


@pytest.mark.parametrize("kind", ["settings", "reference", "runtime", "charges"])
def test_changed_provider_contract_cannot_reuse_or_recalculate(monkeypatch, kind):
    obj, calls = provider(monkeypatch, embedded=True)
    obj.evaluate_surface(qm(), environment())
    if kind == "settings":
        obj.settings = replace(obj.settings, basis="sto-3g")
    elif kind == "reference":
        obj.reference = reference(embedded=True)
    elif kind == "runtime":
        obj._runtime.record["pyscf_version"] = "changed"
    else:
        obj.charges = FixedEmbeddingCharges(charges().atom_ids, [0.2, -0.2], "changed")
    with pytest.raises(EvaluationError, match="changed"):
        obj.evaluate_surface(qm(), environment())
    assert len(calls) == 1 and obj.reuse_statistics()["failures"] == 1


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_reuse_flag_requires_a_boolean(monkeypatch, flag):
    install_synthetic_solver(monkeypatch)
    with pytest.raises(TypeError, match="boolean"):
        excited.PySCFTDABackend(settings(), reference(), reuse_identical_geometry=flag)
