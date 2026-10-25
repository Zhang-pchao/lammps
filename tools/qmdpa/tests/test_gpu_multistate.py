# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU state-set routing, identity and complete-result contracts without CUDA."""

import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from test_excited import T2, charges, environment, install_synthetic_solver, qm, reference
from test_gpu_excited import record
from test_gpu_excited import reference as gpu_reference

from qmdpa import excited, multistate
from qmdpa import gpu_excited as gpu
from qmdpa.core import EvaluationError


def setup(monkeypatch, embedded=False, permute=False):
    _, gradients = install_synthetic_solver(monkeypatch, permute=permute)
    metadata = record()
    if embedded:
        metadata["embedding_source_sha256"] = dict(gpu._EMBEDDING_SOURCES)
    first = replace(
        gpu_reference(),
        settings=gpu.GPUTDASettings(nstates=2, tda_solver="dense"),
        label="first",
        runtime_json=json.dumps(metadata),
        environment_geometry=environment() if embedded else None,
        charges=charges() if embedded else None,
    )
    refs = {
        "first": first,
        "second": replace(
            first, label="second", seed_root=2, transition_density=T2, excitation_energy_ev=3.0
        ),
    }
    runtime = excited._load_runtime(first.settings)
    runtime.record = metadata

    def load(settings, *, electrostatic=False):
        assert electrostatic == embedded
        return runtime

    monkeypatch.setattr(gpu, "_load_runtime", load)
    calls = []

    def solve(*args):
        calls.append(args)
        solved = excited._solve(*args)
        solved.tda_diagnostics.update(engine="gpu4pyscf", device="cuda", cpu_fallback=False)
        return solved

    monkeypatch.setattr(gpu, "_solve", solve)
    monkeypatch.setattr(
        gpu, "_total_gradient", lambda *args, **kwargs: excited._total_gradient(*args)
    )
    return refs, calls, gradients


@pytest.mark.parametrize("embedded", [False, True])
def test_gpu_preparation_retains_field_runtime_and_exact_reference_type(monkeypatch, embedded):
    refs, calls, gradients = setup(monkeypatch, embedded)
    kw = {"environment_geometry": environment(), "charges": charges()} if embedded else {}
    with patch("qmdpa.multistate._solve", side_effect=AssertionError("CPU solve selected")):
        result = multistate.build_gpu_tda_references(
            qm(), refs["first"].settings, states={"first": 1, "second": 2}, **kw
        )
    assert len(calls) == 1 and not gradients
    for name, ref in result.items():
        assert type(ref) is gpu.GPUTDAReference and ref.label == name
        assert gpu.GPUTDAReference.from_record(ref.to_record()).sha256 == ref.sha256
        assert ref.runtime == refs[name].runtime


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("permute", [False, True])
def test_gpu_complete_surfaces_share_solve_and_preserve_named_state_matching(
    monkeypatch, embedded, permute
):
    refs, calls, gradients = setup(monkeypatch, embedded, permute)
    obj = multistate.GPU4PySCFTDAStateSet(refs)
    result = obj.evaluate_surfaces(qm(), environment() if embedded else None)
    assert len(calls) == 1 and gradients == ([1, 0] if permute else [0, 1])
    for name, surface in result.items():
        assert surface.diagnostics["method"] == "gpu-rhf-tda-singlet"
        assert surface.diagnostics["reference_sha256"] == refs[name].sha256
        assert surface.diagnostics["cpu_fallback"] is False
        np.testing.assert_array_equal(surface.qm_forces_ev_per_angstrom, -np.ones((2, 3)))
        if embedded:
            np.testing.assert_array_equal(
                surface.environment_forces_ev_per_angstrom, -np.full((2, 3), 15.0)
            )
    energy_only = obj.evaluate_surfaces(qm(), environment() if embedded else None, forces=False)
    assert len(calls) == 2 and len(gradients) == 2
    assert all(
        x.qm_forces_ev_per_angstrom is None and x.environment_forces_ev_per_angstrom is None
        for x in energy_only.values()
    )


def test_cpu_and_gpu_references_cannot_be_mixed_or_relabeled(monkeypatch):
    refs, calls, _ = setup(monkeypatch)
    with pytest.raises(TypeError):
        multistate.PySCFTDAStateSet(refs)
    refs["second"] = reference()
    with pytest.raises(TypeError):
        multistate.GPU4PySCFTDAStateSet(refs)
    assert not calls


def test_gpu_collisions_and_response_failures_never_return_partial_sets(monkeypatch):
    refs, calls, gradients = setup(monkeypatch)
    colliding = {"first": refs["first"], "alias": replace(refs["first"], label="alias")}
    with pytest.raises(EvaluationError, match="same candidate"):
        multistate.GPU4PySCFTDAStateSet(colliding).evaluate_surfaces(qm())
    assert not gradients
    original = gpu._total_gradient

    def fail_second(solved, index, *args, **kwargs):
        if index == 1:
            raise EvaluationError("Missing second GPU response")
        return original(solved, index, *args, **kwargs)

    obj = multistate.GPU4PySCFTDAStateSet(refs)
    with patch.object(gpu, "_total_gradient", fail_second):
        with pytest.raises(EvaluationError, match="second GPU response"):
            obj.evaluate_surfaces(qm())
    assert len(obj.evaluate_surfaces(qm())) == 2 and len(calls) == 3


def test_gpu_state_set_requires_explicit_dense_solver_before_loading_cuda(monkeypatch):
    refs, calls, _ = setup(monkeypatch)
    settings = replace(refs["first"].settings, tda_solver="iterative")
    with patch.object(gpu, "_load_runtime") as load:
        with pytest.raises(ValueError, match="GPU RHF dense"):
            multistate.build_gpu_tda_references(qm(), settings, states={"first": 1, "second": 2})
    load.assert_not_called()
    assert not calls
