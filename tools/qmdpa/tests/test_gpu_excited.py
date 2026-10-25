# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU selected-state identity and checked-response contracts without CUDA."""

import builtins
import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import test_excited

from qmdpa import excited
from qmdpa import gpu_excited as gpu
from qmdpa.core import EvaluationError


def record():
    return {
        "pyscf_version": excited._PYSCF_VERSION,
        "source_sha256": dict(gpu._PYSCF_SOURCES),
        "gpu4pyscf_version": gpu._GPU_VERSION,
        "cupy_version": gpu._CUPY_VERSION,
        "gpu_source_sha256": dict(gpu._GPU_SOURCES),
    }


def reference():
    return gpu.GPUTDAReference(
        test_excited.qm(),
        gpu.GPUTDASettings(nstates=2),
        test_excited.T1,
        1,
        2.0,
        1.0,
        "GPU test",
        json.dumps(record()),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cpu"},
        {"device": "gpu"},
        {"xc": "pbe0"},
        {"multiplicity": 3},
        {"nstates": 1},
        {"threads": 0},
        {"tda_dense_max_dimension": 1025},
    ],
)
def test_unsupported_settings(changes):
    with pytest.raises((ValueError, TypeError)):
        gpu.GPUTDASettings(**changes)


def test_no_unrequested_density_fitting():
    with pytest.raises(TypeError):
        gpu.GPUTDASettings(density_fit={"auxbasis": "weigend"})


def test_lazy_reference_roundtrip_and_device_separation():
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.startswith(("cupy", "gpu4pyscf", "pyscf")):
            raise AssertionError("Unexpected runtime import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=checked):
        r = reference()
        assert gpu.GPUTDAReference.from_record(r.to_record()).sha256 == r.sha256
        with pytest.raises(ValueError, match="method"):
            excited.TDAReference.from_record(r.to_record())
        with pytest.raises(ValueError, match="method"):
            gpu.GPUTDAReference.from_record(test_excited.reference().to_record())


@pytest.mark.parametrize(
    "key,value",
    [
        ("gpu4pyscf_version", "other"),
        ("cupy_version", "other"),
        ("gpu_source_sha256", {}),
        ("pyscf_version", "other"),
        ("source_sha256", dict(excited._SOURCE_SHA256)),
    ],
)
def test_runtime_identity_rejected(key, value):
    r = reference()
    with pytest.raises((ValueError, EvaluationError)):
        replace(r, runtime_json=json.dumps({**record(), key: value}))


def test_embedded_gpu_reference_requires_its_extra_runtime_contract():
    with pytest.raises(ValueError, match="runtime identity"):
        replace(
            reference(),
            environment_geometry=test_excited.environment(),
            charges=test_excited.charges(),
        )
    r = replace(
        reference(),
        environment_geometry=test_excited.environment(),
        charges=test_excited.charges(),
        runtime_json=json.dumps({**record(), "embedding_source_sha256": gpu._EMBEDDING_SOURCES}),
    )
    assert gpu.GPUTDAReference.from_record(r.to_record()).sha256 == r.sha256
    with pytest.raises(ValueError, match="point charges"):
        replace(r, charges=replace(r.charges, radii_angstrom=[0.5, 0.5]))


def test_checked_response_does_not_change_solver_result():
    solution = (np.array([[-0.1]]), None)
    def solve(*args, **kwargs):
        return solution
    diagnostics = {}
    checked = gpu._checked_response(
        solve, np.asarray, np.asarray, gpu.GPUTDASettings(), diagnostics
    )
    result = checked(lambda z: z * 0, np.array([0.0, 1.0]), np.array([2.0, 0.0]), np.array([[0.1]]))
    assert np.array_equal(result[0], solution[0]) and result[1] is None
    assert diagnostics["cphf_residual_max"] == 0
    with pytest.raises(EvaluationError, match="exactly one"):
        checked(None, None, None, None)


def test_bad_response_fails_before_force_acceptance():
    checked = gpu._checked_response(
        lambda *args, **kwargs: (np.array([[0.0]]), None),
        np.asarray,
        np.asarray,
        gpu.GPUTDASettings(),
        {},
    )
    with pytest.raises(EvaluationError, match="residual"):
        checked(lambda z: z * 0, np.array([0.0, 1.0]), np.array([2.0, 0.0]), np.array([[0.1]]))


def test_shared_dense_diagonalizer_retains_original_operator_checks():
    from types import SimpleNamespace

    settings = excited.TDASettings(nstates=2, tda_solver="dense")
    diagonal = np.array([0.5, 1.5])
    td = SimpleNamespace(
        _scf=SimpleNamespace(mo_occ=np.array([2.0, 0.0, 0.0])),
        gen_vind=lambda: (lambda x: x * diagonal, diagonal),
        positive_eig_threshold=0.001,
    )
    calls = []

    def solver(matrix):
        calls.append(matrix.copy())
        return np.linalg.eigh(matrix)

    report = excited._dense_tda(td, settings, diagonalize=solver)
    assert len(calls) == 1
    assert report["tda_matrix_residuals_hartree"] == (0.0, 0.0)
    assert np.array_equal(td.e, diagonal)


def test_gpu_response_reuses_bounded_normalized_defect_correction():
    calls = []

    def solve(fvind, energies, occupations, h1, **kwargs):
        calls.append(h1.copy())
        # A deliberately inaccurate small initial RHS emulates a Krylov floor.
        return (-h1 + (1e-5 if len(calls) == 1 else 0)), None

    diagnostics = {}
    checked = gpu._checked_response(
        solve, np.asarray, np.asarray, gpu.GPUTDASettings(), diagnostics
    )
    result = checked(lambda z: z * 0, np.array([0.0, 1.0]), np.array([2.0, 0.0]), np.array([[0.1]]))
    assert len(calls) == 2
    assert abs(result[0][0, 0] + 0.1) < 1e-14
    assert diagnostics["cphf_refinement_iterations"] == 1
    assert diagnostics["cphf_residual_before_refinement"] > 1e-7
    assert diagnostics["cphf_residual_after_refinement"] < 1e-14
