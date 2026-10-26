#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit two-stage SCF policy, final residual and unsupported-route rejection."""

from unittest.mock import patch

import numpy as np
import pytest
from test_backends import FakeSCF, pyscf_modules, water

from qmdpa.backends import PySCFBackend
from qmdpa.config import validate_qm
from qmdpa.core import EvaluationError
from qmdpa.embedding import PySCFElectrostaticBackend
from qmdpa.gpu_backends import validate_gpu_settings


@pytest.mark.parametrize("first_converged", [False, True])
@pytest.mark.parametrize("failure", [None, "convergence", "residual", "nonfinite"])
def test_final_stage_and_explicit_physical_residual(first_converged, failure):
    first = FakeSCF(first_converged)
    last = FakeSCF(failure != "convergence")
    calls = []
    first.kernel = lambda: calls.append("diis") or -1.
    first.newton = lambda: last
    last.mo_coeff, last.mo_occ = np.eye(2), np.array([2., 0.])
    last.kernel = lambda *_: calls.append("newton") or -1.1
    residual = 1e-7 if failure == "residual" else float("nan") if failure == "nonfinite" else 1e-12
    last.get_grad = lambda *_: np.array([residual])
    with patch.dict("sys.modules", pyscf_modules(first, {})):
        backend = PySCFBackend(scf_solver="diis-newton", conv_tol_grad=1e-9)
        if failure is not None:
            with pytest.raises(EvaluationError):
                backend.evaluate(water())
            assert last.gradient_called is False
        else:
            assert backend.evaluate(water()).energy == -2.2
            assert last.gradient_called is True
    assert calls == ["diis", "newton"]
    assert last.ah_lindep == 1e-20
    assert last.ah_conv_tol == 1e-24
    assert last.ah_start_tol == 1e-10


def test_default_never_tries_newton_after_failed_scf():
    method = FakeSCF(False)
    method.newton = lambda: pytest.fail("Undeclared solver fallback")
    with patch.dict("sys.modules", pyscf_modules(method, {})):
        backend = PySCFBackend()
        with pytest.raises(EvaluationError, match="SCF did not converge"):
            backend.evaluate(water())


@pytest.mark.parametrize("options", [{"scf_solver": "automatic"}, {"scf_solver": "diis-newton"}])
def test_incomplete_solver_request_is_rejected_before_import(options):
    with pytest.raises(ValueError):
        PySCFBackend(**options)
    with pytest.raises(ValueError):
        validate_qm(options)


def test_unqualified_embedded_and_gpu_paths_are_rejected():
    options = {"scf_solver": "diis-newton", "conv_tol_grad": 1e-9}
    with pytest.raises(ValueError, match="isolated CPU"):
        validate_qm(options, electrostatic=True)
    with pytest.raises(ValueError, match="isolated CPU"):
        validate_gpu_settings(options)
    with pytest.raises(ValueError, match="isolated CPU"):
        PySCFElectrostaticBackend(None, **options)
