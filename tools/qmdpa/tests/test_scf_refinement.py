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



@pytest.mark.parametrize("residual,energy_change,accepted", [
    (1.03e-9, 1e-14, False),
    (1e-10, 2e-12, False),
    (float("nan"), 1e-14, False),
    (1e-10, 1e-14, True),
])
def test_newton_convergence_uses_current_fock_gradient(residual, energy_change, accepted):
    first, last = FakeSCF(), FakeSCF()
    first.newton = lambda: last
    last.mo_coeff, last.mo_occ = np.eye(2), np.array([2., 0.])
    fock = np.eye(2) * 3.

    def gradient(*args):
        if len(args) == 3:
            assert args[0] is last.mo_coeff and args[1] is last.mo_occ
            assert args[2] is fock
            return np.array([residual])
        return np.array([1e-12])

    def kernel(*args):
        # A converged predicted gradient must not hide a current residual or
        # energy change that exceeds the original requested criterion.
        env = dict(mo_coeff=last.mo_coeff, mo_occ=last.mo_occ, fock=fock,
                   e_tot=energy_change, last_hf_e=0., norm_gorb=1e-14)
        assert bool(last.check_convergence(env)) is accepted
        return -1.1

    last.get_grad, last.kernel = gradient, kernel
    with patch.dict("sys.modules", pyscf_modules(first, {})):
        backend = PySCFBackend(scf_solver="diis-newton", conv_tol=1e-12,
                               conv_tol_grad=1e-9, max_cycle=400)
        backend.evaluate(water())
    assert backend.conv_tol == 1e-12 and backend.conv_tol_grad == 1e-9
    assert backend.max_cycle == 400


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
