# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Conventional RHF routing without DFT grids or fitted-integral fallback."""

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import test_backends
import test_gpu_backends
import test_gpu_embedding

from qmdpa.config import validate_qm
from qmdpa.core import EvaluationError
from qmdpa.gpu_backends import GPU4PySCFBackend, GPU4PySCFElectrostaticBackend

GPU = {"engine": "gpu4pyscf", "device": "cuda", "state": {"method": "ground"}}


def setup(monkeypatch, embedded=False):
    factory = test_gpu_embedding.setup if embedded else test_gpu_backends.setup
    cpu, gpu, observed = factory(monkeypatch)
    def rhf(molecule):
        observed["rhf"] = molecule
        return cpu
    sys.modules["pyscf"].scf = SimpleNamespace(RHF=rhf)
    for name in ("grids", "nlcgrids", "with_df", "do_nlc"):
        if hasattr(gpu, name):
            delattr(gpu, name)
    def forbidden(*args, **kwargs):
        raise AssertionError("RHF must not construct DFT or fitted integrals")
    sys.modules["pyscf"].dft.RKS = forbidden
    cpu.density_fit = forbidden
    return cpu, gpu, observed


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("xc", ["hf", " HF "])
def test_rhf_uses_official_conversion_and_complete_forces(monkeypatch, embedded, xc):
    cpu, gpu, observed = setup(monkeypatch, embedded)
    kwargs = dict(xc=xc, basis="sto-3g", conv_tol=1e-12,
                  conv_tol_grad=1e-10, max_cycle=91, threads=2)
    if embedded:
        result = GPU4PySCFElectrostaticBackend(test_gpu_embedding.charges(), **kwargs).evaluate(
            test_backends.water(), test_gpu_embedding.environment())
        energy = result.energy_ev
        force = result.qm_forces_ev_per_angstrom
        np.testing.assert_array_equal(result.environment_forces_ev_per_angstrom, np.full((2, 3), -6.))
        assert result.diagnostics["method"] == "rhf"
        assert result.diagnostics["integrals"] == "conventional"
        assert result.diagnostics["grid_response"] is None
        assert result.diagnostics["regular_grid_level"] is None
        assert result.diagnostics["nlc_enabled"] is False
        assert "density_fit" not in result.diagnostics
        assert observed["density"] is gpu.density
    else:
        result = GPU4PySCFBackend(**kwargs).evaluate(test_backends.water())
        energy, force = result.energy, result.forces
    assert "rhf" in observed and "fit" not in observed
    assert cpu.conv_tol == 1e-12 and cpu.conv_tol_grad == 1e-10 and cpu.max_cycle == 91
    assert energy == -3.
    np.testing.assert_array_equal(force, np.full((3, 3), -4.))
    assert not gpu.gradient.grid_response and not gpu.gradient.auxbasis_response
    assert observed["synced"]


@pytest.mark.parametrize("settings", [
    {"density_fit": {"auxbasis": "weigend"}}, {"multiplicity": 3},
    {"nlc_grid_level": 1}, {"include_qm_dipole": True},
])
def test_unsupported_rhf_combinations_fail_before_import(settings):
    with patch.dict(sys.modules, {"pyscf": None, "cupy": None, "gpu4pyscf": None}):
        with pytest.raises(ValueError):
            GPU4PySCFBackend(xc="hf", **settings)


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("failure", ["cpu-conversion", "cpu-orbitals", "scf"])
def test_rhf_failures_never_return_stale_force(monkeypatch, embedded, failure):
    cpu, gpu, observed = setup(monkeypatch, embedded)
    backend = (GPU4PySCFElectrostaticBackend(test_gpu_embedding.charges(), xc="hf")
               if embedded else GPU4PySCFBackend(xc="hf"))
    args = (test_backends.water(), test_gpu_embedding.environment()) if embedded else (test_backends.water(),)
    backend.evaluate(*args)
    gpu.gradient_called = False
    if failure == "cpu-conversion":
        cpu.to_gpu = lambda: cpu
    elif failure == "cpu-orbitals":
        gpu.mo_coeff = np.eye(2)
    else:
        gpu.converged = False
    with pytest.raises(EvaluationError):
        backend.evaluate(*args)
    assert not gpu.gradient_called


@pytest.mark.parametrize("electrostatic", [False, True])
def test_lazy_rhf_config_normalizes_absent_and_null_fitting(electrostatic):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf", "cupy")):
            raise AssertionError("Unexpected engine import: " + name)
        return original(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=checked):
        plain = validate_qm({"xc": "hf"}, provider=GPU, electrostatic=electrostatic)
        explicit = validate_qm({"xc": "hf", "density_fit": None}, provider=GPU, electrostatic=electrostatic)
    assert plain == explicit and plain["density_fit"] is None
    assert plain["xc"] == "hf"
