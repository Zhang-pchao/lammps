# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Lazy GPU adapter, no-fallback, complete-gradient and failure contracts."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import test_backends

from qmdpa.core import EvaluationError, Geometry
from qmdpa.gpu_backends import GPU4PySCFBackend


class GPUArray:
    def __init__(self, value):
        self.value = np.asarray(value)


class Gradient:
    __module__ = "gpu4pyscf.df.grad.rks"
    grid_response = False
    auxbasis_response = False

    def __init__(self, owner):
        self.owner = owner

    def kernel(self):
        self.owner.gradient_called = True
        return self.owner.raw_gradient


class GPUMethod:
    __module__ = "gpu4pyscf.df.df_jk"

    def __init__(self):
        self.grids = SimpleNamespace(level=0)
        self.nlcgrids = SimpleNamespace(level=0)
        self.with_df = SimpleNamespace()
        self.converged = True
        self.mo_coeff = GPUArray(np.eye(2))
        self.raw_gradient = GPUArray(np.ones((3, 3)))
        self.gradient_called = False
        self.gradient = Gradient(self)
        self.energy = -1.5

    def kernel(self):
        return self.energy

    def nuc_grad_method(self):
        return self.gradient


def setup(monkeypatch, count=1):
    observed = {}
    gpu = GPUMethod()
    cpu = test_backends.FakeSCF()

    def fitting(**options):
        observed["fit"] = options
        cpu.with_df = SimpleNamespace()
        return cpu

    cpu.density_fit = fitting
    cpu.to_gpu = lambda: gpu
    cupy = ModuleType("cupy")
    cupy.ndarray = GPUArray
    cupy.asnumpy = lambda raw: raw.value
    cupy.cuda = SimpleNamespace(
        runtime=SimpleNamespace(getDeviceCount=lambda: count),
        Device=lambda index: nullcontext(),
        get_current_stream=lambda: SimpleNamespace(
            synchronize=lambda: observed.update(synced=True)
        ),
    )
    modules = {
        **test_backends.pyscf_modules(cpu, observed),
        "cupy": cupy,
        "gpu4pyscf": ModuleType("gpu4pyscf"),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    return cpu, gpu, observed


def make(**options):
    return GPU4PySCFBackend(density_fit={"auxbasis": "def2-universal-jkfit"}, **options)


@pytest.mark.parametrize(
    "options",
    [
        {"xc": "wb97m-v"},
        {"xc": "hf"},
        {"xc": None},
        {"multiplicity": 2},
        {"multiplicity": True},
        {"grid_response": False},
        {"grid_response": 1},
        {"nlc_grid_level": 1},
        {"density_fit": None},
        {"density_fit": {"auxbasis": "../basis"}},
    ],
)
def test_unsupported_settings_rejected_before_optional_import(options):
    with patch.dict(sys.modules, {"pyscf": None, "cupy": None, "gpu4pyscf": None}):
        with pytest.raises(ValueError):
            GPU4PySCFBackend(**{"density_fit": {"auxbasis": "weigend"}, **options})


@pytest.mark.parametrize("host_gradient", [False, True])
def test_official_conversion_full_gradient_and_units(monkeypatch, host_gradient):
    cpu, gpu, observed = setup(monkeypatch)
    if host_gradient:
        gpu.raw_gradient = np.ones((3, 3))
    backend = make(xc=" PBE0 ", threads=2)
    result = backend.evaluate(test_backends.water())
    assert cpu.xc == "pbe0" and observed["fit"] == {
        "auxbasis": "def2-universal-jkfit",
        "only_dfj": False,
    }
    assert result.energy == -3.0
    np.testing.assert_array_equal(result.forces, np.full((3, 3), -4.0))
    assert gpu.gradient.grid_response and gpu.gradient.auxbasis_response and observed["synced"]


@pytest.mark.parametrize(
    "failure", ["cpu-conversion", "cpu-orbitals", "cpu-gradient", "scf", "shape", "nan", "kernel"]
)
def test_failures_never_return_an_energy_force_result(monkeypatch, failure):
    cpu, gpu, observed = setup(monkeypatch)
    backend = make()
    if failure == "cpu-conversion":
        cpu.to_gpu = lambda: cpu
    elif failure == "cpu-orbitals":
        gpu.mo_coeff = np.eye(2)
    elif failure == "cpu-gradient":
        gpu.gradient = SimpleNamespace(kernel=lambda: np.ones((3, 3)), auxbasis_response=True)
    elif failure == "scf":
        gpu.converged = False
    elif failure == "shape":
        gpu.raw_gradient = np.ones((2, 3))
    elif failure == "nan":
        gpu.energy = float("nan")
    else:

        def fail():
            raise RuntimeError("Injected device error")

        gpu.kernel = fail
    with pytest.raises(EvaluationError):
        backend.evaluate(test_backends.water())
    if failure in ("cpu-conversion", "cpu-orbitals", "cpu-gradient", "scf", "kernel"):
        assert not gpu.gradient_called


@pytest.mark.parametrize("count", [0, 2])
def test_no_implicit_device_choice(monkeypatch, count):
    setup(monkeypatch, count=count)
    with pytest.raises(RuntimeError, match="exactly one"):
        make()


def test_periodic_input_remains_unsupported(monkeypatch):
    setup(monkeypatch)
    backend = make()
    water = test_backends.water()
    with pytest.raises(ValueError, match="isolated"):
        backend.evaluate(Geometry(water.atom_ids, water.symbols, water.positions, np.eye(3) * 10))


def test_missing_optional_runtime_has_clear_error(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setitem(sys.modules, "gpu4pyscf", None)
    with pytest.raises(ImportError, match="compatible"):
        make()


@pytest.mark.parametrize(
    "field,value",
    [
        ("xc", "wb97m-v"),
        ("multiplicity", 2),
        ("grid_response", False),
        ("density_fit", None),
        ("nlc_grid_level", 1),
    ],
)
def test_mutating_to_unsupported_settings_cannot_bypass_guard(monkeypatch, field, value):
    cpu, gpu, observed = setup(monkeypatch)
    backend = make()
    setattr(backend, field, value)
    with pytest.raises(EvaluationError):
        backend.evaluate(test_backends.water())
    assert not gpu.gradient_called
