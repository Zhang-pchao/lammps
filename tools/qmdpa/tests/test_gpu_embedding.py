# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""GPU point-charge device, derivative, identity and failure contracts."""

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import test_backends
import test_gpu_backends

from qmdpa.core import EvaluationError, Geometry
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.gpu_backends import GPU4PySCFElectrostaticBackend
from qmdpa.providers import make_qm_provider


class DeviceArray(test_gpu_backends.GPUArray):
    def __array__(self, *args, **kwargs):
        raise AssertionError("Density must not be converted implicitly to a host array")

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    def sum(self, axis):
        return DeviceArray(self.value.sum(axis=axis))


def charges(radii=None):
    return FixedEmbeddingCharges([21, 14], [-0.1, 0.1], "Explicit point-charge fixture", radii)


def environment():
    return Geometry([21, 14], ["H", "H"], [[3.1, 0.3, -0.2], [2.6, 0.9, -0.6]])


def setup(monkeypatch):
    cpu, gpu, observed = test_gpu_backends.setup(monkeypatch)
    sys.modules["pyscf"].qmmm = ModuleType("pyscf.qmmm")
    gpu.conv_tol = 1e-9
    gpu.conv_tol_grad = 1e-8
    gpu.grids.coords = gpu.nlcgrids.coords = None
    gpu.do_nlc = lambda: False
    gpu.density = DeviceArray(np.eye(2))
    gpu.make_rdm1 = lambda: gpu.density
    gpu.mm_electronic = DeviceArray(np.ones((2, 3)))
    gpu.mm_nuclear = np.full((2, 3), 0.5)

    def electronic(density):
        observed["density"] = density
        return gpu.mm_electronic

    gpu.gradient.grad_hcore_mm = electronic
    gpu.gradient.grad_nuc_mm = lambda: gpu.mm_nuclear

    def attach(method, positions, values, *, radii, unit):
        observed["field"] = (positions.copy(), values.copy(), radii, unit)
        return observed.get("embedded_method", method)

    qmmm = ModuleType("gpu4pyscf.qmmm")
    qmmm.itrf = SimpleNamespace(mm_charge=attach)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.qmmm", qmmm)
    sys.modules["cupy"].isfinite = lambda array: np.isfinite(array.value)
    return cpu, gpu, observed


def make(**settings):
    return GPU4PySCFElectrostaticBackend(
        charges(), density_fit={"auxbasis": "def2-universal-jkfit"}, **settings
    )


@pytest.mark.parametrize("kind", ["gaussian", "dipole", "wrong-charges"])
def test_unsupported_request_is_rejected_before_optional_import(kind):
    c = charges([0.5, 0.5]) if kind == "gaussian" else charges()
    if kind == "wrong-charges":
        c = object()
    with patch.dict(sys.modules, {"pyscf": None, "gpu4pyscf": None, "cupy": None}):
        with pytest.raises(ValueError):
            GPU4PySCFElectrostaticBackend(
                c, density_fit={"auxbasis": "weigend"}, include_qm_dipole=kind == "dipole"
            )


def test_shared_energy_force_contract_keeps_gpu_density_on_device(monkeypatch):
    cpu, gpu, observed = setup(monkeypatch)
    backend = make()
    result = backend.evaluate(test_backends.water(), environment())
    assert result.energy_ev == -3.0
    np.testing.assert_array_equal(result.qm_forces_ev_per_angstrom, np.full((3, 3), -4.0))
    np.testing.assert_array_equal(result.environment_forces_ev_per_angstrom, np.full((2, 3), -6.0))
    assert observed["density"] is gpu.density
    assert observed["field"][2:] == (None, "Angstrom")
    assert gpu.gradient.grid_response and gpu.gradient.auxbasis_response and observed["synced"]
    assert result.diagnostics["engine"] == "gpu4pyscf"
    assert result.diagnostics["device"] == "cuda" and result.diagnostics["cpu_fallback"] is False
    with pytest.raises(ValueError):
        result.environment_forces_ev_per_angstrom.setflags(write=True)


def test_generated_official_wrapper_accepts_gpu_ancestry(monkeypatch):
    cpu, gpu, observed = setup(monkeypatch)
    wrapper = type("QMMMDFRKS", (type(gpu),), {"__module__": "pyscf.lib.misc"})
    gpu.__class__ = wrapper
    gradient_wrapper = type(
        "QMMMGradients", (type(gpu.gradient),), {"__module__": "pyscf.lib.misc"}
    )
    gpu.gradient.__class__ = gradient_wrapper
    assert make().evaluate(test_backends.water(), environment()).energy_ev == -3.0


@pytest.mark.parametrize(
    "failure",
    [
        "cpu-wrapper",
        "cpu-density",
        "cpu-orbitals",
        "cpu-gradient",
        "density-nan",
        "density-shape",
        "environment-shape",
        "environment-nan",
        "scf",
    ],
)
def test_invalid_device_or_incomplete_derivatives_never_return_result(monkeypatch, failure):
    cpu, gpu, observed = setup(monkeypatch)
    backend = make()
    if failure == "cpu-wrapper":
        observed["embedded_method"] = cpu
    elif failure == "cpu-density":
        gpu.density = np.eye(2)
    elif failure == "cpu-orbitals":
        gpu.mo_coeff = np.eye(2)
    elif failure == "cpu-gradient":
        gpu.gradient = SimpleNamespace(kernel=lambda: np.ones((3, 3)), auxbasis_response=True)
    elif failure == "density-nan":
        gpu.density = DeviceArray(np.full((2, 2), np.nan))
    elif failure == "density-shape":
        gpu.density = DeviceArray(np.ones((2, 3)))
    elif failure == "environment-shape":
        gpu.mm_electronic = np.ones((1, 3))
    elif failure == "environment-nan":
        gpu.mm_nuclear[:] = np.nan
    else:
        gpu.converged = False
    with pytest.raises(EvaluationError):
        backend.evaluate(test_backends.water(), environment())
    if failure in ("cpu-wrapper", "cpu-orbitals", "cpu-gradient", "scf"):
        assert not gpu.gradient_called


def test_fixed_membership_and_unsupported_mutation_remain_errors(monkeypatch):
    cpu, gpu, observed = setup(monkeypatch)
    backend = make()
    qm = test_backends.water()
    env = environment()
    backend.evaluate(qm, env)
    with pytest.raises(ValueError, match="stay fixed"):
        backend.evaluate(replace(qm, atom_ids=[9, 8, 7]), env)
    with pytest.raises(ValueError, match="order"):
        backend.evaluate(qm, env.select([14, 21]))
    backend.include_qm_dipole = True
    with pytest.raises(ValueError, match="point charges"):
        backend.evaluate(qm, env)


def test_factory_selects_complete_gpu_embedding_and_requires_selected_reference():
    spec = {"engine": "gpu4pyscf", "device": "cuda", "state": {"method": "ground"}}
    field = charges()
    with patch("qmdpa.gpu_backends.GPU4PySCFElectrostaticBackend") as embedded:
        assert (
            make_qm_provider({"basis": "sto-3g"}, provider=spec, charges=field)
            is embedded.return_value
        )
        embedded.assert_called_once_with(field, basis="sto-3g")
    spec["state"] = {"method": "rhf-tda"}
    with pytest.raises(ValueError, match="complete inline reference"):
        make_qm_provider({}, provider=spec, charges=field)
