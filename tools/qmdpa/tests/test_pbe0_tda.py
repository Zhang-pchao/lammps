# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""PBE0 method separation and shared response/state/force contract controls."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_excited import T1, T2, charges, environment, qm, reference, response_hook

from qmdpa import excited
from qmdpa import pbe0_tda as pbe0
from qmdpa.core import EvaluationError


def pbe0_reference(embedded=False):
    return pbe0.PBE0TDAReference(
        qm(),
        pbe0.PBE0TDASettings(),
        T1,
        1,
        2.0,
        1.0,
        "synthetic PBE0 state",
        json.dumps({"pyscf_version": excited._PYSCF_VERSION, "source_sha256": pbe0._SOURCE_SHA256}),
        environment() if embedded else None,
        charges() if embedded else None,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"xc": "hf"},
        {"xc": "pbe"},
        {"xc": "wb97m-v"},
        {"basis": "sto-3g"},
        {"charge": 1},
        {"multiplicity": 3},
        {"nstates": 2},
        {"tda_solver": "iterative"},
        {"tda_dense_max_dimension": 512},
        {"grid_level": 5},
        {"grid_level": True},
        {"grid_prune": "none"},
        {"grid_response": False},
        {"grid_response": "true"},
        {"device": "gpu"},
        {"conv_tol": 1e-10},
        {"conv_tol_grad": 1e-7},
        {"cphf_conv_tol": 1e-10},
        {"tda_conv_tol": 1e-9},
        {"response_residual_tol": 1e-6},
        {"density_trace_tol": 1e-6},
        {"threads": 0},
    ],
)
def test_unqualified_method_or_control_cannot_silently_fall_back(change):
    with pytest.raises((ValueError, TypeError)):
        pbe0.PBE0TDASettings(**change)


def test_method_reference_roundtrip_and_canonical_settings_are_distinct():
    r = pbe0_reference(True)
    record = json.loads(json.dumps(r.to_record()))
    loaded = pbe0.PBE0TDAReference.from_record(record)
    assert loaded.to_record() == record and loaded.settings == pbe0.PBE0TDASettings()
    assert record["method"] == "cpu-pbe0-tda-singlet-fixed-reference"
    assert record["settings"]["grid_level"] == 4
    with pytest.raises(ValueError, match="method"):
        excited.TDAReference.from_record(record)
    with pytest.raises(ValueError, match="method"):
        pbe0.PBE0TDAReference.from_record(reference().to_record())
    with pytest.raises(ValueError, match="RHF"):
        excited.TDASettings(xc="pbe0")
    with pytest.raises(TypeError):
        replace(reference(), settings=pbe0.PBE0TDASettings())


@pytest.mark.parametrize(
    "change", ["rhf-runtime", "missing-grid", "missing-grid-response", "wrong-method", "gaussian"]
)
def test_reference_cannot_mix_method_runtime_grid_or_charge_shape(change):
    value = pbe0_reference(True).to_record()
    if change == "rhf-runtime":
        value["runtime"] = reference().runtime
    elif change == "missing-grid":
        del value["settings"]["grid_level"]
    elif change == "missing-grid-response":
        del value["settings"]["grid_response"]
    elif change == "wrong-method":
        value["method"] = "cpu-rhf-tda-singlet-fixed-reference"
    else:
        value["charges"]["radii_angstrom"] = [0.1, 0.1]
    value["sha256"] = excited._digest({k: v for k, v in value.items() if k != "sha256"})
    with pytest.raises((ValueError, EvaluationError)):
        pbe0.PBE0TDAReference.from_record(value)


def test_direct_rks_response_is_checked_without_an_unrequested_solver_change():
    hook = response_hook()

    class OfficialGradient:
        def __init__(self, td):
            pass

        def extra_force(self, atom, envs):
            return np.full(3, 0.25)

        def kernel(self, state):
            return np.asarray([self.extra_force(i, hook) for i in range(2)])

    solved = SimpleNamespace(
        td=object(), overlap=np.eye(2), molecule=SimpleNamespace(natm=2, nelectron=2)
    )
    runtime = SimpleNamespace(tdrhf=SimpleNamespace(Gradients=OfficialGradient))
    gradient, density, diag = excited._total_gradient(
        solved, 0, pbe0.PBE0TDASettings(), runtime, gradient_base=OfficialGradient, refine=False
    )
    np.testing.assert_allclose(gradient, 0.25)
    np.testing.assert_allclose(density, [[1.8, 0.2], [0.2, 0.2]])
    assert diag["cphf_refinement_count"] == 0
    hook["wvo"][0, 0] = 0
    with pytest.raises(EvaluationError, match="residual"):
        excited._total_gradient(
            solved, 0, pbe0.PBE0TDASettings(), runtime, gradient_base=OfficialGradient, refine=False
        )


def install_solver(monkeypatch):
    molecule = SimpleNamespace(nelectron=2, natm=2, intor_symmetric=lambda name: np.eye(2))
    runtime = SimpleNamespace(
        record=pbe0_reference().runtime,
        hartree_ev=10.0,
        bohr_angstrom=2.0,
        gto=SimpleNamespace(intor_cross=lambda *args: np.eye(2)),
    )
    counts = []

    def solve(geometry, *args):
        counts.append(geometry.positions.copy())
        return SimpleNamespace(
            molecule=molecule,
            method=SimpleNamespace(
                nuc_grad_method=lambda: SimpleNamespace(
                    grad_hcore_mm=lambda dm: np.ones((2, 3)),
                    grad_nuc_mm=lambda: np.full((2, 3), 2.0),
                )
            ),
            td=object(),
            energies=np.array([2.0, 3.0, 4.0]),
            transitions=[T1, T2, np.eye(2)],
            tda_diagnostics={"tda_solver": "dense", "rks_grid_response": False},
            overlap=np.eye(2),
            ground_energy_ev=-10.0,
        )

    monkeypatch.setattr(pbe0, "_load_runtime", lambda settings: runtime)
    monkeypatch.setattr(pbe0, "_solve", solve)
    monkeypatch.setattr(excited, "_make_molecule", lambda *args: molecule)
    monkeypatch.setattr(
        pbe0.PySCFPBE0TDABackend,
        "_surface_gradient",
        lambda *args: (
            np.full((2, 3), 0.2),
            np.diag([1.8, 0.2]),
            {"density_trace": 2.0, "cphf_residual_max": 0.0},
        ),
    )
    return counts


def test_same_state_total_energy_both_force_sets_and_exact_reuse(monkeypatch):
    counts = install_solver(monkeypatch)
    r = pbe0_reference(True)
    backend = pbe0.PySCFPBE0TDAElectrostaticBackend(
        r.settings, r, charges(), reuse_identical_geometry=True
    )
    a = backend.evaluate_surface(qm(), environment())
    b = backend.evaluate_surface(qm(), environment())
    assert a.energy_ev == -8.0 and a.ground_energy_ev == -10.0 and a.excitation_energy_ev == 2.0
    assert a.diagnostics["method"] == "cpu-pbe0-tda-singlet"
    assert a.diagnostics["energy_definition"] == "E_PBE0+omega"
    np.testing.assert_allclose(a.qm_forces_ev_per_angstrom, -1.0)
    np.testing.assert_allclose(a.environment_forces_ev_per_angstrom, -15.0)
    assert len(counts) == 1 and backend.reuse_statistics()["hits"] == 1
    assert b is not a
    wrong = copy.deepcopy(dict(a.diagnostics))
    wrong["energy_definition"] = "E_RHF+omega"
    with pytest.raises(EvaluationError):
        excited.TDASurfaceEvaluation(
            a.energy_ev,
            a.qm_atom_ids,
            a.environment_atom_ids,
            a.qm_forces_ev_per_angstrom,
            a.environment_forces_ev_per_angstrom,
            wrong,
        )


def test_cross_method_backend_construction_stops_before_runtime_import(monkeypatch):
    monkeypatch.setattr(
        pbe0, "_load_runtime", lambda *args: pytest.fail("must reject before import")
    )
    r = pbe0_reference()
    with pytest.raises(TypeError):
        pbe0.PySCFPBE0TDABackend(r.settings, reference())
    with pytest.raises(TypeError):
        excited.PySCFTDABackend(r.settings, r)
    with pytest.raises(ValueError, match="isolated reference"):
        pbe0.PySCFPBE0TDABackend(r.settings, pbe0_reference(True))


def test_grid_response_is_assembled_once_after_checked_orbital_response(monkeypatch):
    hook = response_hook()

    class OfficialGradient:
        def __init__(self, td):
            pass

        def extra_force(self, atom, envs):
            return np.full(3, 0.25)

        def kernel(self, state):
            return np.asarray([self.extra_force(i, hook) for i in range(2)])

    solved = SimpleNamespace(
        td=object(),
        overlap=np.eye(2),
        method=object(),
        molecule=SimpleNamespace(natm=2, nelectron=2),
    )
    runtime = SimpleNamespace(
        tdrks=SimpleNamespace(Gradients=OfficialGradient),
        tdrhf=SimpleNamespace(Gradients=OfficialGradient),
    )
    calls = []
    correction = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])

    def grid_response(method, response):
        calls.append(response)
        return correction, {
            "grid_response_model": "test",
            "quadrature_response_points": 10,
            "weight_response": True,
            "coordinate_response": True,
        }

    monkeypatch.setattr(pbe0, "_pbe0_grid_response", grid_response)
    gradient, density, diagnostics = pbe0.PySCFPBE0TDABackend._surface_gradient(
        None, solved, 0, pbe0.PBE0TDASettings(), runtime
    )
    np.testing.assert_allclose(gradient, 0.25 + correction)
    assert len(calls) == 1 and diagnostics["rks_grid_response"] is True
    assert diagnostics["grid_response_force_assembly"] is True
    np.testing.assert_array_equal(diagnostics["grid_response_gradient_hartree_bohr"], correction)
    hook["wvo"][0, 0] = 0
    with pytest.raises(EvaluationError, match="residual"):
        pbe0.PySCFPBE0TDABackend._surface_gradient(None, solved, 0, pbe0.PBE0TDASettings(), runtime)
    assert len(calls) == 1


def test_incomplete_grid_response_propagates_as_failure(monkeypatch):
    hook = response_hook()

    class OfficialGradient:
        def __init__(self, td):
            pass

        def extra_force(self, atom, envs):
            return np.zeros(3)

        def kernel(self, state):
            return np.asarray([self.extra_force(0, hook)])

    solved = SimpleNamespace(
        td=object(),
        overlap=np.eye(2),
        method=object(),
        molecule=SimpleNamespace(natm=1, nelectron=2),
    )
    runtime = SimpleNamespace(
        tdrks=SimpleNamespace(Gradients=OfficialGradient),
        tdrhf=SimpleNamespace(Gradients=OfficialGradient),
    )

    def fail(*args):
        raise EvaluationError("Incomplete XC quadrature response")

    monkeypatch.setattr(pbe0, "_pbe0_grid_response", fail)
    with pytest.raises(EvaluationError, match="Incomplete XC quadrature"):
        pbe0.PySCFPBE0TDABackend._surface_gradient(None, solved, 0, pbe0.PBE0TDASettings(), runtime)
