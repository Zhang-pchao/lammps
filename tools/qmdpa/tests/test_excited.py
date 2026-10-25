# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Synthetic contract tests; real RHF/TDA finite differences run separately."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from qmdpa import excited
from qmdpa.core import EvaluationError, Geometry
from qmdpa.embedding import FixedEmbeddingCharges

T1 = np.array([[0.0, 1.0], [0.0, 0.0]])
T2 = T1.T


def settings(**changes):
    return excited.TDASettings(nstates=2, **changes)


def qm():
    return Geometry((1, 2), ("H", "H"), [[0, 0, 0], [0, 0, 0.8]])


def environment():
    return Geometry((10, 11), ("O", "H"), [[3, 0, 0], [4, 0, 0]])


def charges():
    return FixedEmbeddingCharges((10, 11), [0.1, -0.1], "synthetic test charges")


def runtime_record():
    return {
        "pyscf_version": excited._PYSCF_VERSION,
        "source_sha256": dict(excited._SOURCE_SHA256),
    }


def reference(*, embedded=False, configuration=None):
    return excited.TDAReference(
        qm(),
        settings() if configuration is None else configuration,
        T1,
        1,
        2.0,
        1.0,
        "synthetic fixed state",
        json.dumps(runtime_record()),
        environment() if embedded else None,
        charges() if embedded else None,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"xc": "pbe"},
        {"xc": "wb97m-v"},
        {"xc": "scan"},
        {"device": "gpu"},
        {"multiplicity": 3},
        {"nstates": 1},
        {"nstates": True},
        {"min_overlap": 0},
        {"min_overlap": 1.1},
        {"min_margin": 0},
        {"min_gap_ev": -1},
        {"conv_tol": np.nan},
        {"conv_tol_grad": np.inf},
        {"max_cycle": 1.2},
        {"threads": 0},
        {"response_residual_tol": False},
    ],
)
def test_settings_reject_unsupported_or_invalid_values(change):
    with pytest.raises((ValueError, TypeError)):
        excited.TDASettings(**change)


def test_reference_json_round_trip_and_immutability():
    original = reference(embedded=True)
    restored = excited.TDAReference.from_record(json.loads(json.dumps(original.to_record())))
    assert restored.sha256 == original.sha256
    assert restored.settings == original.settings
    assert restored.to_record() == original.to_record()
    with pytest.raises(ValueError):
        restored.transition_density.flags.writeable = True
    exposed = restored.runtime
    exposed["source_sha256"].clear()
    assert restored.runtime == runtime_record()


@pytest.mark.parametrize("field", ["transition_density", "settings", "charges"])
def test_reference_tamper_is_rejected(field):
    record = reference(embedded=True).to_record()
    if field == "transition_density":
        record[field][0][1] = 0.9
    elif field == "settings":
        record[field]["charge"] = 2
    else:
        record[field]["charges_e"][0] = 0.2
    with pytest.raises(ValueError, match="SHA256"):
        excited.TDAReference.from_record(record)


def test_reference_requires_complete_canonical_settings_and_source_identity():
    record = reference().to_record()
    record["settings"].pop("threads")
    record["sha256"] = excited._digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError, match="all explicit"):
        excited.TDAReference.from_record(record)
    record = reference().to_record()
    record["runtime"]["source_sha256"]["grad/tdrhf.py"] = "0" * 64
    record["sha256"] = excited._digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(EvaluationError, match="source hashes"):
        excited.TDAReference.from_record(record)


def test_source_pin_rejects_changed_version_or_gradient_contract():
    wrong = runtime_record()
    wrong["pyscf_version"] = "2.13.0"
    with pytest.raises(EvaluationError, match="2.14.0"):
        excited._verify_runtime_record(wrong)
    wrong = runtime_record()
    wrong["source_sha256"]["grad/tdrhf.py"] = "f" * 64
    with pytest.raises(EvaluationError, match="source hashes"):
        excited._verify_runtime_record(wrong)


def test_transition_overlap_phase_and_nonorthogonal_ao_metric():
    # Candidate AOs span the same functions after a nonsingular basis transform.
    transform = np.array([[1.2, 0.3], [0.1, 0.9]])
    inverse = np.linalg.inv(transform)
    candidate = inverse @ T1 @ inverse.T
    metric = transform.T @ transform
    score = excited.transition_overlap(T1, candidate, np.eye(2), metric, transform)
    negative = excited.transition_overlap(T1, -candidate, np.eye(2), metric, transform)
    assert score == pytest.approx(1.0)
    assert negative == pytest.approx(score)
    assert excited.transition_overlap(T1, T2, np.eye(2), np.eye(2), np.eye(2)) == 0
    with pytest.raises(EvaluationError, match="shapes"):
        excited.transition_overlap(T1, T2, np.eye(2), np.eye(2), np.eye(3))
    with pytest.raises(EvaluationError, match="positive AO norms"):
        excited.transition_overlap(np.zeros((2, 2)), T1, np.eye(2), np.eye(2), np.eye(2))


def test_state_selection_follows_root_permutation():
    first, d1 = excited.select_tda_state([1, 0], [2, 3], settings())
    second, d2 = excited.select_tda_state([0, 1], [3, 2], settings())
    assert first == 0
    assert second == 1
    assert d1["overlap"] == d2["overlap"] == 1
    assert d1["candidate_gap_ev"] == d2["candidate_gap_ev"] == 1


@pytest.mark.parametrize(
    "scores,energies,message",
    [
        ([0.5, 0.1], [2, 3], "minimum reference overlap"),
        ([0.9, 0.85], [2, 3], "overlap margin"),
        ([0.9, 0.1], [2, 2.0001], "candidate energy"),
        ([1], [2], "every requested"),
        ([1, 0], [2, -3], "invalid"),
        ([np.nan, 0], [2, 3], "finite"),
    ],
)
def test_state_selection_fails_closed(scores, energies, message):
    with pytest.raises(EvaluationError, match=message):
        excited.select_tda_state(scores, energies, settings())


def response_hook():
    return {
        "oo0": np.diag([1.0, 0.0]),
        "dmz1doo": np.array([[-0.2, 0.4], [0.0, 0.2]]),
        "z1": np.array([[0.5]]),
        "wvo": np.array([[-0.5]]),
        "fvind": lambda vector: np.zeros_like(vector),
        "mo_energy": np.array([0.0, 1.0]),
        "nocc": 1,
        "nvir": 1,
    }


def test_relaxed_density_uses_response_and_checks_cphf_equation():
    density, diagnostic = excited._relaxed_density(response_hook(), np.eye(2), 2, settings())
    np.testing.assert_allclose(density, [[1.8, 0.2], [0.2, 0.2]])
    assert diagnostic["density_trace"] == pytest.approx(2)
    assert diagnostic["cphf_residual_max"] == 0
    hook = response_hook()
    hook["wvo"] = np.array([[0.0]])
    with pytest.raises(EvaluationError, match="residual"):
        excited._relaxed_density(hook, np.eye(2), 2, settings())
    hook = response_hook()
    hook["dmz1doo"][0, 0] = 1
    with pytest.raises(EvaluationError, match="electron count"):
        excited._relaxed_density(hook, np.eye(2), 2, settings())
    hook = response_hook()
    hook.pop("z1")
    with pytest.raises(EvaluationError, match="missing"):
        excited._relaxed_density(hook, np.eye(2), 2, settings())


def candidate_objects():
    method = SimpleNamespace(mo_coeff=np.eye(3), mo_occ=np.array([2, 0, 0]))
    td = SimpleNamespace(
        e=np.array([0.2, 0.3]),
        converged=np.array([True, True]),
        xy=[(np.array([[1 / np.sqrt(2), 0]]), 0), (np.array([[0, 1 / np.sqrt(2)]]), 0)],
        gen_vind=lambda: (lambda vectors: vectors * np.array([0.2, 0.3]), None),
    )
    return method, td, SimpleNamespace(hartree_ev=10.0)


def test_candidate_validation_total_roots_normalization_and_convergence():
    method, td, runtime = candidate_objects()
    energies, transitions, residuals = excited._candidate_data(method, td, settings(), runtime)
    np.testing.assert_allclose(energies, [2, 3])
    assert transitions[0].shape == (3, 3)
    assert residuals["tda_eigensystem_residual_max_hartree"] == 0
    td.converged[1] = False
    with pytest.raises(EvaluationError, match="must converge"):
        excited._candidate_data(method, td, settings(), runtime)
    td.converged[1] = True
    td.xy[0][0][0, 0] *= 2
    with pytest.raises(EvaluationError, match="amplitudes violate"):
        excited._candidate_data(method, td, settings(), runtime)
    method, td, runtime = candidate_objects()
    td.xy.pop()
    with pytest.raises(EvaluationError, match="requested number"):
        excited._candidate_data(method, td, settings(), runtime)


def install_synthetic_solver(monkeypatch, *, permute=False):
    molecule = SimpleNamespace(
        nelectron=2,
        natm=2,
        intor_symmetric=lambda name: np.eye(2),
    )
    runtime = SimpleNamespace(
        record=runtime_record(),
        hartree_ev=10.0,
        bohr_angstrom=2.0,
        gto=SimpleNamespace(intor_cross=lambda *args: np.eye(2)),
    )
    captured_densities = []
    gradient_calls = []
    mm_gradient = SimpleNamespace(
        grad_hcore_mm=lambda dm: captured_densities.append(dm.copy()) or np.ones((2, 3)),
        grad_nuc_mm=lambda: np.full((2, 3), 2.0),
    )

    def solve(geometry, *args):
        # The selected total energy depends on coordinates for call-order checks.
        return SimpleNamespace(
            molecule=molecule,
            method=SimpleNamespace(nuc_grad_method=lambda: mm_gradient),
            td=object(),
            energies=np.array([3, 2] if permute else [2, 3]),
            transitions=[T2, -T1] if permute else [T1, T2],
            tda_diagnostics={"tda_solver": "iterative"},
            overlap=np.eye(2),
            ground_energy_ev=-10 + float(np.sum(geometry.positions)),
        )

    def gradient(*args):
        gradient_calls.append(args[1])
        return (
            np.full((2, 3), 0.2),
            np.diag([1.8, 0.2]),
            {
                "density_trace": 2.0,
                "cphf_residual_max": 0.0,
            },
        )

    monkeypatch.setattr(excited, "_load_runtime", lambda configuration: runtime)
    monkeypatch.setattr(excited, "_make_molecule", lambda *args: molecule)
    monkeypatch.setattr(excited, "_solve", solve)
    monkeypatch.setattr(excited, "_total_gradient", gradient)
    return captured_densities, gradient_calls


def test_total_energy_gradient_sign_units_and_energy_only_contract(monkeypatch):
    _, gradient_calls = install_synthetic_solver(monkeypatch)
    provider = excited.PySCFTDABackend(settings(), reference())
    energy_only = provider.evaluate_surface(qm(), forces=False)
    assert energy_only.energy_ev == pytest.approx(-10 + 0.8 + 2)
    assert energy_only.qm_forces_ev_per_angstrom is None
    assert energy_only.environment_forces_ev_per_angstrom is None
    assert gradient_calls == []
    result = provider.evaluate(qm())
    np.testing.assert_allclose(result.forces, -1)
    assert result.energy == energy_only.energy_ev
    assert gradient_calls == [0]


def test_embedded_reaction_force_uses_relaxed_density_and_nuclear_term(monkeypatch):
    densities, _ = install_synthetic_solver(monkeypatch, permute=True)
    provider = excited.PySCFTDAElectrostaticBackend(settings(), reference(embedded=True), charges())
    result = provider.evaluate(qm(), environment())
    assert result.diagnostics["selected_root"] == 2
    np.testing.assert_allclose(densities[0], np.diag([1.8, 0.2]))
    np.testing.assert_allclose(result.qm_forces_ev_per_angstrom, -1)
    np.testing.assert_allclose(result.environment_forces_ev_per_angstrom, -15)
    energy_only = provider.evaluate_surface(qm(), environment(), forces=False)
    assert energy_only.environment_forces_ev_per_angstrom is None
    assert len(densities) == 1


def test_fixed_reference_is_stateless_under_order_and_restart(monkeypatch):
    install_synthetic_solver(monkeypatch)
    pinned = reference()
    provider = excited.PySCFTDABackend(settings(), pinned)
    displaced = Geometry(qm().atom_ids, qm().symbols, qm().positions + 0.02)
    first = provider.evaluate_surface(qm(), forces=False)
    provider.evaluate_surface(displaced, forces=False)
    again = provider.evaluate_surface(qm(), forces=False)
    restarted = excited.PySCFTDABackend(
        settings(), excited.TDAReference.from_record(pinned.to_record())
    ).evaluate_surface(qm(), forces=False)
    assert first.energy_ev == again.energy_ev == restarted.energy_ev
    assert dict(first.diagnostics) == dict(again.diagnostics) == dict(restarted.diagnostics)
    assert provider.reference.sha256 == pinned.sha256


def test_reference_build_records_seed_and_total_state_identity(monkeypatch):
    install_synthetic_solver(monkeypatch)
    pinned = excited.build_tda_reference(
        qm(),
        settings(),
        initial_root=1,
        label="test",
        environment_geometry=environment(),
        charges=charges(),
    )
    assert pinned.seed_root == 1
    assert pinned.excitation_energy_ev == 2
    assert pinned.reference_gap_ev == 1
    assert pinned.charges.source == charges().source
    assert excited.TDAReference.from_record(pinned.to_record()).sha256 == pinned.sha256


def test_fixed_identity_settings_embedding_and_boundary_guards(monkeypatch):
    install_synthetic_solver(monkeypatch)
    with pytest.raises(ValueError, match="settings differ"):
        excited.PySCFTDABackend(replace(settings(), basis="sto-3g"), reference())
    with pytest.raises(ValueError, match="isolated reference"):
        excited.PySCFTDABackend(settings(), reference(embedded=True))
    provider = excited.PySCFTDABackend(settings(), reference())
    with pytest.raises(ValueError, match="membership"):
        provider.evaluate(Geometry((2, 1), ("H", "H"), qm().positions))
    with pytest.raises(ValueError, match="isolated QM"):
        provider.evaluate(Geometry((1, 2), ("H", "H"), qm().positions, np.eye(3)))
    different_charges = FixedEmbeddingCharges((10, 11), [0.2, -0.2], "changed")
    with pytest.raises(ValueError, match="provenance differ"):
        excited.PySCFTDAElectrostaticBackend(
            settings(), reference(embedded=True), different_charges
        )
    embedded = excited.PySCFTDAElectrostaticBackend(settings(), reference(embedded=True), charges())
    with pytest.raises(ValueError, match="Environment identity"):
        embedded.evaluate(qm(), Geometry((10, 11), ("H", "H"), environment().positions))
    record = copy.deepcopy(reference().to_record())
    record["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        excited.TDAReference.from_record(record)


def test_gradient_hook_checks_response_before_and_after_and_preserves_official_gradient():
    calls = []
    hook = response_hook()

    def response(vector):
        calls.append(vector.copy())
        return np.zeros_like(vector)

    hook["fvind"] = response

    class OfficialGradient:
        def __init__(self, td):
            self.base = td

        def solve_cphf(self, fvind, mo_energy, mo_occ, h1):
            return -h1, None

        def extra_force(self, atom_id, envs):
            return np.full(3, 0.25)

        def kernel(self, state):
            assert state == 2
            assert self.cphf_max_cycle == settings().cphf_max_cycle
            hook["z1"] = self.solve_cphf(
                hook["fvind"], hook["mo_energy"], np.array([2, 0]), hook["wvo"]
            )[0]
            return np.array([self.extra_force(atom, hook) for atom in range(2)])

    solved = SimpleNamespace(
        td=object(),
        overlap=np.eye(2),
        molecule=SimpleNamespace(nelectron=2, natm=2),
    )
    runtime = SimpleNamespace(tdrhf=SimpleNamespace(Gradients=OfficialGradient))
    gradient, density, diagnostic = excited._total_gradient(solved, 1, settings(), runtime)
    np.testing.assert_allclose(gradient, 0.25)
    np.testing.assert_allclose(density, [[1.8, 0.2], [0.2, 0.2]])
    assert diagnostic["cphf_residual_max"] == 0
    assert len(calls) == 2
    assert diagnostic["cphf_refinement_iterations"] == 0


def test_gradient_hook_rejects_missing_or_changing_response():
    class BrokenGradient:
        def __init__(self, td):
            pass

        def solve_cphf(self, fvind, mo_energy, mo_occ, h1):
            return -h1, None

        def extra_force(self, atom_id, envs):
            return 0

        def kernel(self, state):
            hook = response_hook()
            hook["z1"] = self.solve_cphf(
                hook["fvind"], hook["mo_energy"], np.array([2, 0]), hook["wvo"]
            )[0].copy()
            self.extra_force(0, hook)
            hook["z1"][0, 0] += 0.1
            self.extra_force(1, hook)
            return np.zeros((2, 3))

    solved = SimpleNamespace(
        td=object(),
        overlap=np.eye(2),
        molecule=SimpleNamespace(nelectron=2, natm=2),
    )
    runtime = SimpleNamespace(tdrhf=SimpleNamespace(Gradients=BrokenGradient))
    with pytest.raises(EvaluationError, match="checked orbital response"):
        excited._total_gradient(solved, 0, settings(), runtime)


def refine_with(solver, configuration=None):
    return excited._refine_cphf(
        solver,
        lambda vector: np.zeros_like(vector),
        np.array([0.0, 1.0]),
        np.array([2, 0]),
        np.array([[-1.0]]),
        settings() if configuration is None else configuration,
    )


def test_cphf_refinement_preserves_an_already_accepted_response():
    calls = []

    def solve(fvind, energy, occupation, rhs):
        calls.append(rhs.copy())
        return -rhs, None

    pair, diagnostic = refine_with(solve)
    np.testing.assert_array_equal(pair[0], [[1.0]])
    assert pair[1] is None
    assert len(calls) == 1
    assert diagnostic["cphf_refinement_iterations"] == 0
    assert diagnostic["cphf_residual_history"] == (0.0,)
    assert diagnostic["cphf_residual_tolerance"] == 1e-7


def test_normalized_cphf_defect_correction_has_the_correct_sign_and_scale():
    calls = []

    def solve(fvind, energy, occupation, rhs):
        calls.append(rhs.copy())
        if len(calls) == 1:
            return np.array([[1.0 + 2e-7]]), None
        # A tiny unnormalized residual would be lost in this synthetic floor.
        if np.max(np.abs(rhs)) < 1e-6:
            return np.zeros_like(rhs), None
        return -rhs, None

    pair, diagnostic = refine_with(solve)
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1], [[1.0]], atol=0, rtol=0)
    np.testing.assert_allclose(pair[0], [[1.0]], atol=1e-15, rtol=0)
    assert diagnostic["cphf_refinement_iterations"] == 1
    assert diagnostic["cphf_residual_before_refinement"] > 1e-7
    assert diagnostic["cphf_residual_after_refinement"] <= 1e-15
    assert diagnostic["cphf_residual_tolerance"] == 1e-7


def test_cphf_refinement_is_bounded_and_records_each_strictly_decreasing_residual():
    for initial_error, should_pass in ((3e-7, True), (1e-3, False)):
        calls = []

        def solve(fvind, energy, occupation, rhs, calls=calls, initial_error=initial_error):
            calls.append(rhs.copy())
            if len(calls) == 1:
                return np.array([[1.0 + initial_error]]), None
            return -0.5 * rhs, None

        if should_pass:
            _, diagnostic = refine_with(solve)
            assert diagnostic["cphf_refinement_iterations"] == 2
            history = diagnostic["cphf_residual_history"]
            assert len(history) == 3 and history[0] > history[1] > history[2]
            assert history[-1] <= settings().response_residual_tol
        else:
            with pytest.raises(EvaluationError, match="two bounded refinements"):
                refine_with(solve)
        assert len(calls) == 3


def test_cphf_refinement_rejects_no_decrease_and_nonfinite_solutions():
    for bad in (np.zeros((1, 1)), np.full((1, 1), np.nan), np.full((1, 1), np.inf)):
        calls = []

        def solve(fvind, energy, occupation, rhs, calls=calls, bad=bad):
            calls.append(rhs.copy())
            return (np.array([[1.0 + 2e-7]]) if len(calls) == 1 else bad), None

        with pytest.raises(EvaluationError, match="strictly decrease|finite"):
            refine_with(solve)
        assert len(calls) == 2

    with pytest.raises(EvaluationError, match="response pair"):
        refine_with(lambda *args: (np.ones((1, 1)), np.ones(1)))


def test_refined_response_precedes_gradient_and_relaxed_density_construction():
    solver_calls = []

    class OfficialGradient:
        def __init__(self, td):
            pass

        def solve_cphf(self, fvind, energy, occupation, rhs):
            solver_calls.append(rhs.copy())
            return (
                np.array([[1 + 2e-7]]) if len(solver_calls) == 1 else -rhs,
                None,
            )

        def extra_force(self, atom_id, envs):
            return 0.0

        def kernel(self, state):
            hook = response_hook()
            hook["wvo"] = np.array([[-1.0]])
            # This is the source hook location: all response-dependent
            # intermediates below use the result returned by solve_cphf.
            z = self.solve_cphf(hook["fvind"], hook["mo_energy"], np.array([2, 0]), hook["wvo"])[0]
            hook["z1"] = z
            hook["dmz1doo"] = np.array([[0, float(z[0, 0])], [0, 0]])
            for atom in range(2):
                self.extra_force(atom, hook)
            return np.full((2, 3), z[0, 0])

    solved = SimpleNamespace(
        td=object(),
        overlap=np.eye(2),
        molecule=SimpleNamespace(nelectron=2, natm=2),
    )
    runtime = SimpleNamespace(tdrhf=SimpleNamespace(Gradients=OfficialGradient))
    gradient, density, diagnostic = excited._total_gradient(solved, 0, settings(), runtime)
    np.testing.assert_allclose(gradient, 1.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(density, [[2, 0.5], [0.5, 0]], atol=1e-15, rtol=0)
    assert diagnostic["cphf_refinement_iterations"] == 1
    assert diagnostic["cphf_residual_before_refinement"] > 1e-7
    assert diagnostic["cphf_residual_max"] <= 1e-15


@pytest.mark.parametrize(
    "changes",
    [
        {"tda_solver": "automatic"},
        {"tda_dense_max_dimension": 1025},
        {"tda_dense_max_dimension": 0},
        {"tda_dense_max_dimension": True},
    ],
)
def test_dense_solver_setting_controls(changes):
    with pytest.raises(ValueError):
        settings(**changes)


def dense_fixture(matrix):
    matrix = np.asarray(matrix)
    dimension = len(matrix)
    calls = []

    def vind(vectors):
        calls.append(vectors.copy())
        return vectors @ matrix.T

    td = SimpleNamespace(
        _scf=SimpleNamespace(mo_occ=np.array([2] + [0] * dimension)),
        positive_eig_threshold=1e-3,
        gen_vind=lambda: (vind, None),
    )
    return td, calls


def test_dense_tda_sorts_official_operator_roots_and_preserves_restricted_normalization():
    td, calls = dense_fixture([[0.3, 0.04], [0.04, 0.3]])
    diagnostic = excited._dense_tda(td, settings(tda_solver="dense"))
    np.testing.assert_allclose(td.e, [0.26, 0.34], atol=1e-15, rtol=0)
    assert td.converged.tolist() == [True, True]
    for x, y in td.xy:
        assert y == 0
        assert 2 * np.sum(x * x) == pytest.approx(1)
    assert diagnostic["tda_solver"] == "dense"
    assert diagnostic["tda_dimension"] == 2
    assert max(diagnostic["tda_matrix_residuals_hartree"]) < 1e-14
    assert calls[0].shape == (2, 2)


def test_dense_tda_cap_precedes_all_operator_actions_and_batches_are_bounded():
    td, calls = dense_fixture(np.eye(3))
    with pytest.raises(EvaluationError, match="declared cap"):
        excited._dense_tda(td, settings(tda_solver="dense", tda_dense_max_dimension=2))
    assert calls == []
    td, calls = dense_fixture(np.diag(np.linspace(0.1, 0.5, 40)))
    excited._dense_tda(td, settings(tda_solver="dense"))
    assert [len(batch) for batch in calls[:-1]] == [16, 16, 8]
    assert len(calls[-1]) == 2  # Independent selected-root operator check.


@pytest.mark.parametrize(
    "matrix,message",
    [
        ([[0.2, 0.1], [0, 0.3]], "Hermiticity"),
        ([[0.2, 0], [0, float("nan")]], "finite"),
        ([[-0.2, 0], [0, 0.0001]], "too few roots"),
    ],
)
def test_dense_tda_rejects_invalid_operator_or_spectrum(matrix, message):
    td, _ = dense_fixture(matrix)
    with pytest.raises(EvaluationError, match=message):
        excited._dense_tda(td, settings(tda_solver="dense"))


def test_eigensystem_check_uses_normalized_residual_not_restricted_amplitude_scale():
    vector = np.array([[1 / np.sqrt(2), 0]])
    with pytest.raises(EvaluationError, match="original tolerance"):
        excited._eigensystem_residuals(
            lambda vectors: vectors * (0.2 + 1.1e-10),
            np.array([0.2]),
            vector,
            1e-10,
        )
    with pytest.raises(EvaluationError, match="wrong shape"):
        excited._eigensystem_residuals(
            lambda vectors: np.zeros((1, 1)), np.array([0.2]), vector, 1e-10
        )


def test_true_iterative_convergence_flags_cannot_override_actual_eigenpair_residuals():
    method, td, runtime = candidate_objects()
    td.gen_vind = lambda: (lambda vectors: vectors * np.array([0.21, 0.31]), None)
    with pytest.raises(EvaluationError, match="eigensystem residual"):
        excited._candidate_data(method, td, settings(), runtime)


def test_iterative_failure_does_not_trigger_a_dense_fallback(monkeypatch):
    method, td, runtime = candidate_objects()
    molecule = SimpleNamespace(intor_symmetric=lambda name: np.eye(3))
    method.kernel = lambda: -1.0
    method.converged = True
    method.TDA = lambda: td

    def fail_kernel():
        raise RuntimeError("declared iterative failure")

    td.kernel = fail_kernel
    runtime.scf = SimpleNamespace(RHF=lambda mol: method)
    monkeypatch.setattr(excited, "_make_molecule", lambda *args: molecule)
    monkeypatch.setattr(
        excited,
        "_dense_tda",
        lambda *args: pytest.fail("An implicit dense fallback must never run"),
    )
    with pytest.raises(RuntimeError, match="declared iterative failure"):
        excited._solve(qm(), settings(), runtime)


def surface_arguments(*, embedded=False, forces=True):
    """A standalone same-call ledger, independent of a quantum engine."""
    return {
        "energy_ev": -8.0,
        "qm_atom_ids": [1, 2],
        "environment_atom_ids": [10, 11] if embedded else [],
        "qm_forces_ev_per_angstrom": np.ones((2, 3)) if forces else None,
        "environment_forces_ev_per_angstrom": (
            np.full((2, 3), -1.0) if embedded and forces else None
        ),
        "diagnostics": {
            "method": "cpu-rhf-tda-singlet",
            "energy_definition": "E_RHF+omega",
            "ground_energy_ev": -10.0,
            "excitation_energy_ev": 2.0,
            "selected_root": 1,
            "candidate_excitation_energies_ev": [2.0, 3.0],
            "scf_converged": True,
            "candidate_roots_converged": True,
            "forces_evaluated": forces,
        },
    }


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("forces", [False, True])
def test_standalone_surface_preserves_total_energy_and_derivative_scope(embedded, forces):
    arguments = surface_arguments(embedded=embedded, forces=forces)
    surface = excited.TDASurfaceEvaluation(**arguments)
    assert surface.energy_ev == -8.0
    assert surface.ground_energy_ev == -10.0
    assert surface.excitation_energy_ev == 2.0
    assert surface.qm_atom_ids == (1, 2)
    assert surface.environment_atom_ids == ((10, 11) if embedded else ())
    assert (surface.qm_forces_ev_per_angstrom is not None) is forces
    assert (surface.environment_forces_ev_per_angstrom is not None) is (embedded and forces)
    arguments["qm_atom_ids"][0] = 99
    arguments["diagnostics"]["candidate_excitation_energies_ev"][0] = 99.0
    arguments["diagnostics"]["ground_energy_ev"] = 99.0
    assert surface.qm_atom_ids == (1, 2)
    assert surface.diagnostics["candidate_excitation_energies_ev"] == (2.0, 3.0)
    assert surface.ground_energy_ev == -10.0
    if forces:
        arguments["qm_forces_ev_per_angstrom"][:] = 99.0
        np.testing.assert_array_equal(surface.qm_forces_ev_per_angstrom, np.ones((2, 3)))
        with pytest.raises(ValueError):
            surface.qm_forces_ev_per_angstrom.flags.writeable = True


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy_ev", 2.0),  # An excitation gap is not the potential energy.
        ("energy_ev", np.nan),
        ("energy_ev", [-8.0]),
        ("energy_ev", True),
        ("qm_atom_ids", []),
        ("qm_atom_ids", [1, 1]),
        ("qm_atom_ids", [True, 2]),
        ("environment_atom_ids", [10, 10]),
        ("environment_atom_ids", [1, 10]),
        ("environment_atom_ids", [-1, 10]),
        ("qm_forces_ev_per_angstrom", [[1, 2]]),
        ("qm_forces_ev_per_angstrom", np.full((2, 3), np.inf)),
        ("qm_forces_ev_per_angstrom", np.ones((2, 3), dtype=complex)),
        ("environment_forces_ev_per_angstrom", np.zeros((1, 3))),
        ("environment_forces_ev_per_angstrom", np.full((2, 3), np.nan)),
        ("environment_forces_ev_per_angstrom", np.ones((2, 3), dtype=bool)),
        ("qm_forces_ev_per_angstrom", None),
        ("environment_forces_ev_per_angstrom", None),
        ("diagnostics", None),
        ("diagnostics", []),
    ],
)
def test_standalone_surface_rejects_invalid_values_and_missing_derivatives(field, value):
    arguments = surface_arguments(embedded=True)
    arguments[field] = value
    with pytest.raises(EvaluationError):
        excited.TDASurfaceEvaluation(**arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "cpu-rks-tda-singlet"),
        ("energy_definition", "omega"),
        ("ground_energy_ev", np.nan),
        ("ground_energy_ev", [-10.0]),
        ("ground_energy_ev", True),
        ("excitation_energy_ev", -2.0),
        ("excitation_energy_ev", 3.0),
        ("excitation_energy_ev", np.inf),
        ("selected_root", True),
        ("selected_root", 0),
        ("selected_root", 3),
        ("selected_root", 2),
        ("candidate_excitation_energies_ev", [2.0]),
        ("candidate_excitation_energies_ev", [[2.0, 3.0]]),
        ("candidate_excitation_energies_ev", [2.0, -3.0]),
        ("candidate_excitation_energies_ev", [2.0, np.nan]),
        ("scf_converged", False),
        ("candidate_roots_converged", False),
        ("forces_evaluated", False),
        ("forces_evaluated", 1),
    ],
)
def test_standalone_surface_rejects_inconsistent_state_ledger(field, value):
    arguments = surface_arguments()
    arguments["diagnostics"][field] = value
    with pytest.raises(EvaluationError):
        excited.TDASurfaceEvaluation(**arguments)


@pytest.mark.parametrize("field", tuple(surface_arguments()["diagnostics"]))
def test_standalone_surface_requires_each_energy_and_derivative_field(field):
    arguments = surface_arguments()
    del arguments["diagnostics"][field]
    with pytest.raises(EvaluationError, match="omit required"):
        excited.TDASurfaceEvaluation(**arguments)


def test_isolated_surface_rejects_spurious_environment_derivative():
    arguments = surface_arguments()
    arguments["environment_forces_ev_per_angstrom"] = np.empty((0, 3))
    with pytest.raises(EvaluationError, match="Isolated"):
        excited.TDASurfaceEvaluation(**arguments)


def test_energy_only_surface_rejects_claimed_force_evaluation():
    arguments = surface_arguments(embedded=True, forces=False)
    arguments["diagnostics"]["forces_evaluated"] = True
    with pytest.raises(EvaluationError, match="availability"):
        excited.TDASurfaceEvaluation(**arguments)


def test_dense_larger_resource_cap_is_explicit_and_reference_bound():
    assert excited.TDASettings().tda_dense_max_dimension == 256
    large = settings(tda_solver="dense", tda_dense_max_dimension=1024)
    pinned = reference(configuration=large)
    loaded = excited.TDAReference.from_record(pinned.to_record())
    assert loaded.settings.tda_dense_max_dimension == 1024
    assert loaded.sha256 != reference(configuration=settings(tda_solver="dense")).sha256
    assert large.tda_conv_tol == settings().tda_conv_tol
    assert large.response_residual_tol == settings().response_residual_tol
