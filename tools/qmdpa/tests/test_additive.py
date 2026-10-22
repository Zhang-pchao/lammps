# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Analytic contract tests; fixtures do not represent a fitted hybrid potential."""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from qmdpa import AdditiveCalculator, CrossLennardJones
from qmdpa.core import Evaluation, EvaluationError, Geometry
from qmdpa.embedding import EmbeddedEvaluation


def geometry():
    return Geometry(
        (10000000001, 7, 30, 4, 12, 18, 5, 22, 26),
        ("O", "H", "H", "O", "H", "H", "O", "H", "H"),
        [
            [0.1, 0.3, 0.2],
            [1.03, 0.1, 0.0],
            [-0.12, 1.21, 0.3],
            [3.2, 0.7, 0.4],
            [4.1, 0.95, 0.5],
            [2.88, 1.56, 0.1],
            [-1.0, -2.9, 1.2],
            [-0.13, -3.12, 1.0],
            [-1.27, -2.02, 1.4],
        ],
    )


QM_IDS = (30, 10000000001, 7)
ENVIRONMENT_IDS = (18, 4, 26, 5, 12, 22)
PARAMETERS = {
    ("O", "O"): (0.02, 2.1),
    ("O", "H"): (0.011, 1.4),
    ("H", "O"): (0.008, 1.6),
    ("H", "H"): (0.004, 1.0),
}


def quadratic(positions, stiffness):
    energy = 0.0
    forces = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i):
            delta = positions[i] - positions[j]
            energy += stiffness * np.dot(delta, delta) / 2
            forces[i] -= stiffness * delta
            forces[j] += stiffness * delta
    return energy, forces


class EnvironmentMock:
    def __init__(self, stiffness=0.13):
        self.stiffness = stiffness
        self.inputs = []

    def evaluate(self, atoms):
        self.inputs.append(atoms)
        return Evaluation(*quadratic(atoms.positions, self.stiffness))


class EmbeddedMock:
    def __init__(self, interaction=0.07):
        self.interaction = interaction
        self.inputs = []

    def evaluate(self, qm, environment):
        self.inputs.append((qm, environment))
        energy, qm_forces = quadratic(qm.positions, 0.19)
        environment_forces = np.zeros_like(environment.positions)
        for i, position in enumerate(qm.positions):
            for j, other in enumerate(environment.positions):
                delta = position - other
                energy += self.interaction * np.dot(delta, delta) / 2
                force = -self.interaction * delta
                qm_forces[i] += force
                environment_forces[j] -= force
        return EmbeddedEvaluation(
            energy, qm.atom_ids, environment.atom_ids, qm_forces, environment_forces
        )


def cross(epsilon_scale=1):
    return CrossLennardJones(
        {key: (epsilon * epsilon_scale, sigma) for key, (epsilon, sigma) in PARAMETERS.items()},
        source="Explicit analytic unit-test parameters; not a physical model",
    )


def calculator(environment=None, embedded=None, cross_backend=None):
    return AdditiveCalculator(
        EnvironmentMock() if environment is None else environment,
        EmbeddedMock() if embedded is None else embedded,
        cross() if cross_backend is None else cross_backend,
        qm_atom_ids=QM_IDS,
        environment_atom_ids=ENVIRONMENT_IDS,
    )


def finite_difference(atoms, energy, step):
    forces = np.empty_like(atoms.positions)
    for atom in range(len(atoms.atom_ids)):
        for axis in range(3):
            values = []
            for sign in (1, -1):
                positions = atoms.positions.copy()
                positions[atom, axis] += sign * step
                values.append(energy(Geometry(atoms.atom_ids, atoms.symbols, positions)))
            forces[atom, axis] = -(values[0] - values[1]) / (2 * step)
    return forces


class CrossLennardJonesTests(unittest.TestCase):
    def test_known_zero_and_minimum_for_a_single_pair(self):
        backend = CrossLennardJones({("O", "H"): (0.1, 2.0)}, source="analytic fixture")
        qm = Geometry((8,), ("O",), [[0, 0, 0]])
        for distance, energy, xforce in (
            (2.0, 0.0, -1.2),
            (2.0 * 2 ** (1 / 6), -0.1, 0.0),
        ):
            with self.subTest(distance=distance):
                environment = Geometry((3,), ("H",), [[distance, 0, 0]])
                result = backend.evaluate(qm, environment)
                self.assertAlmostEqual(result.energy_ev, energy, places=14)
                np.testing.assert_allclose(
                    result.qm_forces_ev_per_angstrom, [[xforce, 0, 0]], atol=1e-14
                )
                np.testing.assert_allclose(
                    result.environment_forces_ev_per_angstrom,
                    -result.qm_forces_ev_per_angstrom,
                    atol=1e-14,
                )

    def test_all_force_components_match_energy_derivatives(self):
        atoms = geometry()
        backend = cross()
        result = backend.evaluate(atoms.select(QM_IDS), atoms.select(ENVIRONMENT_IDS))
        expected = {
            atom_id: force
            for ids, forces in (
                (QM_IDS, result.qm_forces_ev_per_angstrom),
                (ENVIRONMENT_IDS, result.environment_forces_ev_per_angstrom),
            )
            for atom_id, force in zip(ids, forces)
        }
        force = np.array([expected[atom_id] for atom_id in atoms.atom_ids])
        energy = lambda value: (
            backend.evaluate(value.select(QM_IDS), value.select(ENVIRONMENT_IDS)).energy_ev
        )
        for step in (1e-4, 5e-5):
            with self.subTest(step=step):
                np.testing.assert_allclose(
                    finite_difference(atoms, energy, step), force, atol=2e-8, rtol=0
                )

    def test_reaction_force_translation_and_explicit_pair_diagnostics(self):
        atoms = geometry()
        backend = cross()
        qm, environment = atoms.select(QM_IDS), atoms.select(ENVIRONMENT_IDS)
        result = backend.evaluate(qm, environment)
        np.testing.assert_allclose(
            result.qm_forces_ev_per_angstrom.sum(axis=0)
            + result.environment_forces_ev_per_angstrom.sum(axis=0),
            0,
            atol=1e-15,
        )
        shifted = Geometry(atoms.atom_ids, atoms.symbols, atoms.positions + [2.1, -4.7, 0.8])
        translated = backend.evaluate(shifted.select(QM_IDS), shifted.select(ENVIRONMENT_IDS))
        self.assertAlmostEqual(result.energy_ev, translated.energy_ev, places=14)
        np.testing.assert_allclose(
            result.qm_forces_ev_per_angstrom, translated.qm_forces_ev_per_angstrom, atol=1e-14
        )
        self.assertEqual(result.diagnostics["pair_count"], 18)
        self.assertEqual(result.diagnostics["parameter_source"], backend.source)

    def test_directional_keys_have_no_implicit_reverse_or_mixing(self):
        qm = Geometry((1,), ("O",), [[0, 0, 0]])
        environment = Geometry((2,), ("H",), [[2, 0, 0]])
        for parameters in ({("H", "O"): (0.1, 2)}, {("O", "O"): (0.1, 2)}, {}):
            with (
                self.subTest(parameters=parameters),
                self.assertRaisesRegex(ValueError, "Missing explicit"),
            ):
                CrossLennardJones(parameters, source="fixture").evaluate(qm, environment)

    def test_invalid_parameters_and_sources(self):
        invalid = [
            None,
            [(("O", "H"), (0.1, 1))],
            {"OH": (0.1, 1)},
            {("O",): (0.1, 1)},
            {("O", ""): (0.1, 1)},
            {("O", "H"): 1},
            {("O", "H"): (1,)},
            {("O", "H"): (1, 2, 3)},
            {("O", "H"): (-0.1, 1)},
            {("O", "H"): (0.1, 0)},
            {("O", "H"): (0.1, -1)},
            {("O", "H"): (np.nan, 1)},
            {("O", "H"): (0.1, np.inf)},
            {("O", "H"): (True, 1)},
            {("O", "H"): (0.1, False)},
            {("O", "H"): ("0.1", 1)},
            {("O", "H"): (1j, 1)},
            {("O", "H"): ([0.1], 1)},
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises((TypeError, ValueError)):
                CrossLennardJones(parameters, source="fixture")
        for source in ("", " ", None, 3):
            with self.subTest(source=source), self.assertRaises(ValueError):
                CrossLennardJones(PARAMETERS, source=source)
        with self.assertRaises(TypeError):
            CrossLennardJones(PARAMETERS)

    def test_parameter_snapshot_and_source_are_immutable(self):
        values = [0.1, 2.0]
        parameters = {("O", "H"): values}
        backend = CrossLennardJones(parameters, source="fixture")
        values[0] = 8.0
        parameters.clear()
        self.assertEqual(backend.parameters[("O", "H")], (0.1, 2.0))
        with self.assertRaises(TypeError):
            backend.parameters[("O", "H")] = (9, 9)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            backend.source = "modified"

    def test_zero_epsilon_and_invalid_geometry(self):
        atoms = geometry()
        qm, environment = atoms.select(QM_IDS), atoms.select(ENVIRONMENT_IDS)
        result = cross(0).evaluate(qm, environment)
        self.assertEqual(result.energy_ev, 0)
        np.testing.assert_array_equal(result.qm_forces_ev_per_angstrom, np.zeros((3, 3)))
        np.testing.assert_array_equal(result.environment_forces_ev_per_angstrom, np.zeros((6, 3)))
        periodic = dataclasses.replace(qm, cell=np.eye(3) * 20)
        for first, second, pattern in (
            (periodic, environment, "nonperiodic"),
            (qm, dataclasses.replace(environment, cell=np.eye(3) * 20), "nonperiodic"),
            (qm, qm, "disjoint"),
            (qm, Geometry((99,), ("H",), qm.positions[:1]), "coincide"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                cross().evaluate(first, second)

    def test_floating_point_overflow_fails_without_a_nonfinite_result(self):
        backend = CrossLennardJones({("O", "H"): (1, 1e100)}, source="invalid numerical scale")
        with self.assertRaisesRegex(EvaluationError, "finite numerical"):
            backend.evaluate(
                Geometry((1,), ("O",), [[0, 0, 0]]),
                Geometry((2,), ("H",), [[1, 0, 0]]),
            )


class AdditiveTests(unittest.TestCase):
    def test_terms_are_evaluated_once_and_assembled_in_full_atom_order(self):
        atoms = geometry()
        environment = EnvironmentMock()
        embedded = EmbeddedMock()
        cross_backend = cross()
        with patch.object(CrossLennardJones, "evaluate", wraps=cross_backend.evaluate) as observed:
            calc = calculator(environment, embedded, cross_backend)
            terms = calc.evaluate_terms(atoms)
        self.assertEqual(len(environment.inputs), 1)
        self.assertEqual(len(embedded.inputs), 1)
        self.assertEqual(observed.call_count, 1)
        self.assertEqual(environment.inputs[0].atom_ids, ENVIRONMENT_IDS)
        self.assertEqual(embedded.inputs[0][0].atom_ids, QM_IDS)
        self.assertEqual(embedded.inputs[0][1].atom_ids, ENVIRONMENT_IDS)
        self.assertEqual(
            (terms.atom_ids, terms.qm_atom_ids, terms.environment_atom_ids),
            (atoms.atom_ids, QM_IDS, ENVIRONMENT_IDS),
        )
        self.assertAlmostEqual(
            terms.total.energy,
            terms.environment.energy + terms.embedded.energy_ev + terms.cross.energy_ev,
        )
        assembled = {}
        for index, atom_id in enumerate(QM_IDS):
            assembled[atom_id] = (
                terms.embedded.qm_forces_ev_per_angstrom[index]
                + terms.cross.qm_forces_ev_per_angstrom[index]
            )
        for index, atom_id in enumerate(ENVIRONMENT_IDS):
            assembled[atom_id] = (
                terms.environment.forces[index]
                + terms.embedded.environment_forces_ev_per_angstrom[index]
                + terms.cross.environment_forces_ev_per_angstrom[index]
            )
        np.testing.assert_array_equal(
            terms.total.forces, [assembled[atom_id] for atom_id in atoms.atom_ids]
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            terms.total = None
        with self.assertRaises(ValueError):
            terms.total.forces.setflags(write=True)

    def test_evaluate_returns_the_terms_path_total(self):
        calc = calculator()
        sentinel = Evaluation(1.0, np.zeros((9, 3)))
        with patch.object(
            calc, "evaluate_terms", return_value=SimpleNamespace(total=sentinel)
        ) as call:
            self.assertIs(calc.evaluate(geometry()), sentinel)
        call.assert_called_once()

    def test_all_27_components_match_total_energy_derivatives_at_two_steps(self):
        atoms = geometry()
        calc = calculator()
        result = calc.evaluate(atoms)
        for step in (1e-4, 5e-5):
            with self.subTest(step=step):
                fd = finite_difference(atoms, lambda value: calc.evaluate(value).energy, step)
                np.testing.assert_allclose(result.forces, fd, atol=2e-8, rtol=0)

    def test_full_input_reorder_preserves_identity_and_canonical_provider_orders(self):
        atoms = geometry()
        environment, embedded = EnvironmentMock(), EmbeddedMock()
        calc = calculator(environment, embedded)
        original = calc.evaluate(atoms)
        reordered = atoms.select(atoms.atom_ids[::-1])
        result = calc.evaluate(reordered)
        self.assertEqual(original.energy, result.energy)
        np.testing.assert_array_equal(original.forces[::-1], result.forces)
        self.assertTrue(all(value.atom_ids == ENVIRONMENT_IDS for value in environment.inputs))
        self.assertTrue(all(qm.atom_ids == QM_IDS for qm, _ in embedded.inputs))
        self.assertTrue(all(env.atom_ids == ENVIRONMENT_IDS for _, env in embedded.inputs))

    def test_total_reaction_force_and_translation(self):
        atoms = geometry()
        calc = calculator()
        original = calc.evaluate(atoms)
        shifted = dataclasses.replace(atoms, positions=atoms.positions + [3.2, -1.7, 2.6])
        translated = calc.evaluate(shifted)
        np.testing.assert_allclose(original.forces.sum(axis=0), 0, atol=2e-14)
        self.assertAlmostEqual(original.energy, translated.energy, places=12)
        np.testing.assert_allclose(original.forces, translated.forces, atol=2e-14)

    def test_zero_cross_and_embedding_coupling_decouple_without_extra_coulomb(self):
        atoms = geometry()
        calc = calculator(embedded=EmbeddedMock(interaction=0), cross_backend=cross(0))
        terms = calc.evaluate_terms(atoms)
        expected_qm_energy, expected_qm_forces = quadratic(atoms.select(QM_IDS).positions, 0.19)
        expected_env = EnvironmentMock().evaluate(atoms.select(ENVIRONMENT_IDS))
        self.assertEqual(terms.total.energy, expected_qm_energy + expected_env.energy)
        lookup = {atom_id: i for i, atom_id in enumerate(atoms.atom_ids)}
        np.testing.assert_array_equal(
            terms.total.forces[[lookup[i] for i in QM_IDS]], expected_qm_forces
        )
        np.testing.assert_array_equal(
            terms.total.forces[[lookup[i] for i in ENVIRONMENT_IDS]], expected_env.forces
        )
        np.testing.assert_array_equal(
            terms.embedded.environment_forces_ev_per_angstrom, np.zeros((6, 3))
        )
        self.assertEqual(terms.cross.energy_ev, 0)

    def test_bad_region_membership_and_ids_are_rejected(self):
        for qm_ids, env_ids in (
            ((), ENVIRONMENT_IDS),
            (QM_IDS, ()),
            ((7, 7), ENVIRONMENT_IDS),
            ((True,), ENVIRONMENT_IDS),
            ((1.0,), ENVIRONMENT_IDS),
            ((-1,), ENVIRONMENT_IDS),
            (QM_IDS, (7, 4)),
        ):
            with self.subTest(qm_ids=qm_ids, env_ids=env_ids), self.assertRaises(ValueError):
                AdditiveCalculator(
                    None, None, None, qm_atom_ids=qm_ids, environment_atom_ids=env_ids
                )
        atoms = geometry()
        calc = calculator()
        extra = Geometry(
            atoms.atom_ids + (99,), atoms.symbols + ("H",), np.vstack((atoms.positions, [9, 9, 9]))
        )
        for invalid in (extra, atoms.select(atoms.atom_ids[:-1])):
            with self.assertRaisesRegex(ValueError, "cover all"):
                calc.evaluate(invalid)
        self.assertEqual(calc.environment_backend.inputs, [])

    def test_nonperiodic_and_fixed_symbol_contract_precedes_provider_calls(self):
        atoms = geometry()
        calc = calculator()
        with self.assertRaisesRegex(ValueError, "nonperiodic"):
            calc.evaluate(dataclasses.replace(atoms, cell=np.eye(3) * 20))
        self.assertEqual(calc.environment_backend.inputs, [])
        calc.evaluate(atoms)
        changed = dataclasses.replace(atoms, symbols=("N",) + atoms.symbols[1:])
        with self.assertRaisesRegex(ValueError, "symbols"):
            calc.evaluate(changed)
        self.assertEqual(len(calc.environment_backend.inputs), 1)
        with self.assertRaises(AttributeError):
            calc.qm_atom_ids = (99,)

    def test_bad_environment_result_or_force_count_fails(self):
        for result in (object(), Evaluation(0, np.zeros((1, 3)))):
            with self.subTest(result=result), self.assertRaises(EvaluationError):
                calculator(
                    environment=SimpleNamespace(evaluate=lambda _, result=result: result)
                ).evaluate(geometry())

    def test_both_embedded_style_backends_require_exact_ids_types_and_shapes(self):
        atoms = geometry()
        valid = EmbeddedMock().evaluate(atoms.select(QM_IDS), atoms.select(ENVIRONMENT_IDS))
        bad_shape = dataclasses.replace(valid)
        object.__setattr__(bad_shape, "environment_forces_ev_per_angstrom", np.zeros((1, 3)))
        bad_qm_shape = dataclasses.replace(valid)
        object.__setattr__(bad_qm_shape, "qm_forces_ev_per_angstrom", np.zeros((1, 3)))
        results = (
            object(),
            dataclasses.replace(valid, qm_atom_ids=QM_IDS[::-1]),
            dataclasses.replace(valid, environment_atom_ids=ENVIRONMENT_IDS[::-1]),
            dataclasses.replace(valid, qm_atom_ids=(99,) + QM_IDS[1:]),
            bad_shape,
            bad_qm_shape,
        )
        for field in ("embedded", "cross_backend"):
            for result in results:
                with self.subTest(field=field, result=result), self.assertRaises(EvaluationError):
                    backend = SimpleNamespace(evaluate=lambda *_, result=result: result)
                    calculator(**{field: backend}).evaluate(atoms)

    def test_provider_exception_propagates_without_retry_or_stale_total(self):
        atoms = geometry()
        for field in ("environment", "embedded", "cross_backend"):
            with self.subTest(field=field):
                failed = EvaluationError("deliberate provider failure")
                backend = SimpleNamespace(evaluate=lambda *_: None)
                calc = calculator(**{field: backend})
                with patch.object(backend, "evaluate", side_effect=failed) as call:
                    for _ in range(2):
                        with self.assertRaises(EvaluationError) as captured:
                            calc.evaluate(atoms)
                        self.assertIs(captured.exception, failed)
                    self.assertEqual(call.call_count, 2)


if __name__ == "__main__":
    unittest.main()
