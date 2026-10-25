# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Analytic model/derivative controls; these do not establish DPA accuracy."""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.embedding import EmbeddedEvaluation, FixedEmbeddingCharges
from qmdpa.reference import (
    COULOMB_EV_ANGSTROM,
    COULOMB_SOURCE,
    FixedChargeCrossCoulomb,
    ReferenceSubtractiveCalculator,
)

QM_IDS = (30, 10000000001)
ENV_IDS = (8, 7)


def geometry():
    return Geometry(
        (10000000001, 7, 30, 8),
        ("O", "H", "H", "O"),
        [[0.1, 0.2, 0.3], [3.2, 1.1, -0.2], [1.05, 0.25, 0.1], [3.0, 0.1, 0.3]],
    )


def reference(scale=1):
    return FixedChargeCrossCoulomb(
        FixedEmbeddingCharges(QM_IDS, [0.4 * scale, -0.8 * scale], source="analytic Q"),
        FixedEmbeddingCharges(ENV_IDS, [-0.8, 0.4], source="analytic M"),
    )


def pair_quadratic(positions, stiffness):
    energy = 0.0
    forces = np.zeros_like(positions)
    for i, position in enumerate(positions):
        for j in range(i):
            delta = position - positions[j]
            energy += stiffness * np.dot(delta, delta) / 2
            forces[i] -= stiffness * delta
            forces[j] += stiffness * delta
    return energy, forces


class LowModel:
    def __init__(self, stiffness):
        self.stiffness = stiffness
        self.inputs = []

    def evaluate(self, atoms):
        self.inputs.append(atoms)
        return Evaluation(*pair_quadratic(atoms.positions, self.stiffness))


class EmbeddedModel:
    def __init__(self, interaction=0.07):
        self.interaction = interaction
        self.inputs = []

    def evaluate(self, qm, environment):
        self.inputs.append((qm, environment))
        energy, qforce = pair_quadratic(qm.positions, 0.21)
        mforce = np.zeros_like(environment.positions)
        for i, first in enumerate(qm.positions):
            for j, second in enumerate(environment.positions):
                delta = first - second
                energy += self.interaction * np.dot(delta, delta) / 2
                force = -self.interaction * delta
                qforce[i] += force
                mforce[j] -= force
        return EmbeddedEvaluation(energy, qm.atom_ids, environment.atom_ids, qforce, mforce)


def calculator(*, cross=None, embedded=None, full_low=None, region_low=None):
    return ReferenceSubtractiveCalculator(
        LowModel(0.13) if full_low is None else full_low,
        LowModel(0.11) if region_low is None else region_low,
        EmbeddedModel() if embedded is None else embedded,
        reference() if cross is None else cross,
        qm_atom_ids=QM_IDS,
        environment_atom_ids=ENV_IDS,
    )


def finite_difference(atoms, energy, step):
    force = np.empty_like(atoms.positions)
    for atom in range(len(atoms.atom_ids)):
        for axis in range(3):
            values = []
            for sign in (1, -1):
                positions = atoms.positions.copy()
                positions[atom, axis] += sign * step
                values.append(energy(Geometry(atoms.atom_ids, atoms.symbols, positions)))
            force[atom, axis] = -(values[0] - values[1]) / (2 * step)
    return force


def cross_force_in_full_order(result, atoms):
    by_id = dict(zip(result.qm_atom_ids, result.qm_forces_ev_per_angstrom)) | dict(
        zip(result.environment_atom_ids, result.environment_forces_ev_per_angstrom)
    )
    return np.array([by_id[atom_id] for atom_id in atoms.atom_ids])


class CoulombReferenceTests(unittest.TestCase):
    def test_known_attraction_and_repulsion(self):
        qm = Geometry((1,), ("H",), [[0, 0, 0]])
        env = Geometry((2,), ("H",), [[2, 0, 0]])
        for other_charge in (-0.5, 0.5, 0.0):
            with self.subTest(other_charge=other_charge):
                backend = FixedChargeCrossCoulomb(
                    FixedEmbeddingCharges((1,), [2.0], source="analytic"),
                    FixedEmbeddingCharges((2,), [other_charge], source="analytic"),
                )
                result = backend.evaluate(qm, env)
                self.assertAlmostEqual(result.energy_ev, COULOMB_EV_ANGSTROM * other_charge)
                np.testing.assert_allclose(
                    result.qm_forces_ev_per_angstrom,
                    [[-COULOMB_EV_ANGSTROM * other_charge / 2, 0, 0]],
                    atol=1e-14,
                )
                np.testing.assert_array_equal(
                    result.environment_forces_ev_per_angstrom,
                    -result.qm_forces_ev_per_angstrom,
                )
        expected_constant = 1.602176634e-19 / (4 * np.pi * 8.8541878128e-12) * 1e10
        self.assertAlmostEqual(COULOMB_EV_ANGSTROM, expected_constant, places=12)
        self.assertIn("CODATA 2018", COULOMB_SOURCE)

    def test_all_coordinate_derivatives_and_reaction_forces(self):
        atoms, backend = geometry(), reference()
        result = backend.evaluate(atoms.select(QM_IDS), atoms.select(ENV_IDS))
        analytic = cross_force_in_full_order(result, atoms)
        for step in (1e-4, 5e-5):
            with self.subTest(step=step):
                numerical = finite_difference(
                    atoms,
                    lambda g: backend.evaluate(g.select(QM_IDS), g.select(ENV_IDS)).energy_ev,
                    step,
                )
                np.testing.assert_allclose(numerical, analytic, atol=1e-7, rtol=0)
        np.testing.assert_allclose(analytic.sum(axis=0), 0, atol=1e-15, rtol=0)
        self.assertEqual(result.diagnostics["qm_charge_source"], "analytic Q")
        self.assertEqual(result.diagnostics["environment_charge_source"], "analytic M")
        self.assertEqual(result.diagnostics["pair_count"], 4)

    def test_translation_rotation_and_charge_snapshot(self):
        atoms = geometry()
        q_values = [0.4, -0.8]
        backend = FixedChargeCrossCoulomb(
            FixedEmbeddingCharges(QM_IDS, q_values, source="snapshot"),
            FixedEmbeddingCharges(ENV_IDS, [-0.8, 0.4], source="snapshot"),
        )
        q_values[0] = 99
        result = backend.evaluate(atoms.select(QM_IDS), atoms.select(ENV_IDS))
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        moved = Geometry(atoms.atom_ids, atoms.symbols, atoms.positions @ rotation + [3, -2, 1])
        transformed = backend.evaluate(moved.select(QM_IDS), moved.select(ENV_IDS))
        self.assertAlmostEqual(result.energy_ev, transformed.energy_ev, places=13)
        np.testing.assert_allclose(
            transformed.qm_forces_ev_per_angstrom,
            result.qm_forces_ev_per_angstrom @ rotation,
            atol=1e-13,
        )
        with self.assertRaises(ValueError):
            backend.qm_charges.charges_e.setflags(write=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            backend.qm_charges = None

    def test_rejects_unsupported_charge_models_and_geometry(self):
        q = FixedEmbeddingCharges(QM_IDS, [0.4, -0.8], source="fixture")
        env = FixedEmbeddingCharges(ENV_IDS, [-0.8, 0.4], source="fixture")
        for first, second in ((None, env), (q, None)):
            with self.subTest(first=first), self.assertRaises(TypeError):
                FixedChargeCrossCoulomb(first, second)
        gaussian = FixedEmbeddingCharges(
            QM_IDS, [0.4, -0.8], source="fixture", radii_angstrom=[1, 1]
        )
        with self.assertRaisesRegex(ValueError, "point charges"):
            FixedChargeCrossCoulomb(gaussian, env)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            FixedChargeCrossCoulomb(q, q)
        atoms, backend = geometry(), reference()
        qm, environment = atoms.select(QM_IDS), atoms.select(ENV_IDS)
        with self.assertRaisesRegex(ValueError, "in order"):
            backend.evaluate(qm.select(tuple(reversed(QM_IDS))), environment)
        with self.assertRaisesRegex(ValueError, "nonperiodic"):
            backend.evaluate(dataclasses.replace(qm, cell=np.eye(3) * 20), environment)
        positions = environment.positions.copy()
        positions[0] = qm.positions[0]
        with self.assertRaisesRegex(ValueError, "coincide"):
            reference(0).evaluate(qm, dataclasses.replace(environment, positions=positions))
        with self.assertRaises(ValueError):
            FixedEmbeddingCharges(QM_IDS, [np.nan, 0], source="fixture")
        with self.assertRaises(ValueError):
            FixedEmbeddingCharges(QM_IDS, [0, 0], source="")


class ReferenceSubtractionTests(unittest.TestCase):
    def test_decomposition_signs_both_force_sets_and_provider_orders(self):
        atoms, model = geometry(), calculator()
        terms = model.evaluate_terms(atoms)
        self.assertEqual(terms.model, "experimental-reference-subtractive")
        self.assertEqual(terms.atom_ids, atoms.atom_ids)
        self.assertEqual(model.full_low_backend.inputs[0].atom_ids, QM_IDS + ENV_IDS)
        self.assertEqual(model.region_low_backend.inputs[0].atom_ids, QM_IDS)
        self.assertEqual(model.embedded_backend.inputs[0][1].atom_ids, ENV_IDS)
        self.assertAlmostEqual(
            terms.total.energy,
            terms.full_low.energy
            + terms.embedded.energy_ev
            - terms.region_low.energy
            - terms.reference_cross.energy_ev,
            places=14,
        )
        expected = terms.full_low.forces.copy()
        qrows = [atoms.atom_ids.index(atom_id) for atom_id in QM_IDS]
        expected[qrows] -= terms.region_low.forces
        expected += cross_force_in_full_order(terms.embedded, atoms)
        expected -= cross_force_in_full_order(terms.reference_cross, atoms)
        np.testing.assert_allclose(terms.total.forces, expected, atol=1e-14)
        np.testing.assert_allclose(terms.total.forces.sum(axis=0), 0, atol=2e-15)
        for array in (terms.total.forces, terms.full_low.forces, terms.region_low.forces):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_total_energy_all_coordinate_derivatives(self):
        atoms, model = geometry(), calculator()
        result = model.evaluate(atoms)
        for step in (1e-4, 5e-5):
            with self.subTest(step=step):
                numerical = finite_difference(atoms, lambda g: model.evaluate(g).energy, step)
                np.testing.assert_allclose(result.forces, numerical, atol=1e-7, rtol=0)

    def test_reordering_preserves_results_and_fixed_provider_input(self):
        atoms, model = geometry(), calculator()
        first = model.evaluate_terms(atoms)
        reordered = atoms.select(tuple(reversed(atoms.atom_ids)))
        second = model.evaluate_terms(reordered)
        self.assertAlmostEqual(first.total.energy, second.total.energy, places=14)
        np.testing.assert_allclose(second.total.forces, first.total.forces[::-1], atol=1e-14)
        np.testing.assert_allclose(second.full_low.forces, first.full_low.forces[::-1], atol=1e-14)
        self.assertEqual(model.full_low_backend.inputs[-1].atom_ids, QM_IDS + ENV_IDS)

    def test_zero_field_and_zero_reference_recover_mechanical_subtraction(self):
        atoms = geometry()
        result = calculator(cross=reference(0), embedded=EmbeddedModel(0)).evaluate(atoms)
        mechanical = SubtractiveCalculator(LowModel(0.11), LowModel(0.21), full_low=LowModel(0.13))
        expected = mechanical.evaluate(atoms, atoms.select(QM_IDS))
        self.assertAlmostEqual(result.energy, expected.energy, places=14)
        np.testing.assert_allclose(result.forces, expected.forces, atol=1e-14)

    def test_reference_can_be_removed_without_removing_embedded_response(self):
        atoms = geometry()
        with_reference = calculator().evaluate_terms(atoms)
        no_reference = calculator(cross=reference(0)).evaluate_terms(atoms)
        self.assertAlmostEqual(
            no_reference.total.energy - with_reference.total.energy,
            with_reference.reference_cross.energy_ev,
            places=14,
        )
        np.testing.assert_allclose(
            no_reference.total.forces - with_reference.total.forces,
            cross_force_in_full_order(with_reference.reference_cross, atoms),
            atol=1e-14,
        )

    def test_invalid_regions_geometry_and_identity(self):
        arguments = (LowModel(0.1), LowModel(0.2), EmbeddedModel(), reference())
        invalid_regions = [([], ENV_IDS), (QM_IDS, []), (QM_IDS, QM_IDS), ((30, 30), ENV_IDS)]
        for qm_ids, env_ids in invalid_regions:
            with self.subTest(qm=qm_ids, env=env_ids), self.assertRaises(ValueError):
                ReferenceSubtractiveCalculator(
                    *arguments, qm_atom_ids=qm_ids, environment_atom_ids=env_ids
                )
        with self.assertRaisesRegex(ValueError, "Reference charge IDs"):
            ReferenceSubtractiveCalculator(
                *arguments, qm_atom_ids=tuple(reversed(QM_IDS)), environment_atom_ids=ENV_IDS
            )
        with self.assertRaises(TypeError):
            ReferenceSubtractiveCalculator(
                *arguments[:3], None, qm_atom_ids=QM_IDS, environment_atom_ids=ENV_IDS
            )
        atoms, model = geometry(), calculator()
        with self.assertRaisesRegex(ValueError, "nonperiodic"):
            model.evaluate(dataclasses.replace(atoms, cell=np.eye(3) * 20))
        with self.assertRaisesRegex(ValueError, "cover all"):
            model.evaluate(atoms.select(atoms.atom_ids[:-1]))
        model.evaluate(atoms)
        symbols = list(atoms.symbols)
        symbols[0] = "C"
        with self.assertRaisesRegex(ValueError, "symbols"):
            model.evaluate(dataclasses.replace(atoms, symbols=tuple(symbols)))

    def test_backend_errors_and_bad_returns_are_not_replaced(self):
        atoms = geometry()
        for field in ("full_low", "region_low"):
            for bad in (None, Evaluation(0, np.zeros((1, 3)))):
                with self.subTest(field=field, bad=bad), self.assertRaises(EvaluationError):
                    calculator(
                        **{field: SimpleNamespace(evaluate=lambda g, bad=bad: bad)}
                    ).evaluate(atoms)
        for bad in (
            Evaluation(0, np.zeros((2, 3))),
            EmbeddedEvaluation(0, ENV_IDS, QM_IDS, np.zeros((2, 3)), np.zeros((2, 3))),
        ):
            with self.subTest(bad=bad), self.assertRaises(EvaluationError):
                calculator(embedded=SimpleNamespace(evaluate=lambda q, m, bad=bad: bad)).evaluate(
                    atoms
                )
        model = calculator()
        model.evaluate(atoms)
        with (
            patch.object(model.embedded_backend, "evaluate", side_effect=EvaluationError("SCF")),
            self.assertRaisesRegex(EvaluationError, "SCF"),
        ):
            model.evaluate(atoms)
        with (
            patch.object(FixedChargeCrossCoulomb, "evaluate", return_value=None),
            self.assertRaisesRegex(EvaluationError, "Reference cross"),
        ):
            calculator().evaluate(atoms)


if __name__ == "__main__":
    unittest.main()
