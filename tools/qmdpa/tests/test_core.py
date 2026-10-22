# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Bookkeeping tests; these analytic fixtures are not physical benchmarks."""

import unittest

import numpy as np

from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator


class PairQuadratic:
    """Analytic pair energy used only to test composition and differentiation."""

    def __init__(self, stiffness):
        self.stiffness = stiffness

    def evaluate(self, geometry):
        positions = geometry.positions
        energy = 0.0
        forces = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(i):
                delta = positions[i] - positions[j]
                energy += self.stiffness * np.dot(delta, delta) / 2.0
                forces[i] -= self.stiffness * delta
                forces[j] += self.stiffness * delta
        return Evaluation(energy, forces)


def sample_geometry(cell=None):
    return Geometry(
        (10000000001, 7, 30, 4),
        ("O", "H", "O", "H"),
        [[0.1, 0.3, 0.2], [1.2, 0.1, 0.0], [2.4, 0.7, 0.4], [2.1, 1.3, 0.2]],
        cell,
    )


class GeometryTests(unittest.TestCase):
    def test_ids_are_distinct_positive_integers(self):
        for ids in ((True, 2), (1.0, 2), (1, 1), (0, 2), (-1, 2), ()):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                Geometry(ids, ("H",) * len(ids), np.zeros((len(ids), 3)))
        self.assertEqual(sample_geometry().atom_ids[0], 10000000001)

    def test_snapshot_cannot_be_changed_through_input_or_array(self):
        positions = np.zeros((1, 3))
        cell = np.eye(3)
        geometry = Geometry((1,), ("H",), positions, cell)
        positions[0, 0] = 12.0
        cell[0, 0] = 12.0
        self.assertEqual(geometry.positions[0, 0], 0.0)
        self.assertEqual(geometry.cell[0, 0], 1.0)
        with self.assertRaises(ValueError):
            geometry.positions[0, 0] = 4.0
        with self.assertRaises(ValueError):
            geometry.positions.setflags(write=True)

    def test_invalid_coordinates_and_cells(self):
        for positions in ([[0, 0]], [[np.nan, 0, 0]], [[np.inf, 0, 0]]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                Geometry((1,), ("H",), positions)
        for cell in (np.zeros((3, 3)), np.eye(2), np.diag([-1, 1, 1])):
            with self.subTest(cell=cell), self.assertRaises(ValueError):
                Geometry((1,), ("H",), [[0, 0, 0]], cell)

    def test_selection_preserves_requested_id_order_and_explicit_boundary(self):
        geometry = sample_geometry(np.eye(3) * 8.0)
        selected = geometry.select((30, 10000000001))
        self.assertEqual(selected.atom_ids, (30, 10000000001))
        np.testing.assert_array_equal(selected.positions, geometry.positions[[2, 0]])
        self.assertTrue(selected.periodic)
        self.assertFalse(geometry.select((7,), isolated=True).periodic)
        with self.assertRaisesRegex(ValueError, "absent"):
            geometry.select((99,))


class EvaluationTests(unittest.TestCase):
    def test_invalid_energy_and_force_values(self):
        for energy in (np.inf, np.nan, True, [1.0], 1.0j):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                Evaluation(energy, np.zeros((1, 3)))
        for forces in (np.zeros((0, 3)), np.zeros(3), [[np.inf, 0, 0]]):
            with self.subTest(forces=forces), self.assertRaises(ValueError):
                Evaluation(0.0, forces)

    def test_forces_are_an_immutable_snapshot(self):
        forces = np.ones((1, 3))
        result = Evaluation(2.0, forces)
        forces[:] = 9.0
        np.testing.assert_array_equal(result.forces, np.ones((1, 3)))
        with self.assertRaises(ValueError):
            result.forces.setflags(write=True)


class SubtractiveTests(unittest.TestCase):
    def setUp(self):
        self.geometry = sample_geometry()
        self.ids = (30, 10000000001)
        self.calculator = SubtractiveCalculator(
            PairQuadratic(0.3), PairQuadratic(0.9), full_low=PairQuadratic(0.5)
        )

    def test_full_quantum_cancellation_with_identical_low_inputs(self):
        low = PairQuadratic(0.3)
        high = PairQuadratic(0.9)
        calculator = SubtractiveCalculator(low, high)
        result = calculator.evaluate(self.geometry, self.geometry)
        expected = high.evaluate(self.geometry)
        self.assertAlmostEqual(result.energy, expected.energy, places=13)
        np.testing.assert_allclose(result.forces, expected.forces, atol=1e-14)

    def test_force_matches_energy_derivative_on_qm_and_environment_atoms(self):
        result = self.calculator.evaluate(self.geometry, self.geometry.select(self.ids))
        step = 1e-5
        for atom in range(len(self.geometry.atom_ids)):
            for axis in range(3):
                energies = []
                for sign in (-1, 1):
                    positions = self.geometry.positions.copy()
                    positions[atom, axis] += sign * step
                    displaced = Geometry(self.geometry.atom_ids, self.geometry.symbols, positions)
                    energies.append(
                        self.calculator.evaluate(displaced, displaced.select(self.ids)).energy
                    )
                derivative_force = -(energies[1] - energies[0]) / (2 * step)
                self.assertAlmostEqual(result.forces[atom, axis], derivative_force, places=8)

    def test_environment_receives_only_full_low_force(self):
        result = self.calculator.evaluate(self.geometry, self.geometry.select(self.ids))
        full = PairQuadratic(0.5).evaluate(self.geometry)
        np.testing.assert_array_equal(result.forces[[1, 3]], full.forces[[1, 3]])

    def test_qm_input_order_does_not_change_full_result(self):
        forward = self.calculator.evaluate(self.geometry, self.geometry.select(self.ids))
        reverse = self.calculator.evaluate(self.geometry, self.geometry.select(self.ids[::-1]))
        self.assertAlmostEqual(forward.energy, reverse.energy)
        np.testing.assert_allclose(forward.forces, reverse.forces, atol=1e-14)

    def test_periodic_image_mapping_and_isolated_correction(self):
        geometry = sample_geometry(np.eye(3) * 8.0)
        inner = geometry.select(self.ids, isolated=True)
        images = inner.positions.copy()
        images[0, 0] += 8.0
        shifted = Geometry(inner.atom_ids, inner.symbols, images)
        self.calculator.evaluate(geometry, shifted)
        with self.assertRaisesRegex(ValueError, "isolated"):
            self.calculator.correction(geometry.select(self.ids))
        images[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "coordinates"):
            self.calculator.evaluate(geometry, Geometry(inner.atom_ids, inner.symbols, images))

    def test_wrong_symbols_and_force_count_fail(self):
        inner = self.geometry.select(self.ids)
        wrong = Geometry(inner.atom_ids, ("H", "H"), inner.positions)
        with self.assertRaisesRegex(ValueError, "symbols"):
            self.calculator.evaluate(self.geometry, wrong)

        class WrongSize:
            def evaluate(self, geometry):
                return Evaluation(0.0, np.zeros((1, 3)))

        calculator = SubtractiveCalculator(WrongSize(), PairQuadratic(1.0))
        with self.assertRaisesRegex(EvaluationError, "force count"):
            calculator.correction(inner)


if __name__ == "__main__":
    unittest.main()
