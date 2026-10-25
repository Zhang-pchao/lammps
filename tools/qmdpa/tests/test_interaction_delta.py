# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Independent pair-law, all-coordinate derivative and rejection controls."""

import unittest

import numpy as np

from qmdpa.core import Evaluation, EvaluationError, Geometry
from qmdpa.interaction_delta import CorrectedCalculator, CrossInteractionDelta


class PairModel:
    def __init__(self, stiffness, offset=0.0):
        self.stiffness = stiffness
        self.offset = offset
        self.calls = []

    def evaluate(self, geometry):
        self.calls.append(geometry.atom_ids)
        positions = geometry.positions
        energy = self.offset * len(positions)
        force = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(i):
                vector = positions[i] - positions[j]
                energy += 0.5 * self.stiffness * np.dot(vector, vector)
                force[i] -= self.stiffness * vector
                force[j] += self.stiffness * vector
        return Evaluation(energy, force)


def atoms():
    return Geometry(
        (91, 10000000001, 18, 7, 46),
        ("O", "H", "He", "C", "H"),
        [[3.0, 0.2, 0.1], [0.1, 0.7, 0.2], [8.0, 9.0, 7.0], [0.0, 0.0, 0.0], [3.4, 0.7, -0.1]],
    )


CORE = (7, 10000000001)
FRAGMENT = (46, 91)


def calculator(high=None, low=None, core=CORE, fragment=FRAGMENT):
    h = PairModel(0.21, 4.0) if high is None else high
    low_model = PairModel(0.13, -9.0) if low is None else low
    return CrossInteractionDelta(h, h, h, low_model, low_model, low_model, core_atom_ids=core, fragment_atom_ids=fragment)


def analytic_cross(geometry, stiffness):
    forces = np.zeros_like(geometry.positions)
    energy = 0.0
    for atom in CORE:
        i = geometry.atom_ids.index(atom)
        for other in FRAGMENT:
            j = geometry.atom_ids.index(other)
            vector = geometry.positions[i] - geometry.positions[j]
            energy += 0.5 * stiffness * np.dot(vector, vector)
            forces[i] -= stiffness * vector
            forces[j] += stiffness * vector
    return Evaluation(energy, forces)


class InteractionDeltaTests(unittest.TestCase):
    def test_explicit_cross_pair_oracle_and_ledger(self):
        geometry = atoms()
        result = calculator().evaluate_terms(geometry)
        expected = analytic_cross(geometry, 0.21 - 0.13)
        self.assertAlmostEqual(result.correction.energy, expected.energy, places=12)
        np.testing.assert_allclose(result.correction.forces, expected.forces, atol=1e-14, rtol=0)
        self.assertEqual(result.atom_ids, geometry.atom_ids)
        self.assertAlmostEqual(
            result.high_interaction.energy - result.low_interaction.energy,
            result.correction.energy,
            places=14,
        )
        np.testing.assert_array_equal(result.correction.forces[2], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result.correction.forces.sum(axis=0), 0.0, atol=1e-14)
        self.assertFalse(result.correction.forces.flags.writeable)

    def test_all_coordinate_two_step_derivatives(self):
        g = atoms()
        calc = calculator()
        expected = calc.evaluate(g).forces
        for step in (1e-3, 5e-4):
            for i in range(len(g.atom_ids)):
                for axis in range(3):
                    with self.subTest(step=step, atom=i, axis=axis):
                        energies = []
                        for sign in (1, -1):
                            x = g.positions.copy()
                            x[i, axis] += sign * step
                            energies.append(
                                calc.evaluate(Geometry(g.atom_ids, g.symbols, x)).energy
                            )
                        self.assertAlmostEqual(
                            -(energies[0] - energies[1]) / (2 * step), expected[i, axis], delta=1e-9
                        )

    def test_reorder_large_ids_and_provider_orders(self):
        g = atoms()
        h = PairModel(0.21)
        low_model = PairModel(0.13)
        calc = calculator(h, low_model)
        original = calc.evaluate(g)
        order = (46, 7, 18, 91, 10000000001)
        permuted = g.select(order)
        result = calc.evaluate(permuted)
        self.assertEqual(h.calls, [CORE + FRAGMENT, CORE, FRAGMENT] * 2)
        self.assertEqual(low_model.calls, h.calls)
        self.assertEqual(result.energy, original.energy)
        np.testing.assert_array_equal(
            result.forces, original.forces[[g.atom_ids.index(i) for i in order]]
        )

    def test_equal_methods_cancel_and_atomic_offsets_cancel(self):
        g = atoms()
        same = PairModel(0.2, 17.0)
        zero = calculator(same, same).evaluate(g)
        self.assertEqual(zero.energy, 0.0)
        np.testing.assert_array_equal(zero.forces, np.zeros_like(g.positions))
        a = calculator(PairModel(0.2, 23.0), PairModel(0.1, -12.0)).evaluate(g)
        b = calculator(PairModel(0.2), PairModel(0.1)).evaluate(g)
        self.assertAlmostEqual(a.energy, b.energy, places=12)
        np.testing.assert_array_equal(a.forces, b.forces)

    def test_translation_and_group_swap(self):
        g = atoms()
        first = calculator().evaluate(g)
        shifted = Geometry(g.atom_ids, g.symbols, g.positions + [2.0, -4.0, 0.6])
        second = calculator(core=FRAGMENT, fragment=CORE).evaluate(shifted)
        self.assertAlmostEqual(first.energy, second.energy, places=12)
        np.testing.assert_allclose(first.forces, second.forces, atol=1e-14, rtol=0)

    def test_invalid_groups_and_backend(self):
        for core, fragment in (
            ((), FRAGMENT),
            (CORE, ()),
            ((7, 7), FRAGMENT),
            (CORE, (7, 91)),
            ((True,), FRAGMENT),
        ):
            with (
                self.subTest(core=core, fragment=fragment),
                self.assertRaises((ValueError, TypeError)),
            ):
                calculator(core=core, fragment=fragment)
        with self.assertRaises(TypeError):
            calculator(high=object())

    def test_missing_ids_periodic_and_changed_symbols(self):
        g = atoms()
        calc = calculator()
        calc.evaluate(g)
        with self.assertRaises(ValueError):
            calc.evaluate(g.select((91, 18, 7, 46)))
        with self.assertRaises(ValueError):
            calc.evaluate(Geometry(g.atom_ids, g.symbols, g.positions, cell=np.eye(3) * 20))
        symbols = list(g.symbols)
        symbols[0] = "N"
        with self.assertRaises(ValueError):
            calc.evaluate(Geometry(g.atom_ids, symbols, g.positions))

    def test_backend_error_and_shape_propagate(self):
        class Broken:
            def evaluate(self, g):
                raise EvaluationError("deliberate failure")

        class Wrong:
            def evaluate(self, g):
                return Evaluation(0.0, np.zeros((1, 3)))

        for backend in (Broken(), Wrong()):
            with self.subTest(backend=type(backend).__name__), self.assertRaises(EvaluationError):
                calculator(high=backend).evaluate(atoms())

    def test_complete_baseline_composition_and_failure(self):
        g = atoms()
        low = PairModel(0.13)
        delta = calculator()
        wrapped = CorrectedCalculator(low, delta)
        terms = wrapped.evaluate_terms(g)
        base = low.evaluate(g)
        cross = analytic_cross(g, 0.08)
        self.assertAlmostEqual(terms.total.energy, base.energy + cross.energy, places=12)
        np.testing.assert_allclose(terms.total.forces, base.forces + cross.forces, atol=1e-14)
        self.assertEqual(terms.atom_ids, g.atom_ids)
        np.testing.assert_array_equal(wrapped.evaluate(g).forces, terms.total.forces)
        with self.assertRaises(TypeError):
            CorrectedCalculator(low, object())

        class Broken:
            def evaluate(self, g):
                raise EvaluationError("correction failed")

        with self.assertRaisesRegex(EvaluationError, "correction failed"):
            CorrectedCalculator(low, Broken()).evaluate(g)


if __name__ == "__main__":
    unittest.main()
