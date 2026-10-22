# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Adapter contracts, plus explicitly enabled real-backend numerical tests.

Set QMDPA_REAL_TESTS=1 for real PySCF tests. Set QMDPA_MODEL to also test a
real DeePMD model and the coupled force derivative. QMDPA_CHARGE_SPIN is an
optional JSON array, e.g. [0, 1]. Real tests must run in an allocated job.
"""

import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from qmdpa.backends import DeepMDBackend, PySCFBackend
from qmdpa.core import EvaluationError, Geometry, SubtractiveCalculator


def water():
    return Geometry(
        (11, 4, 8), ("O", "H", "H"), [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.25, 0.94, 0.0]]
    )


class FakePotential:
    def __init__(self, *, conditioned=False, frame=0, atomic=0):
        self.conditioned = conditioned
        self.frame = frame
        self.atomic = atomic
        self.last_call = None

    def get_type_map(self):
        return ["H", "O"]

    def has_chg_spin_ebd(self):
        return self.conditioned

    def get_dim_fparam(self):
        return self.frame

    def has_default_fparam(self):
        return False

    def get_dim_aparam(self):
        return self.atomic

    def eval(self, positions, cell, atom_types, **kwargs):
        self.last_call = positions, cell, atom_types, kwargs
        return np.array([[-3.0]]), np.ones_like(positions), np.zeros((1, 9))


def deepmd_modules(potential):
    package = ModuleType("deepmd")
    infer = ModuleType("deepmd.infer")
    infer.DeepPot = lambda model: potential
    return {"deepmd": package, "deepmd.infer": infer}


class DeepMDContractTests(unittest.TestCase):
    def test_type_map_boundary_and_conditions_are_passed_separately(self):
        potential = FakePotential(conditioned=True, frame=1, atomic=2)
        with patch.dict(sys.modules, deepmd_modules(potential)):
            backend = DeepMDBackend("model.pt", charge_spin=[0, 1], fparam=[2.5], aparam=[3, 4])
        result = backend.evaluate(water())
        positions, cell, types, kwargs = potential.last_call
        self.assertIsNone(cell)
        self.assertEqual(types, [1, 0, 0])
        self.assertEqual(positions.shape, (1, 3, 3))
        np.testing.assert_array_equal(kwargs["charge_spin"], [[0, 1]])
        np.testing.assert_array_equal(kwargs["fparam"], [2.5])
        np.testing.assert_array_equal(kwargs["aparam"], [3, 4])
        self.assertEqual(result.energy, -3.0)
        geometry = water()
        backend.evaluate(
            Geometry(geometry.atom_ids, geometry.symbols, geometry.positions, np.eye(3) * 7)
        )
        np.testing.assert_array_equal(potential.last_call[1], (np.eye(3) * 7).reshape(1, 9))

    def test_condition_and_parameter_dimensions_are_validated(self):
        potential = FakePotential(conditioned=True, frame=1, atomic=2)
        with patch.dict(sys.modules, deepmd_modules(potential)):
            for options in (
                {},
                {"charge_spin": [0]},
                {"charge_spin": [0, 1]},
                {"charge_spin": [0, 1], "fparam": [1, 2]},
                {"charge_spin": [0, 1], "fparam": [1], "aparam": [1]},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    DeepMDBackend("model.pt", **options)
            backend = DeepMDBackend("model.pt", charge_spin=[0, 1], fparam=[1], aparam=[[1, 2]])
        with self.assertRaisesRegex(ValueError, "rows"):
            backend.evaluate(water())

    def test_unknown_elements_and_invalid_output_are_rejected(self):
        potential = FakePotential()
        with patch.dict(sys.modules, deepmd_modules(potential)):
            backend = DeepMDBackend("model.pt")
        with self.assertRaisesRegex(ValueError, "Element C"):
            backend.evaluate(Geometry((1,), ("C",), [[0, 0, 0]]))
        potential.eval = lambda *args, **kwargs: (np.array([[np.nan]]), np.zeros((1, 3, 3)), None)
        with self.assertRaises(EvaluationError):
            backend.evaluate(water())


class FakeSCF:
    def __init__(self, converged=True):
        self.converged = converged
        self.grids = SimpleNamespace(level=None)
        self.gradient_called = False

    def kernel(self):
        return -1.5

    def nuc_grad_method(self):
        self.gradient_called = True
        return SimpleNamespace(kernel=lambda: np.ones((3, 3)))


def pyscf_modules(method, observed):
    def molecule(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(spin=kwargs["spin"])

    package = ModuleType("pyscf")
    package.gto = SimpleNamespace(M=molecule)
    package.dft = SimpleNamespace(RKS=lambda mol: method, UKS=lambda mol: method)
    package.lib = SimpleNamespace(num_threads=lambda value: observed.update(threads=value))
    data = ModuleType("pyscf.data")
    data.nist = SimpleNamespace(HARTREE2EV=2.0, BOHR=0.5)
    return {"pyscf": package, "pyscf.data": data}


class PySCFContractTests(unittest.TestCase):
    def test_units_force_sign_spin_and_thread_setting(self):
        observed = {}
        method = FakeSCF()
        with patch.dict(sys.modules, pyscf_modules(method, observed)):
            backend = PySCFBackend(charge=1, multiplicity=2, threads=2)
        result = backend.evaluate(water())
        self.assertEqual(observed["spin"], 1)
        self.assertEqual(observed["charge"], 1)
        self.assertEqual(observed["unit"], "Angstrom")
        self.assertEqual(observed["threads"], 2)
        self.assertEqual(result.energy, -3.0)
        np.testing.assert_array_equal(result.forces, -4.0 * np.ones((3, 3)))

    def test_unconverged_scf_never_returns_a_gradient(self):
        method = FakeSCF(converged=False)
        with patch.dict(sys.modules, pyscf_modules(method, {})):
            backend = PySCFBackend(max_cycle=1)
        with self.assertRaisesRegex(EvaluationError, "did not converge"):
            backend.evaluate(water())
        self.assertFalse(method.gradient_called)

    def test_periodicity_and_noninteger_electronic_state_are_rejected(self):
        for options in ({"charge": 1.2}, {"multiplicity": True}, {"multiplicity": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PySCFBackend(**options)
        with patch.dict(sys.modules, pyscf_modules(FakeSCF(), {})):
            backend = PySCFBackend()
        geometry = water()
        with self.assertRaisesRegex(ValueError, "isolated"):
            backend.evaluate(
                Geometry(geometry.atom_ids, geometry.symbols, geometry.positions, np.eye(3))
            )


@unittest.skipUnless(os.environ.get("QMDPA_REAL_TESTS") == "1", "Real backend tests are opt-in")
class RealBackendTests(unittest.TestCase):
    def test_pyscf_force_derivative(self):
        backend = PySCFBackend(conv_tol=1e-11, grid_level=4, threads=1)
        geometry = water()
        result = backend.evaluate(geometry)
        step = 1e-4
        energies = []
        for sign in (-1, 1):
            positions = geometry.positions.copy()
            positions[1, 0] += sign * step
            energies.append(
                backend.evaluate(Geometry(geometry.atom_ids, geometry.symbols, positions)).energy
            )
        force = -(energies[1] - energies[0]) / (2 * step)
        self.assertAlmostEqual(result.forces[1, 0], force, delta=2e-3)

    @unittest.skipUnless(os.environ.get("QMDPA_MODEL"), "QMDPA_MODEL is required")
    def test_real_low_level_cancellation_and_mixed_force_derivative(self):
        state = os.environ.get("QMDPA_CHARGE_SPIN")
        low = DeepMDBackend(
            os.environ["QMDPA_MODEL"], charge_spin=None if state is None else json.loads(state)
        )
        high = PySCFBackend(conv_tol=1e-11, grid_level=4, threads=1)
        calculator = SubtractiveCalculator(low, high)
        inner = water()
        result = calculator.evaluate(inner, inner)
        expected = high.evaluate(inner)
        self.assertAlmostEqual(result.energy, expected.energy, delta=1e-7)
        np.testing.assert_allclose(result.forces, expected.forces, atol=1e-7, rtol=0.0)
        positions = np.vstack([inner.positions, inner.positions + [3.2, 0.2, 0.1]])
        geometry = Geometry((11, 4, 8, 21, 14, 18), inner.symbols * 2, positions)
        result = calculator.evaluate(geometry, geometry.select(inner.atom_ids))
        # Expose model precision explicitly rather than assuming one FD step fits all models.
        step = float(os.environ.get("QMDPA_FD_STEP", "0.002"))
        tolerance = float(os.environ.get("QMDPA_FD_TOL", "0.02"))
        for atom in (1, 4):
            energies = []
            for sign in (-1, 1):
                displaced_positions = geometry.positions.copy()
                displaced_positions[atom, 0] += sign * step
                displaced = Geometry(geometry.atom_ids, geometry.symbols, displaced_positions)
                energies.append(
                    calculator.evaluate(displaced, displaced.select(inner.atom_ids)).energy
                )
            force = -(energies[1] - energies[0]) / (2 * step)
            self.assertAlmostEqual(result.forces[atom, 0], force, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
