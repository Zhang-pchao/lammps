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
from unittest.mock import Mock, patch

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
    def test_default_and_explicit_false_keep_the_original_loader_call(self):
        for model in ("model.pt", "model.pt2"):
            for options in ({}, {"no_jit": False}):
                with self.subTest(model=model, options=options):
                    potential = FakePotential()
                    modules = deepmd_modules(potential)
                    loader = Mock(return_value=potential)
                    modules["deepmd.infer"].DeepPot = loader
                    with patch.dict(sys.modules, modules):
                        backend = DeepMDBackend(model, **options)
                    loader.assert_called_once_with(model)
                    self.assertIs(backend.no_jit, False)

    def test_explicit_no_jit_is_forwarded_for_raw_checkpoint(self):
        potential = FakePotential()
        modules = deepmd_modules(potential)
        loader = Mock(return_value=potential)
        modules["deepmd.infer"].DeepPot = loader
        with patch.dict(sys.modules, modules):
            backend = DeepMDBackend("model.pt", no_jit=True)
        loader.assert_called_once_with("model.pt", no_jit=True)
        self.assertIs(backend.no_jit, True)
        self.assertEqual(backend.evaluate(water()).energy, -3.0)

    def test_no_jit_rejects_nonboolean_values_before_import(self):
        with patch.dict(sys.modules, {"deepmd": None, "deepmd.infer": None}):
            for value in (0, 1, np.bool_(True), "true", None, []):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "no_jit"):
                    DeepMDBackend("model.pt", no_jit=value)

    def test_no_jit_rejects_compiled_or_nonraw_files_before_import(self):
        with patch.dict(sys.modules, {"deepmd": None, "deepmd.infer": None}):
            for model in ("model.pt2", "model.pte", "model.pth", "model.pb", "model.PT", "model"):
                with self.subTest(model=model), self.assertRaisesRegex(ValueError, "raw PyTorch"):
                    DeepMDBackend(model, no_jit=True)

    def test_no_jit_loader_error_propagates_without_fallback(self):
        failure = RuntimeError("Explicit raw loader failed")
        modules = deepmd_modules(FakePotential())
        loader = Mock(side_effect=failure)
        modules["deepmd.infer"].DeepPot = loader
        with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError) as caught:
            DeepMDBackend("model.pt", no_jit=True)
        self.assertIs(caught.exception, failure)
        loader.assert_called_once_with("model.pt", no_jit=True)

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
        self.nlcgrids = SimpleNamespace(level=2)
        self.nlc_grid_level_used = None
        self.gradient_called = False
        self.conv_tol_grad = None
        self.orbital_gradient_tolerance_used = None
        self.gradient_response_used = None
        self.gradient = SimpleNamespace(grid_response=None, kernel=self.gradient_kernel)

    def kernel(self):
        self.orbital_gradient_tolerance_used = self.conv_tol_grad
        self.nlc_grid_level_used = self.nlcgrids.level
        return -1.5

    def nuc_grad_method(self):
        self.gradient_called = True
        return self.gradient

    def gradient_kernel(self):
        self.gradient_response_used = self.gradient.grid_response
        return np.ones((3, 3))


def pyscf_modules(method, observed):
    def molecule(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(spin=kwargs["spin"])

    package = ModuleType("pyscf")
    package.gto = SimpleNamespace(M=molecule)

    def select_method(kind):
        observed["method"] = kind
        return method

    package.dft = SimpleNamespace(
        RKS=lambda mol: select_method("RKS"), UKS=lambda mol: select_method("UKS")
    )
    package.lib = SimpleNamespace(num_threads=lambda value: observed.update(threads=value))
    data = ModuleType("pyscf.data")
    data.nist = SimpleNamespace(HARTREE2EV=2.0, BOHR=0.5)
    return {"pyscf": package, "pyscf.data": data}


class PySCFContractTests(unittest.TestCase):
    def test_explicit_level_shift_and_unchanged_default(self):
        for shift in (None, 0., .3):
            method = FakeSCF()
            method.level_shift = .125
            with patch.dict(sys.modules, pyscf_modules(method, {})):
                backend = PySCFBackend(scf_level_shift_hartree=shift)
                backend.evaluate(water())
            self.assertEqual(method.level_shift, .125 if shift is None else shift)

    def test_invalid_level_shift_rejected_before_optional_import(self):
        for shift in (True, -.1, float("nan"), float("inf"), "0.3", []):
            with self.subTest(shift=shift), self.assertRaisesRegex(ValueError, "scf_level_shift"):
                PySCFBackend(scf_level_shift_hartree=shift)


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

    def test_dft_grid_response_is_explicit_for_restricted_and_unrestricted_gradients(self):
        for multiplicity, kind in ((1, "RKS"), (2, "UKS")):
            for options, expected in (
                ({}, True),
                ({"grid_response": True}, True),
                ({"grid_response": False}, False),
            ):
                with self.subTest(multiplicity=multiplicity, options=options):
                    method, observed = FakeSCF(), {}
                    with patch.dict(sys.modules, pyscf_modules(method, observed)):
                        backend = PySCFBackend(multiplicity=multiplicity, **options)
                    backend.evaluate(water())
                    self.assertEqual(observed["method"], kind)
                    self.assertIs(method.gradient_response_used, expected)

    def test_grid_response_rejects_nonboolean_values_before_importing_pyscf(self):
        for value in (0, 1, "true", "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "grid_response"):
                PySCFBackend(grid_response=value)

    def test_explicit_orbital_gradient_tolerance_reaches_scf_before_kernel(self):
        for multiplicity in (1, 2):
            for tolerance in (None, 1e-7):
                with self.subTest(multiplicity=multiplicity, tolerance=tolerance):
                    method = FakeSCF()
                    method.conv_tol_grad = 2e-5
                    with patch.dict(sys.modules, pyscf_modules(method, {})):
                        backend = PySCFBackend(multiplicity=multiplicity, conv_tol_grad=tolerance)
                    backend.evaluate(water())
                    expected = 2e-5 if tolerance is None else tolerance
                    self.assertEqual(method.orbital_gradient_tolerance_used, expected)

    def test_invalid_orbital_gradient_tolerances_are_rejected_before_backend_import(self):
        for tolerance in (True, 0, -1, float("inf"), float("nan"), "1e-7"):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "tolerance"):
                PySCFBackend(conv_tol_grad=tolerance)

    def test_nlc_grid_is_independent_and_reaches_restricted_and_unrestricted_scf(self):
        for multiplicity, kind in ((1, "RKS"), (2, "UKS")):
            for options, expected in (
                ({}, 2),
                ({"nlc_grid_level": None}, 2),
                ({"nlc_grid_level": 0}, 0),
                ({"nlc_grid_level": 5}, 5),
                ({"nlc_grid_level": np.int64(4)}, 4),
            ):
                with self.subTest(multiplicity=multiplicity, options=options):
                    method, observed = FakeSCF(), {}
                    with patch.dict(sys.modules, pyscf_modules(method, observed)):
                        backend = PySCFBackend(
                            xc="wb97m-v", multiplicity=multiplicity, grid_level=4, **options
                        )
                    result = backend.evaluate(water())
                    self.assertEqual(observed["method"], kind)
                    self.assertEqual(method.grids.level, 4)
                    self.assertEqual(method.nlc_grid_level_used, expected)
                    self.assertEqual(backend.nlc_grid_level, options.get("nlc_grid_level"))
                    self.assertEqual(result.energy, -3.0)
                    np.testing.assert_array_equal(result.forces, -4.0 * np.ones((3, 3)))

    def test_invalid_nlc_grid_levels_fail_before_importing_pyscf(self):
        with patch.dict(sys.modules, {"pyscf": None, "pyscf.data": None}):
            for level in (True, False, np.bool_(True), -1, 1.0, 1.2, "4", [], {}, np.nan, np.inf):
                with (
                    self.subTest(level=level),
                    self.assertRaisesRegex(ValueError, "NLC grid level"),
                ):
                    PySCFBackend(nlc_grid_level=level)

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
