# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite embedding contracts; real PySCF checks require an allocated job.

Set QMDPA_REAL_EMBEDDING_TESTS=1 to opt into the point/Gaussian derivative test.
The small external-charge fixture checks an interface, not an oxide model.
"""

import dataclasses
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from qmdpa.core import EvaluationError, Geometry
from qmdpa.embedding import EmbeddedEvaluation, FixedEmbeddingCharges, PySCFElectrostaticBackend


def geometries():
    qm = Geometry((11, 4, 8), ("O", "H", "H"), [[0, 0, 0], [0.963, 0, 0], [-0.244, 0.928, 0.035]])
    environment = Geometry(
        (21, 14, 18), ("O", "H", "H"), [[3.1, 0.3, -0.2], [3.8, 0.7, 0.2], [2.6, 0.9, -0.6]]
    )
    return qm, environment


def fixed_charges(radii=None):
    return FixedEmbeddingCharges(
        (21, 14, 18),
        [-1.040, 0.520, 0.520],
        "PySCF examples/qmmm/31-gaussian_charge.py; interface fixture only",
        radii,
    )


class FakeSCF:
    def __init__(self, spin=0):
        self.grids = SimpleNamespace(level=None, coords=None)
        self.nlcgrids = SimpleNamespace(level=2, coords=None)
        self.converged = True
        self.conv_tol_grad = 3e-5
        self.gradient_called = False
        self.qm_gradient = np.arange(9).reshape(3, 3) / 10
        self.mm_electronic = -np.arange(9).reshape(3, 3) / 20
        self.mm_nuclear = np.full((3, 3), 0.3)
        self.density = np.array([[1.0, 0.2], [0.2, 2.0]])
        if spin:
            self.density = np.array([self.density, 0.25 * self.density])
        self.gradient = SimpleNamespace(
            grid_response=None,
            kernel=lambda: self.qm_gradient,
            grad_hcore_mm=self.mm_gradient,
            grad_nuc_mm=lambda: self.mm_nuclear,
        )

    def kernel(self):
        return -7.5

    def do_nlc(self):
        return self.xc == "wb97m-v"

    def dip_moment(self, **kwargs):
        self.dipole_call = kwargs
        return np.array([0.1, 0.2, 0.3])

    def nuc_grad_method(self):
        self.gradient_called = True
        return self.gradient

    def make_rdm1(self):
        return self.density

    def mm_gradient(self, density):
        self.density_used = np.array(density, copy=True)
        return self.mm_electronic


def fake_modules(observed):
    package = ModuleType("pyscf")
    data = ModuleType("pyscf.data")
    data.nist = SimpleNamespace(HARTREE2EV=2.0, BOHR=0.5)

    def molecule(**kwargs):
        observed["molecule"] = kwargs
        return SimpleNamespace(spin=kwargs["spin"])

    def method(mol):
        value = FakeSCF(mol.spin)
        observed.setdefault("methods", []).append(value)
        if "mutate" in observed:
            observed["mutate"](value)
        return value

    def mm_charge(value, positions, charges, *, radii, unit):
        observed["embedding"] = {
            "positions": np.array(positions, copy=True),
            "charges": np.array(charges, copy=True),
            "radii": None if radii is None else np.array(radii, copy=True),
            "unit": unit,
        }
        return value

    package.gto = SimpleNamespace(M=molecule)
    package.dft = SimpleNamespace(RKS=method, UKS=method)
    package.lib = SimpleNamespace(num_threads=lambda value: observed.update(threads=value))
    package.qmmm = SimpleNamespace(mm_charge=mm_charge)
    return {"pyscf": package, "pyscf.data": data}


class ChargeSpecificationTests(unittest.TestCase):
    def test_explicit_source_dimensions_and_finite_data(self):
        for values in (
            {"source": " "},
            {"source": None},
            {"charges_e": [1, 2]},
            {"charges_e": [1, np.nan, 3]},
            {"charges_e": [True, False, True]},
            {"radii_angstrom": [0, 1, 1]},
            {"radii_angstrom": [1, 2]},
            {"radii_angstrom": [1, np.inf, 1]},
            {"atom_ids": (21, 21, 18)},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                dataclasses.replace(fixed_charges(), **values)

    def test_caller_arrays_cannot_change_fixed_charge_specification(self):
        values = np.array([-1.04, 0.52, 0.52])
        radii = np.array([0.63, 0.32, 0.32])
        specification = FixedEmbeddingCharges((21, 14, 18), values, "explicit fixture", radii)
        values[:] = 0
        radii[:] = 0
        self.assertEqual(specification.charges_e[0], -1.04)
        self.assertEqual(specification.radii_angstrom[0], 0.63)
        with self.assertRaises(ValueError):
            specification.charges_e.setflags(write=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            specification.source = "changed"


class EmbeddingContractTests(unittest.TestCase):
    def backend(self, *, charges=None, observed=None, **settings):
        observed = {} if observed is None else observed
        with patch.dict(sys.modules, fake_modules(observed)):
            result = PySCFElectrostaticBackend(
                fixed_charges() if charges is None else charges, **settings
            )
        return result, observed

    def test_official_qmmm_receives_fixed_charge_data_and_shared_scf_settings(self):
        qm, environment = geometries()
        for radii in (None, [0.63, 0.32, 0.32]):
            with self.subTest(radii=radii):
                backend, observed = self.backend(
                    charges=fixed_charges(radii),
                    xc="pbe",
                    basis="def2-svp",
                    conv_tol=1e-11,
                    conv_tol_grad=1e-7,
                    grid_level=4,
                    threads=1,
                )
                result = backend.evaluate(qm, environment)
                method = observed["methods"][-1]
                self.assertEqual(
                    (method.xc, method.conv_tol, method.conv_tol_grad), ("pbe", 1e-11, 1e-7)
                )
                self.assertEqual(method.grids.level, 4)
                self.assertTrue(method.gradient.grid_response)
                self.assertEqual(observed["threads"], 1)
                self.assertEqual(observed["molecule"]["basis"], "def2-svp")
                self.assertEqual(observed["embedding"]["unit"], "Angstrom")
                np.testing.assert_array_equal(
                    observed["embedding"]["positions"], environment.positions
                )
                np.testing.assert_array_equal(
                    observed["embedding"]["charges"], backend.charges.charges_e
                )
                if radii is None:
                    self.assertIsNone(observed["embedding"]["radii"])
                else:
                    np.testing.assert_array_equal(observed["embedding"]["radii"], radii)
                self.assertEqual(result.energy_ev, -15)
                self.assertEqual(result.qm_atom_ids, qm.atom_ids)
                self.assertEqual(result.environment_atom_ids, environment.atom_ids)
                np.testing.assert_allclose(
                    result.qm_forces_ev_per_angstrom, -4 * method.qm_gradient
                )
                np.testing.assert_allclose(
                    result.environment_forces_ev_per_angstrom,
                    -4 * (method.mm_electronic + method.mm_nuclear),
                )

    def test_nlc_grid_and_qm_only_dipole_diagnostics_are_explicit(self):
        for level in (None, 4):
            with self.subTest(level=level):
                backend, observed = self.backend(
                    xc="wb97m-v",
                    basis="def2-tzvpd",
                    grid_level=4,
                    nlc_grid_level=level,
                    include_qm_dipole=True,
                    conv_tol_grad=1e-8,
                )
                result = backend.evaluate(*geometries())
                method = observed["methods"][-1]
                self.assertEqual(method.nlcgrids.level, 2 if level is None else 4)
                self.assertEqual(result.diagnostics["regular_grid_level"], 4)
                self.assertEqual(result.diagnostics["nlc_grid_level"], method.nlcgrids.level)
                self.assertTrue(result.diagnostics["nlc_enabled"])
                self.assertEqual(result.diagnostics["conv_tol_grad"], 1e-8)
                self.assertEqual(result.diagnostics["qm_dipole_debye"], (0.1, 0.2, 0.3))
                self.assertEqual(method.dipole_call["unit"], "Debye")
                np.testing.assert_array_equal(method.dipole_call["dm"], method.density)
                np.testing.assert_array_equal(method.dipole_call["origin"], np.zeros(3))
                with self.assertRaises(TypeError):
                    result.diagnostics["xc"] = "changed"
        for settings in (
            {"nlc_grid_level": True},
            {"nlc_grid_level": -1},
            {"nlc_grid_level": 1.2},
            {"include_qm_dipole": "yes"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PySCFElectrostaticBackend(fixed_charges(), **settings)

    def test_total_density_for_restricted_and_unrestricted_mm_reaction(self):
        for multiplicity in (1, 2):
            with self.subTest(multiplicity=multiplicity):
                backend, observed = self.backend(multiplicity=multiplicity)
                backend.evaluate(*geometries())
                method = observed["methods"][-1]
                expected = method.density if multiplicity == 1 else method.density.sum(axis=0)
                np.testing.assert_array_equal(method.density_used, expected)

    def test_each_frame_passes_current_environment_positions_without_reusing_scf(self):
        qm, environment = geometries()
        backend, observed = self.backend()
        backend.evaluate(qm, environment)
        moved = dataclasses.replace(environment, positions=environment.positions + [0.1, 0, 0])
        backend.evaluate(qm, moved)
        self.assertEqual(len(observed["methods"]), 2)
        np.testing.assert_array_equal(observed["embedding"]["positions"], moved.positions)

    def test_periodicity_id_overlap_and_charge_order_are_rejected_before_scf(self):
        qm, environment = geometries()
        for q, m in (
            (dataclasses.replace(qm, cell=np.eye(3) * 8), environment),
            (qm, dataclasses.replace(environment, cell=np.eye(3) * 8)),
            (dataclasses.replace(qm, atom_ids=(21, 4, 8)), environment),
            (qm, environment.select(tuple(reversed(environment.atom_ids)))),
            (qm, dataclasses.replace(environment, positions=qm.positions)),
        ):
            with self.subTest(q=q.atom_ids, m=m.atom_ids):
                backend, observed = self.backend()
                with self.assertRaises(ValueError):
                    backend.evaluate(q, m)
                self.assertNotIn("methods", observed)

    def test_membership_and_symbols_stay_fixed(self):
        qm, environment = geometries()
        backend, observed = self.backend()
        backend.evaluate(qm, environment)
        for q, m in (
            (dataclasses.replace(qm, atom_ids=(12, 4, 8)), environment),
            (dataclasses.replace(qm, symbols=("N", "H", "H")), environment),
            (qm, dataclasses.replace(environment, symbols=("N", "H", "H"))),
        ):
            with self.subTest(q=q.atom_ids, m=m.symbols), self.assertRaises(ValueError):
                backend.evaluate(q, m)
        self.assertEqual(len(observed["methods"]), 1)

    def test_unconverged_scf_never_reuses_previous_gradients(self):
        backend, observed = self.backend()
        backend.evaluate(*geometries())
        observed["mutate"] = lambda method: setattr(method, "converged", False)
        with self.assertRaisesRegex(EvaluationError, "did not converge"):
            backend.evaluate(*geometries())
        self.assertFalse(observed["methods"][-1].gradient_called)

    def test_invalid_qm_mm_gradient_or_density_never_produces_a_result(self):
        for field, value in (
            ("qm_gradient", np.full((3, 3), np.nan)),
            ("qm_gradient", np.zeros((2, 3))),
            ("mm_electronic", np.zeros((1, 3))),
            ("mm_nuclear", np.full((3, 3), np.nan)),
            ("density", np.zeros((3, 2, 2))),
            ("density", np.full((2, 2), np.nan)),
        ):
            with self.subTest(field=field, shape=value.shape):
                observed = {"mutate": lambda method: setattr(method, field, value)}
                backend, _ = self.backend(observed=observed)
                with self.assertRaises(EvaluationError):
                    backend.evaluate(*geometries())

    def test_result_is_immutable_and_keeps_the_two_atom_orders_distinct(self):
        qm, environment = geometries()
        result = EmbeddedEvaluation(
            1, qm.atom_ids, environment.atom_ids, np.zeros((3, 3)), np.ones((3, 3))
        )
        with self.assertRaises(ValueError):
            result.environment_forces_ev_per_angstrom.setflags(write=True)
        with self.assertRaises(ValueError):
            dataclasses.replace(result, environment_atom_ids=qm.atom_ids)


@unittest.skipUnless(
    os.environ.get("QMDPA_REAL_EMBEDDING_TESTS") == "1",
    "Real embedding gradients require an allocated job and explicit opt-in",
)
class RealEmbeddingTests(unittest.TestCase):
    def test_point_and_gaussian_qm_and_environment_force_derivatives(self):
        qm, environment = geometries()
        for radii in (None, [0.63, 0.32, 0.32]):
            with self.subTest(radii=radii):
                backend = PySCFElectrostaticBackend(
                    fixed_charges(radii),
                    xc="pbe",
                    basis="def2-svp",
                    conv_tol=1e-11,
                    conv_tol_grad=1e-8,
                    grid_level=4,
                    grid_response=True,
                    threads=1,
                )
                result = backend.evaluate(qm, environment)
                total_force = result.qm_forces_ev_per_angstrom.sum(
                    axis=0
                ) + result.environment_forces_ev_per_angstrom.sum(axis=0)
                np.testing.assert_allclose(total_force, 0, atol=2e-4, rtol=0)
                step = 1e-4
                for group, reference in (
                    ("qm", result.qm_forces_ev_per_angstrom),
                    ("environment", result.environment_forces_ev_per_angstrom),
                ):
                    original = qm if group == "qm" else environment
                    for axis in range(3):
                        energies = []
                        for sign in (-1, 1):
                            positions = original.positions.copy()
                            positions[1, axis] += sign * step
                            moved = dataclasses.replace(original, positions=positions)
                            value = (
                                backend.evaluate(moved, environment)
                                if group == "qm"
                                else backend.evaluate(qm, moved)
                            )
                            energies.append(value.energy_ev)
                        force = -(energies[1] - energies[0]) / (2 * step)
                        self.assertAlmostEqual(reference[1, axis], force, delta=2e-4)


if __name__ == "__main__":
    unittest.main()
