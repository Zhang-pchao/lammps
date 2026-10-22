# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""All-atom additive bridge contracts; no real model or MD trajectory runs."""

import unittest
from types import SimpleNamespace

import numpy as np
from test_lammps_bridge import Driver

from qmdpa.core import Evaluation, EvaluationError
from qmdpa.lammps import AdditiveExternalCoupling
from qmdpa.region import FixedRegion


class TotalCalculator:
    def __init__(self):
        self.geometries = []

    def evaluate(self, geometry):
        self.geometries.append(geometry)
        return Evaluation(9.5, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def correction(self, geometry):
        raise AssertionError("The additive bridge must not request a correction")


class AdditiveDriver(Driver):
    def __init__(self, *, tags=(5, 99, 22), world_size=1, empty=False):
        super().__init__(tags=tags, world_size=world_size, empty=empty)
        self.styles = dict.fromkeys(
            ("bond_style", "angle_style", "dihedral_style", "improper_style", "kspace_style")
        )
        self.styles["pair_style"] = "zero"
        self.natoms = 3
        self.ntypes = 2
        self.box = ([0, 0, 0], [8, 8, 8], 0.0, 0.0, 0.0, [0, 0, 0], 0)
        self.set_owned(None if empty else tags)
        self.numpy = SimpleNamespace(
            extract_atom=lambda name: {
                "image": self.images,
                "type": self.types,
                "f": self.native_forces,
            }[name]
        )
        self.style_reads = []

    def set_owned(self, tags):
        self.nlocal = 0 if tags is None else len(tags)
        self.tags = None if tags is None else np.array(tags, dtype=np.int64)
        self.positions = None if tags is None else np.zeros((len(tags), 3))
        self.forces = None if tags is None else np.full((len(tags), 3), 42.0)
        self.images = None if tags is None else np.zeros(len(tags), dtype=np.int64)
        self.native_forces = None if tags is None else np.zeros((len(tags), 3))
        self.types = (
            None
            if tags is None
            else np.array([2 if tag == 5 else 1 for tag in tags], dtype=np.int32)
        )

    def extract_global(self, name):
        if name in ("natoms", "ntypes"):
            return getattr(self, name)
        if name in self.styles:
            self.style_reads.append(name)
            return self.styles[name]
        return super().extract_global(name)

    def decode_image_flags(self, value):
        return [value, 0, 0]


class GatherComm:
    rank = 0
    size = 2

    def __init__(self, remote_ids):
        self.remote_ids = remote_ids
        self.remote_error = None
        self.remote_write_error = None
        self.broadcasts = []

    def gather(self, value, root=0):
        if isinstance(value, tuple):
            remote = (
                (self.remote_error, None)
                if self.remote_error
                else (None, (self.remote_ids, np.zeros((len(self.remote_ids), 3))))
            )
        else:
            remote = self.remote_write_error
        return [value, remote]

    def bcast(self, value, root=0):
        self.broadcasts.append(value)
        return value


class AdditiveWorkerComm:
    rank = 1
    size = 2

    def __init__(self, messages):
        self.messages = iter(messages)
        self.gathers = []

    def gather(self, value, root=0):
        self.gathers.append(value)

    def bcast(self, value, root=0):
        return next(self.messages)


class AdditiveBridgeTests(unittest.TestCase):
    def region(self):
        return FixedRegion((22, 5, 99), ("O", "H", "O"))

    def attach(self, lmp, calculator=None, **options):
        return AdditiveExternalCoupling(
            lmp,
            self.region(),
            TotalCalculator() if calculator is None else calculator,
            type_symbols=("O", "H"),
            **options,
        )

    def test_total_energy_and_all_forces_use_current_owned_ids(self):
        lmp = AdditiveDriver()
        calculator = TotalCalculator()
        bridge = self.attach(lmp, calculator)
        self.assertIsNone(bridge.last_energy)
        self.assertEqual(bridge.run(0), 9.5)
        np.testing.assert_array_equal(lmp.forces, [[4, 5, 6], [7, 8, 9], [1, 2, 3]])
        self.assertEqual(bridge.last_energy, 9.5)
        self.assertEqual(bridge.last_step, 0)
        self.assertEqual(lmp.energy_calls[-1], ("qmdpa_additive", 9.5))
        self.assertEqual(calculator.geometries[0].atom_ids, (22, 5, 99))
        self.assertFalse(calculator.geometries[0].periodic)
        lmp.set_owned((99, 22, 5))
        lmp.positions[:] = [[9, 0, 0], [2, 0, 0], [5, 0, 0]]
        bridge.run(1)
        np.testing.assert_array_equal(lmp.forces, [[7, 8, 9], [1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(
            calculator.geometries[-1].positions, [[2, 0, 0], [5, 0, 0], [9, 0, 0]]
        )
        self.assertEqual(len(calculator.geometries), 2)
        self.assertEqual(lmp.swallowed, [])

    def test_large_atom_ids_are_preserved_through_gather_and_scatter(self):
        lmp = AdditiveDriver(tags=(99, 10000000001, 22))
        lmp.types[:] = [1, 2, 1]
        region = FixedRegion((10000000001, 22, 99), ("H", "O", "O"))
        calculator = TotalCalculator()
        bridge = AdditiveExternalCoupling(lmp, region, calculator, type_symbols=("O", "H"))
        bridge.run(0)
        self.assertEqual(calculator.geometries[0].atom_ids, region.atom_ids)
        np.testing.assert_array_equal(lmp.forces, [[7, 8, 9], [1, 2, 3], [4, 5, 6]])

    def test_attach_does_not_change_integrator_or_native_interactions(self):
        lmp = AdditiveDriver()
        self.attach(lmp)
        self.assertEqual(
            lmp.commands,
            [
                "fix qmdpa_additive all external pf/callback 1 1",
                "fix_modify qmdpa_additive energy yes virial no",
            ],
        )

    def test_constructor_rejects_nonzero_native_styles_periodicity_and_counts(self):
        cases = [
            ("pair_style", "deepmd"),
            ("pair_style", "hybrid/overlay"),
            ("pair_style", "zero/kk"),
            ("bond_style", "harmonic"),
            ("angle_style", "zero"),
            ("dihedral_style", "opls"),
            ("improper_style", "cvff"),
            ("kspace_style", "pppm"),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                lmp = AdditiveDriver()
                lmp.styles[name] = value
                with self.assertRaises(ValueError):
                    self.attach(lmp)
                self.assertEqual(lmp.commands, [])
        for name, value in (("natoms", 2), ("natoms", 4), ("ntypes", 3)):
            with self.subTest(name=name, value=value):
                lmp = AdditiveDriver()
                setattr(lmp, name, value)
                with self.assertRaises(ValueError):
                    self.attach(lmp)
                self.assertEqual(lmp.commands, [])
        lmp = AdditiveDriver()
        lmp.box = (*lmp.box[:5], [1, 0, 0], lmp.box[6])
        with self.assertRaisesRegex(ValueError, "nonperiodic"):
            self.attach(lmp)
        with self.assertRaisesRegex(ValueError, "explicit type_symbols"):
            AdditiveExternalCoupling(
                AdditiveDriver(), self.region(), TotalCalculator(), type_symbols=None
            )

    def test_inactive_none_styles_are_accepted(self):
        lmp = AdditiveDriver()
        for name in lmp.styles:
            if name != "pair_style":
                lmp.styles[name] = "none"
        self.assertEqual(self.attach(lmp).run(0), 9.5)

    def test_styles_and_counts_are_checked_again_on_every_callback(self):
        for field, value in (
            ("pair_style", "zero/kk"),
            ("pair_style", None),
            ("bond_style", "harmonic"),
            ("kspace_style", "pppm"),
            ("natoms", 4),
            ("ntypes", 3),
        ):
            with self.subTest(field=field):
                lmp = AdditiveDriver()
                bridge = self.attach(lmp)
                bridge.run(0)
                if field in lmp.styles:
                    lmp.styles[field] = value
                else:
                    setattr(lmp, field, value)
                with self.assertRaises(EvaluationError):
                    bridge.run(1)
                self.assertTrue(lmp.timeout)
                self.assertEqual(lmp.swallowed, [])
                self.assertEqual(bridge.last_step, 0)
                self.assertEqual(lmp.energy_calls[-1][1], 0)
        lmp = AdditiveDriver()
        bridge = self.attach(lmp)
        bridge.run(0)
        bridge.run(1)
        self.assertEqual(lmp.style_reads.count("pair_style"), 3)

    def test_extra_missing_and_duplicate_ownership_are_collective_failures(self):
        for tags, pattern in (
            ((5, 99, 77), "Unexpected atoms"),
            ((5, 99), "missing region atom"),
            ((5, 99, 99), "multiple owners"),
        ):
            with self.subTest(tags=tags):
                lmp = AdditiveDriver(tags=tags)
                bridge = self.attach(lmp)
                with self.assertRaisesRegex(EvaluationError, pattern):
                    bridge.run(0)
                self.assertTrue(lmp.timeout)
                self.assertEqual(lmp.swallowed, [])
                np.testing.assert_array_equal(lmp.forces, np.zeros_like(lmp.forces))
                self.assertIsNone(bridge.last_energy)

    def test_every_atom_type_is_checked_after_reordering(self):
        for replacement in (0, 3, 2):
            with self.subTest(replacement=replacement):
                lmp = AdditiveDriver()
                bridge = self.attach(lmp)
                bridge.run(0)
                lmp.set_owned((99, 22, 5))
                lmp.types[0] = replacement
                with self.assertRaises(EvaluationError):
                    bridge.run(1)
                self.assertEqual(lmp.swallowed, [])

    def test_bad_native_force_arrays_fail_before_model_evaluation(self):
        for value in (
            None,
            np.zeros((2, 3)),
            np.zeros((3, 2)),
            np.full((3, 3), np.nan),
            np.full((3, 3), 1e-30),
        ):
            with self.subTest(value=value):
                lmp = AdditiveDriver()
                calculator = TotalCalculator()
                bridge = self.attach(lmp, calculator)
                lmp.native_forces = value
                with self.assertRaises(EvaluationError):
                    bridge.run(0)
                self.assertEqual(calculator.geometries, [])
                self.assertEqual(lmp.swallowed, [])
        lmp = AdditiveDriver()
        lmp.native_forces = np.vstack((np.zeros((3, 3)), np.ones((2, 3))))
        self.assertEqual(self.attach(lmp).run(0), 9.5)

    def test_nonperiodic_image_flags_and_later_box_changes_are_rejected(self):
        lmp = AdditiveDriver()
        bridge = self.attach(lmp)
        lmp.images[0] = 1
        with self.assertRaisesRegex(EvaluationError, "zero image flags"):
            bridge.run(0)
        for change in (
            ([0, 0, 0], [9, 8, 8], 0.0, 0.0, 0.0, [0, 0, 0], 0),
            ([1, 0, 0], [9, 8, 8], 0.0, 0.0, 0.0, [0, 0, 0], 0),
            ([0, 0, 0], [8, 8, 8], 0.0, 0.0, 0.0, [1, 0, 0], 0),
        ):
            with self.subTest(change=change):
                lmp = AdditiveDriver()
                bridge = self.attach(lmp)
                lmp.box = change
                with self.assertRaises(EvaluationError):
                    bridge.run(0)
                self.assertEqual(lmp.swallowed, [])

    def test_empty_root_and_empty_worker_still_publish_global_energy(self):
        root_lmp = AdditiveDriver(world_size=2, empty=True)
        root = self.attach(root_lmp, comm=GatherComm([22, 5, 99]))
        self.assertEqual(root.run(0), 9.5)
        self.assertEqual(root_lmp.energy_calls[-1][1], 9.5)
        self.assertEqual(root_lmp.swallowed, [])
        worker_lmp = AdditiveDriver(world_size=2, empty=True)
        comm = AdditiveWorkerComm([(None, 9.5, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), None])
        worker = AdditiveExternalCoupling(
            worker_lmp, self.region(), None, type_symbols=("O", "H"), comm=comm
        )
        self.assertEqual(worker.run(0), 9.5)
        self.assertEqual(worker_lmp.energy_calls[-1][1], 9.5)
        self.assertEqual(len(comm.gathers), 2)
        self.assertEqual(worker_lmp.swallowed, [])

    def test_atom_migration_changes_local_scatter_without_changing_region(self):
        lmp = AdditiveDriver(tags=(5,), world_size=2)
        comm = GatherComm([22, 99])
        bridge = self.attach(lmp, comm=comm)
        bridge.run(0)
        np.testing.assert_array_equal(lmp.forces, [[4, 5, 6]])
        lmp.set_owned((22, 5))
        comm.remote_ids = [99]
        bridge.run(1)
        np.testing.assert_array_equal(lmp.forces, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(lmp.swallowed, [])

    def test_remote_local_validation_failure_is_broadcast_without_model_call(self):
        lmp = AdditiveDriver(tags=(5,), world_size=2)
        calculator = TotalCalculator()
        comm = GatherComm([22, 99])
        comm.remote_error = "ValueError: Native forces are not zero"
        bridge = self.attach(lmp, calculator, comm=comm)
        with self.assertRaisesRegex(EvaluationError, "Native forces"):
            bridge.run(1)
        self.assertEqual(calculator.geometries, [])
        self.assertEqual(len(comm.broadcasts), 1)
        self.assertEqual(lmp.swallowed, [])

    def test_remote_force_write_failure_invalidates_root_and_worker(self):
        root_lmp = AdditiveDriver(world_size=2)
        comm = GatherComm([])
        comm.remote_write_error = "Remote force write failed"
        bridge = self.attach(root_lmp, comm=comm)
        with self.assertRaisesRegex(EvaluationError, "Remote force write"):
            bridge.run(1)
        self.assertIsNone(bridge.last_energy)
        worker_lmp = AdditiveDriver(world_size=2)
        worker = AdditiveExternalCoupling(
            worker_lmp,
            self.region(),
            None,
            type_symbols=("O", "H"),
            comm=AdditiveWorkerComm(comm.broadcasts),
        )
        with self.assertRaisesRegex(EvaluationError, "Remote force write"):
            worker.run(1)
        for driver in (root_lmp, worker_lmp):
            self.assertTrue(driver.timeout)
            self.assertEqual(driver.energy_calls[-1][1], 0)
            np.testing.assert_array_equal(driver.forces, np.zeros_like(driver.forces))
            self.assertEqual(driver.swallowed, [])

    def test_model_and_invalid_result_failures_never_accept_or_retry(self):
        for result in (object(), Evaluation(0, np.zeros((2, 3)))):
            with self.subTest(result=result):
                lmp = AdditiveDriver()
                bridge = self.attach(lmp, SimpleNamespace(evaluate=lambda _, result=result: result))
                with self.assertRaises(EvaluationError):
                    bridge.run(1)
                self.assertIsNone(bridge.last_energy)
                self.assertEqual(lmp.swallowed, [])
        lmp = AdditiveDriver()

        def fail(_):
            raise EvaluationError("SCF failed")

        bridge = self.attach(lmp, SimpleNamespace(evaluate=fail))
        with self.assertRaisesRegex(EvaluationError, "SCF failed"):
            bridge.run(1)
        commands = list(lmp.commands)
        with self.assertRaises(EvaluationError):
            bridge.run(1)
        self.assertEqual(lmp.commands, commands)

    def test_local_force_or_energy_publication_failure_is_propagated(self):
        lmp = AdditiveDriver()
        lmp.forces.setflags(write=False)
        bridge = self.attach(lmp)
        with self.assertRaises(EvaluationError):
            bridge.run(1)
        self.assertTrue(lmp.timeout)
        self.assertEqual(lmp.swallowed, [])
        lmp = AdditiveDriver()
        bridge = self.attach(lmp)

        def fail_energy(*_):
            raise RuntimeError("Energy publication failed")

        lmp.fix_external_set_energy_global = fail_energy
        with self.assertRaisesRegex(EvaluationError, "Energy publication failed"):
            bridge.run(1)
        self.assertTrue(lmp.timeout)
        self.assertEqual(lmp.swallowed, [])

    def test_incomplete_segment_is_not_accepted_or_resumed(self):
        lmp = AdditiveDriver()
        bridge = self.attach(lmp)
        bridge.run(0)
        lmp.early_stop = True
        with self.assertRaisesRegex(EvaluationError, "requested step 5"):
            bridge.run(5)
        with self.assertRaises(EvaluationError):
            bridge.run(1)


if __name__ == "__main__":
    unittest.main()
