# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Callback bookkeeping and failure protocol tests, not physical acceptance.

The driver fixture deliberately swallows uncaught callback exceptions, as a
ctypes callback does. Tests must detect failure through the coupling protocol.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from qmdpa.core import Evaluation, EvaluationError
from qmdpa.lammps import ExternalCoupling
from qmdpa.region import FixedRegion


class ConstantCorrection:
    def correction(self, geometry):
        return Evaluation(7.5, [[1, 2, 3], [4, 5, 6]])


class Driver:
    def __init__(self, *, tags=(5, 99, 22), world_size=1, empty=False):
        self.tags = None if empty else np.asarray(tags, dtype=np.int64)
        self.positions = None if empty else np.zeros((len(tags), 3))
        self.forces = None if empty else np.full((len(tags), 3), 42.0)
        self.nlocal = 0 if empty else len(tags)
        self.images = None if empty else np.zeros(len(tags), dtype=np.int64)
        self.types = np.array([2, 1, 1], dtype=np.int32)
        self.numpy = SimpleNamespace(
            extract_atom=lambda name: self.images if name == "image" else self.types
        )
        self.step = 0
        self.early_stop = False
        self.box = ([0, 0, 0], [8, 8, 8], 0.0, 0.0, 0.0, [1, 1, 1], 0)
        self.world_size = world_size
        self.commands = []
        self.swallowed = []
        self.energy_calls = []
        self.timeout = False
        self.callback = None

    def extract_global(self, name):
        if name == "units":
            return "metal"
        if name == "ntimestep":
            return self.step
        raise ValueError(name)

    def extract_setting(self, name):
        if name == "world_size":
            return self.world_size
        if name == "world_rank":
            return 0
        raise ValueError(name)

    def extract_box(self):
        return self.box

    def get_mpi_comm(self):
        return None

    def decode_image_flags(self, value):
        return [0, 0, 0]

    def set_fix_external_callback(self, fix_id, callback):
        self.callback = callback

    def fix_external_set_energy_global(self, fix_id, energy):
        self.energy_calls.append((fix_id, energy))

    def force_timeout(self):
        self.timeout = True

    def command(self, command):
        self.commands.append(command)
        if command.startswith("run "):
            steps = int(command.split()[1])
            self.step += min(steps, 1) if self.early_stop else steps
            try:
                self.callback(None, self.step, self.nlocal, self.tags, self.positions, self.forces)
            except Exception as exc:
                # Emulate ctypes: an unhandled callback error does not stop run.
                self.swallowed.append(exc)


class EmptyRootComm:
    """Supply a second rank's owned atoms while rank zero owns no atoms."""

    rank = 0
    size = 2

    def gather(self, value, root=0):
        if isinstance(value, tuple) and len(value) == 2:
            remote = (None, ([22, 5], np.zeros((2, 3))))
        else:
            remote = None
        return [value, remote]

    def allgather(self, value):
        return [value, None]

    def bcast(self, value, root=0):
        return value


class RemoteWriteFailureComm(EmptyRootComm):
    def __init__(self):
        self.broadcasts = []

    def gather(self, value, root=0):
        if isinstance(value, tuple):
            return [value, (None, ([], np.empty((0, 3))))]
        return [value, "Remote force buffer is invalid"]

    def bcast(self, value, root=0):
        self.broadcasts.append(value)
        return value


class WorkerComm:
    rank = 1
    size = 2

    def __init__(self):
        self.messages = iter(
            [
                (None, 7.5, np.array([[1, 2, 3], [4, 5, 6]])),
                "Remote force buffer is invalid",
            ]
        )

    def gather(self, value, root=0):
        return None

    def bcast(self, value, root=0):
        return next(self.messages)


class BridgeTests(unittest.TestCase):
    def region(self):
        return FixedRegion((22, 5), ("O", "H"))

    def test_correction_is_mapped_by_current_tags_and_environment_is_zero(self):
        lmp = Driver()
        coupling = ExternalCoupling(lmp, self.region(), ConstantCorrection())
        self.assertEqual(coupling.run(0), 7.5)
        np.testing.assert_array_equal(lmp.forces, [[4, 5, 6], [0, 0, 0], [1, 2, 3]])
        self.assertEqual(lmp.energy_calls[-1], ("qmdpa", 7.5))
        self.assertEqual(lmp.swallowed, [])
        lmp.tags[:] = [22, 5, 99]
        coupling.run(1)
        np.testing.assert_array_equal(lmp.forces, [[1, 2, 3], [4, 5, 6], [0, 0, 0]])

    def test_backend_failure_stops_and_cannot_be_resumed(self):
        class FailingCalculator:
            def correction(self, geometry):
                raise EvaluationError("SCF did not converge")

        lmp = Driver()
        coupling = ExternalCoupling(lmp, self.region(), FailingCalculator())
        with self.assertRaisesRegex(EvaluationError, "SCF did not converge"):
            coupling.run(4)
        self.assertTrue(lmp.timeout)
        self.assertEqual(lmp.energy_calls[-1][1], 0.0)
        np.testing.assert_array_equal(lmp.forces, np.zeros((3, 3)))
        commands = list(lmp.commands)
        with self.assertRaises(EvaluationError):
            coupling.run(1)
        self.assertEqual(commands, lmp.commands)
        self.assertEqual(lmp.swallowed, [])

    def test_local_packet_failure_is_reported_through_run(self):
        lmp = Driver()
        coupling = ExternalCoupling(lmp, self.region(), ConstantCorrection())
        lmp.images = None
        with self.assertRaises(EvaluationError):
            coupling.run(0)
        self.assertTrue(lmp.timeout)
        self.assertEqual(lmp.swallowed, [])

    def test_local_mapping_failure_is_not_swallowed_by_ctypes(self):
        class BrokenRegion(FixedRegion):
            def local_forces(self, local_ids, correction):
                raise ValueError("Invalid local force mapping")

        lmp = Driver()
        coupling = ExternalCoupling(lmp, BrokenRegion((22, 5), ("O", "H")), ConstantCorrection())
        with self.assertRaisesRegex(EvaluationError, "local force mapping"):
            coupling.run(1)
        self.assertTrue(lmp.timeout)
        self.assertEqual(lmp.swallowed, [])

    def test_empty_rank_null_arrays_still_participate_and_receive_total_energy(self):
        lmp = Driver(empty=True, world_size=2)
        coupling = ExternalCoupling(lmp, self.region(), ConstantCorrection(), comm=EmptyRootComm())
        self.assertEqual(coupling.run(0), 7.5)
        self.assertEqual(lmp.energy_calls[-1][1], 7.5)
        self.assertFalse(lmp.timeout)
        self.assertEqual(lmp.swallowed, [])

    def test_parallel_lammps_without_communicator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "communicator|mpi4py"):
            ExternalCoupling(Driver(world_size=2), self.region(), ConstantCorrection())

    def test_force_write_failure_is_not_swallowed_by_ctypes(self):
        lmp = Driver()
        lmp.forces.setflags(write=False)
        coupling = ExternalCoupling(lmp, self.region(), ConstantCorrection())
        with self.assertRaises(EvaluationError):
            coupling.run(0)
        self.assertTrue(lmp.timeout)
        self.assertEqual(lmp.swallowed, [])

    def test_write_failure_on_another_rank_is_broadcast_and_stops_both_ranks(self):
        root_lmp = Driver(world_size=2)
        root_comm = RemoteWriteFailureComm()
        root = ExternalCoupling(root_lmp, self.region(), ConstantCorrection(), comm=root_comm)
        with self.assertRaisesRegex(EvaluationError, "Remote force buffer"):
            root.run(1)
        self.assertEqual(root_comm.broadcasts[-1], "Remote force buffer is invalid")
        worker_lmp = Driver(world_size=2)
        worker = ExternalCoupling(worker_lmp, self.region(), None, comm=WorkerComm())
        with self.assertRaisesRegex(EvaluationError, "Remote force buffer"):
            worker.run(1)
        for lmp in (root_lmp, worker_lmp):
            self.assertTrue(lmp.timeout)
            self.assertEqual(lmp.swallowed, [])
            np.testing.assert_array_equal(lmp.forces, np.zeros((3, 3)))

    def test_optional_type_mapping_is_checked_for_each_qm_atom(self):
        lmp = Driver()
        coupling = ExternalCoupling(
            lmp, self.region(), ConstantCorrection(), type_symbols=("O", "H")
        )
        coupling.run(0)
        lmp.types[0] = 1
        with self.assertRaisesRegex(EvaluationError, "Element mapping differs"):
            coupling.run(1)
        self.assertEqual(lmp.swallowed, [])

    def test_incomplete_run_is_not_accepted(self):
        lmp = Driver()
        lmp.early_stop = True
        coupling = ExternalCoupling(lmp, self.region(), ConstantCorrection())
        with self.assertRaisesRegex(EvaluationError, "requested step 5"):
            coupling.run(5)
        with self.assertRaises(EvaluationError):
            coupling.run(1)

    def test_changed_box_and_invalid_steps_are_rejected(self):
        lmp = Driver()
        coupling = ExternalCoupling(lmp, self.region(), ConstantCorrection())
        for steps in (-1, 1.0, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                coupling.run(steps)
        lmp.box = ([0, 0, 0], [9, 8, 8], 0.0, 0.0, 0.0, [1, 1, 1], 0)
        with self.assertRaisesRegex(EvaluationError, "fixed simulation box"):
            coupling.run(0)
        self.assertEqual(lmp.swallowed, [])


if __name__ == "__main__":
    unittest.main()
