# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Driver wall-time accounting across accepted checkpoints and failures."""

import json
import shlex
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from qmdpa import cli
from qmdpa.config import file_digest


class DriverMetricsTests(unittest.TestCase):
    def run_fixture(self, output, *, failure=None, restart=None, steps=1):
        clock = SimpleNamespace(now=0.0)
        drivers = []
        couplings = []
        config = {
            "periodic": False,
            "data_file": "unused.data",
            "timestep_fs": 0.25,
            "steps": steps,
            "sample_interval": 1,
            "physics_sha256": "same-physics",
        }

        class Driver:
            def __init__(self, **kwargs):
                clock.now += 2.0
                self.step = 0
                self.closed = False
                self.commands = []
                drivers.append(self)

            def get_mpi_comm(self):
                return None

            def extract_setting(self, name):
                return 1

            def extract_global(self, name):
                return self.step

            def commands_list(self, commands):
                self.commands.extend(commands)

            def command(self, command):
                self.commands.append(command)
                words = shlex.split(command)
                if words[0] == "read_restart":
                    self.step = int(Path(words[1]).read_text())
                elif words[0] == "write_restart":
                    clock.now += 23.0
                    if failure == "checkpoint":
                        raise RuntimeError("native checkpoint failed")
                    Path(words[1]).write_text(str(self.step))

            def close(self):
                clock.now += 31.0
                self.closed = True

        class Coupling:
            def __init__(self, lmp):
                self.lmp = lmp
                self.calls = 0
                self.failed = 0
                couplings.append(self)

            def run(self, count):
                clock.now += 11.0
                self.calls += 1
                if failure == "run0" or (failure == "run" and self.calls == 2):
                    self.failed += 1
                    raise RuntimeError("calculator failed")
                self.lmp.step += count

            def statistics(self):
                return {
                    "schema_version": 1,
                    "rank": 0,
                    "callback_calls": self.calls,
                    "calculator_attempts": self.calls,
                    "calculator_successes": self.calls - self.failed,
                    "calculator_failures": self.failed,
                    "consecutive_repeated_geometry_attempts": 0,
                    "calculator_wall_seconds": 7.0 * self.calls,
                }

        def load_config(path):
            clock.now += 3.0
            return config

        def setup_pair(lmp, settings):
            clock.now += 5.0

        def make_coupling(lmp, comm, settings):
            clock.now += 7.0
            if failure == "initialization":
                raise RuntimeError("provider initialization failed")
            return Coupling(lmp)

        def frame(lmp, comm):
            clock.now += 19.0
            return {
                "step": lmp.step,
                "potential_eV": -1.0,
                "atoms": [
                    {
                        "id": 1,
                        "x": [0.0, 0.0, 0.0],
                        "v": [0.0, 0.0, 0.0],
                        "f": [0.0, 0.0, 0.0],
                        "image": [0, 0, 0],
                    }
                ],
            }

        def digest(path):
            clock.now += 29.0
            if failure == "checkpoint-hash":
                raise OSError("checkpoint hash failed")
            return file_digest(path)

        error = None
        with (
            patch.dict(sys.modules, {"lammps": SimpleNamespace(lammps=Driver)}),
            patch("qmdpa.cli.time.perf_counter", side_effect=lambda: clock.now),
            patch("qmdpa.cli._frame", side_effect=frame),
            patch("qmdpa.cli.file_digest", side_effect=digest),
            patch("qmdpa.runtime.file_digest", side_effect=digest),
        ):
            try:
                cli._run(
                    "unused.json",
                    output,
                    restart=restart,
                    config_loader=load_config,
                    setup_pair=setup_pair,
                    make_coupling=make_coupling,
                )
            except RuntimeError as exc:
                error = exc
        self.assertTrue(drivers[0].closed)
        name = "failure.json" if error else "result.json"
        report = json.loads((output / name).read_text())
        json.dumps(report, allow_nan=False)
        return report, error, drivers[0], couplings

    def test_success_times_only_declared_work_and_keeps_checkpoint_schema(self):
        with TemporaryDirectory(prefix="qmdpa-metrics-tests-") as temporary:
            output = Path(temporary) / "run"
            report, error, driver, _ = self.run_fixture(output)
            self.assertIsNone(error)
            metrics = report["metrics"]
            self.assertEqual(metrics["scope"], "owner_rank_host_wall")
            self.assertEqual(metrics["clock"], "time.perf_counter")
            self.assertEqual(metrics["schema_version"], 1)
            self.assertEqual(metrics["initialization_wall_seconds"], 17.0)
            self.assertEqual(metrics["run_wall_seconds"], 22.0)
            self.assertEqual(metrics["checkpoint_wall_seconds"], 52.0)
            self.assertEqual(metrics["coupling"]["calculator_attempts"], 2)
            self.assertEqual(metrics["coupling"]["calculator_failures"], 0)
            self.assertEqual(report["final_step"], 1)
            checkpoint = json.loads((output / "checkpoint-000000000001.json").read_text())
            self.assertEqual(
                set(checkpoint), {"physics_sha256", "restart_file", "restart_sha256", "frame"}
            )
            self.assertEqual(checkpoint["physics_sha256"], "same-physics")
            self.assertEqual(
                checkpoint["restart_sha256"], file_digest(output / checkpoint["restart_file"])
            )
            self.assertEqual(checkpoint["frame"]["step"], 1)
            self.assertEqual(len((output / "frames.jsonl").read_text().splitlines()), 2)
            self.assertEqual(sum(c.startswith("write_restart ") for c in driver.commands), 1)
            self.assertFalse((output / "failure.json").exists())

    def test_failure_timings_include_failed_work_and_preserve_primary_error(self):
        for failure, run_seconds, checkpoint_seconds, attempts, failures in (
            ("initialization", 0.0, 0.0, None, None),
            ("run0", 11.0, 0.0, 1, 1),
            ("run", 22.0, 0.0, 2, 1),
            ("checkpoint", 22.0, 23.0, 2, 0),
            ("checkpoint-hash", 22.0, 52.0, 2, 0),
        ):
            with (
                self.subTest(failure=failure),
                TemporaryDirectory(prefix="qmdpa-metrics-tests-") as temporary,
            ):
                output = Path(temporary) / "run"
                report, error, _, _ = self.run_fixture(output, failure=failure)
                self.assertIsNotNone(error)
                self.assertIn(str(error), report["error"])
                self.assertEqual(report["status"], "failed")
                metrics = report["metrics"]
                self.assertEqual(metrics["initialization_wall_seconds"], 17.0)
                self.assertEqual(metrics["run_wall_seconds"], run_seconds)
                self.assertEqual(metrics["checkpoint_wall_seconds"], checkpoint_seconds)
                if attempts is None:
                    self.assertIsNone(metrics["coupling"])
                else:
                    self.assertEqual(metrics["coupling"]["calculator_attempts"], attempts)
                    self.assertEqual(metrics["coupling"]["calculator_failures"], failures)
                self.assertFalse((output / "result.json").exists())
                self.assertFalse(list(output.glob("checkpoint-*.json")))

    def test_new_and_restarted_runs_start_new_measurements(self):
        with TemporaryDirectory(prefix="qmdpa-metrics-tests-") as temporary:
            root = Path(temporary)
            first, _, _, _ = self.run_fixture(root / "first")
            checkpoint = root / "first" / "checkpoint-000000000001.json"
            resumed, error, _, _ = self.run_fixture(root / "resumed", restart=checkpoint)
            self.assertIsNone(error)
            self.assertEqual(resumed["final_step"], 2)
            self.assertEqual(resumed["metrics"]["initialization_wall_seconds"], 46.0)
            self.assertEqual(resumed["metrics"]["run_wall_seconds"], 22.0)
            self.assertEqual(resumed["metrics"]["checkpoint_wall_seconds"], 52.0)
            self.assertEqual(resumed["metrics"]["coupling"], first["metrics"]["coupling"])
            empty, error, _, _ = self.run_fixture(root / "zero-steps", steps=0)
            self.assertIsNone(error)
            self.assertEqual(empty["metrics"]["run_wall_seconds"], 11.0)
            self.assertEqual(empty["metrics"]["checkpoint_wall_seconds"], 0.0)
            self.assertEqual(empty["metrics"]["coupling"]["calculator_attempts"], 1)
            self.assertFalse(list((root / "zero-steps").glob("checkpoint-*.json")))


if __name__ == "__main__":
    unittest.main()
