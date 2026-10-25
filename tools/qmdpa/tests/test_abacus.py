# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""ABACUS settings and bounded process contracts, without executing DFT."""

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np

from qmdpa.abacus import AbacusBackend, map_abacus_coordinates, validate_abacus_settings
from qmdpa.abacus_worker import _FileIOSession, _Session
from qmdpa.core import EvaluationError, Geometry

FAKE_WORKER = r"""
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

mode, log_path = sys.argv[1:]
calls = 0
for line in sys.stdin:
    request = json.loads(line)
    with Path(log_path).open('a') as log:
        log.write(json.dumps({'pid': os.getpid(), 'omp': os.environ.get('OMP_NUM_THREADS'),
                              'request': request}) + '\n')
    response = {'id': request['id']}
    if request['op'] == 'initialize':
        response['status'] = 'ready'
        if request['settings'].get('execution_mode') == 'fileio':
            source = Path(log_path).parent / 'fake-interface.py'
            response.update(execution_mode='fileio', interface={
                'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        if mode == 'combined-deadline':
            time.sleep(0.1)
    elif request['op'] == 'close':
        if mode == 'combined-deadline':
            time.sleep(0.1)
        if mode == 'close-hang':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(3600)
        response['status'] = 'closed'
        print(json.dumps(response), flush=True)
        break
    elif request['op'] == 'evaluate':
        calls += 1
        if mode == 'combined-deadline':
            time.sleep(0.1)
        if mode == 'exit':
            os._exit(7)
        if mode == 'timeout':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(3600)
        if mode == 'malformed':
            print('not-json', flush=True)
            continue
        geometry = request['geometry']
        xyz = geometry['positions_angstrom']
        response.update(status='ok', atom_ids=geometry['atom_ids'],
                        symbols=geometry['symbols'], scf_converged=True,
                        energy_ev=sum(x*x for row in xyz for x in row),
                        forces_ev_per_angstrom=[[-2*x for x in row] for row in xyz])
        if mode == 'wrong-id':
            response['id'] += 1
        if mode == 'wrong-atoms':
            response['atom_ids'] = list(reversed(response['atom_ids']))
        if mode == 'bad-shape':
            response['forces_ev_per_angstrom'] = [[0, 0, 0]]
        if mode == 'nonfinite':
            response['energy_ev'] = float('nan')
        if mode == 'scf-failure' and calls > 1:
            response['scf_converged'] = False
        if mode == 'cold-second-scf-failure':
            evaluations = sum(json.loads(line)['request']['op'] == 'evaluate'
                              for line in Path(log_path).read_text().splitlines())
            if evaluations > 1:
                response['scf_converged'] = False
    else:
        response.update(status='error', error='unexpected operation')
    print(json.dumps(response), flush=True)
"""


OWNER_DEATH_WORKER = r"""
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from qmdpa import abacus_worker

class BlockingSession:
    def __init__(self, request):
        self.directory = Path(request['directory'])
        self.socket_path = '/tmp/ipi_' + request['unixsocket'] if request['unixsocket'] else None
        source = Path(__file__).resolve()
        self.interface = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

    def evaluate(self, request):
        ready = self.directory / 'native-ready.json'
        native = (
            "import json,os,signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "ready=Path(sys.argv[1]); temporary=ready.with_suffix('.tmp'); "
            "temporary.write_text(json.dumps({'pid':os.getpid()})); temporary.replace(ready); "
            "time.sleep(300)"
        )
        self.child = subprocess.Popen([sys.executable, '-u', '-c', native, str(ready)])
        if self.socket_path:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.bind(self.socket_path)
        deadline = time.monotonic() + 3
        while not ready.is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError('Fake native process did not become ready')
            time.sleep(0.01)
        (self.directory / 'owner-may-exit').write_text('ready\n')
        time.sleep(300)

    def close(self):
        pass

abacus_worker._Session = BlockingSession
abacus_worker._FileIOSession = BlockingSession
raise SystemExit(abacus_worker.main())
"""

OWNER_DEATH_HARNESS = r"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from qmdpa import abacus
from qmdpa.core import Geometry

case = json.loads(Path(sys.argv[1]).read_text())
runtime = Path(case['runtime'])
def command(owner_pid, unixsocket):
    value = [sys.executable, '-u', case['worker_script'], '--owner-pid', str(owner_pid)]
    if unixsocket is not None:
        value += ['--unixsocket', unixsocket]
    return value
abacus._worker_command = command
backend = abacus.AbacusBackend(**case['settings'], directory=runtime)
original_popen = subprocess.Popen
def record_child(*args, **kwargs):
    process = original_popen(*args, **kwargs)
    (runtime / 'owned.json').write_text(json.dumps({
        'worker_pid': process.pid,
        'socket_path': '/tmp/ipi_' + backend._socket_name if backend._socket_name else None,
    }))
    return process
subprocess.Popen = record_child

def exit_when_ready():
    deadline = time.monotonic() + 5
    while not list(runtime.rglob('owner-may-exit')):
        if time.monotonic() >= deadline:
            os._exit(11)
        time.sleep(0.01)
    os._exit(23)
threading.Thread(target=exit_when_ready, daemon=True).start()
backend.evaluate(Geometry((1, 2, 3), ('O', 'H', 'H'),
                          [[0, 0, 0], [0.96, 0, 0], [-0.25, 0.94, 0]]))
raise SystemExit(12)
"""


def water(*, atom_ids=(11, 4, 8), symbols=("O", "H", "H"), displacement=0, cell=None):
    return Geometry(
        atom_ids,
        symbols,
        [[0, 0, 0], [0.96 + displacement, 0, 0], [-0.25, 0.94, 0]],
        cell,
    )


class AbacusTestCase(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="qmdpa-abacus-tests-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.executable = self.directory / "fake-abacus"
        self.executable.write_text("#!/bin/sh\nexit 99\n")
        self.executable.chmod(0o700)
        self.settings = {
            "executable": "fake-abacus",
            "executable_sha256": hashlib.sha256(self.executable.read_bytes()).hexdigest(),
            "pseudopotentials": {},
            "supercell_angstrom": [20, 21, 22],
            "coordinate_offset_angstrom": [5, 6, 7],
        }
        for symbol in ("H", "O"):
            path = self.directory / f"{symbol}.upf"
            path.write_text(f"Fake {symbol} pseudopotential; never evaluated\n")
            self.settings["pseudopotentials"][symbol] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    def resolve(self, settings=None):
        return validate_abacus_settings(
            self.settings if settings is None else settings, self.directory
        )


class AbacusSettingsTests(AbacusTestCase):
    def test_resolved_paths_defaults_and_input_immutability(self):
        original = deepcopy(self.settings)
        result = self.resolve()
        self.assertEqual(self.settings, original)
        self.assertEqual(result["engine"], "abacus")
        self.assertEqual(result["execution_mode"], "socket")
        self.assertEqual(result["executable"], str(self.executable))
        self.assertEqual(result["pseudopotentials"]["H"]["path"], str(self.directory / "H.upf"))
        self.assertEqual(result["ecutwfc"], 60)
        self.assertEqual(result["scf_thr"], 1e-9)
        self.assertEqual(result["scf_nmax"], 100)
        self.assertEqual(result["ks_solver"], "cg")
        self.assertEqual(result["coordinate_gauge"], "fixed-offset")
        self.assertEqual(result["timeout_seconds"], 120)
        self.assertEqual((result["charge"], result["multiplicity"]), (0, 1))

    def test_unknown_keys_and_unsupported_state_are_rejected(self):
        for extra in (
            {"directory": "not-a-setting"},
            {"unknown": 1},
            {"engine": "pyscf"},
            {"charge": 1},
            {"charge": False},
            {"multiplicity": 2},
            {"multiplicity": True},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.resolve({**self.settings, **extra})

    def test_execution_mode_default_and_strict_explicit_selection(self):
        self.assertEqual(
            self.resolve(), self.resolve({**self.settings, "execution_mode": "socket"})
        )
        self.assertEqual(
            self.resolve({**self.settings, "execution_mode": "fileio"})["execution_mode"], "fileio"
        )
        for value in (None, True, 1, [], {}, "FILEIO", "", "auto", "fallback"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "execution_mode"):
                self.resolve({**self.settings, "execution_mode": value})

    def test_solver_selection_is_explicit_and_strict(self):
        for solver in ("cg", "dav", "dav_subspace"):
            with self.subTest(solver=solver):
                resolved = self.resolve({**self.settings, "ks_solver": solver})
                self.assertEqual(resolved["ks_solver"], solver)
        for solver in (None, True, 1, [], {}, "CG", "", "lapack", "auto"):
            with self.subTest(solver=solver), self.assertRaisesRegex(ValueError, "ks_solver"):
                self.resolve({**self.settings, "ks_solver": solver})

    def test_coordinate_gauge_is_explicit_and_strict(self):
        for gauge in ("fixed-offset", "arithmetic-centroid"):
            with self.subTest(gauge=gauge):
                self.assertEqual(
                    self.resolve({**self.settings, "coordinate_gauge": gauge})["coordinate_gauge"],
                    gauge,
                )
        for gauge in (None, True, 1, [], {}, "FIXED-OFFSET", "", "mass-weighted", "auto"):
            with self.subTest(gauge=gauge), self.assertRaisesRegex(ValueError, "coordinate_gauge"):
                self.resolve({**self.settings, "coordinate_gauge": gauge})

    def test_coordinate_mapping_does_not_wrap_or_modify_inputs(self):
        value = Geometry((1, 2, 3), ("O", "H", "H"), [[-50, 2, 3], [40, 4, 5], [0, 6, 7]])
        original = value.positions.copy()
        settings = self.resolve({**self.settings, "coordinate_gauge": "arithmetic-centroid"})
        mapped = map_abacus_coordinates(value, settings)
        np.testing.assert_allclose(mapped.mean(axis=0), [5, 6, 7], atol=1e-14, rtol=0)
        np.testing.assert_allclose(mapped[1] - mapped[0], original[1] - original[0])
        self.assertLess(mapped[0, 0], 0)
        self.assertGreater(mapped[1, 0], settings["supercell_angstrom"][0])
        mapped[0, 0] = 123
        np.testing.assert_array_equal(value.positions, original)

    def test_coordinate_mapping_rejects_periodic_and_nonfinite_output(self):
        for gauge in ("fixed-offset", "arithmetic-centroid"):
            settings = self.resolve({**self.settings, "coordinate_gauge": gauge})
            with self.subTest(gauge=gauge), self.assertRaisesRegex(ValueError, "nonperiodic"):
                map_abacus_coordinates(water(cell=np.eye(3) * 20), settings)
        largest = np.finfo(float).max
        value = Geometry((1, 2), ("H", "H"), np.full((2, 3), largest))
        settings = self.resolve({**self.settings, "coordinate_gauge": "arithmetic-centroid"})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            map_abacus_coordinates(value, settings)

    def test_invalid_numerical_parameters_are_rejected(self):
        for key in ("ecutwfc", "scf_thr", "timeout_seconds"):
            for value in (True, 0, -1, float("nan"), float("inf"), "1"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.resolve({**self.settings, key: value})
        for value in (True, 0, -1, 2.5):
            with self.subTest(scf_nmax=value), self.assertRaises(ValueError):
                self.resolve({**self.settings, "scf_nmax": value})
        for key, values in (
            ("supercell_angstrom", ([20, 20], [20, 0, 20], [20, True, 20], [20, float("inf"), 20])),
            ("coordinate_offset_angstrom", ([1, 2], [1, float("nan"), 3], [1, True, 3])),
        ):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.resolve({**self.settings, key: value})

    def test_file_identity_permissions_and_element_names_are_checked(self):
        modified = deepcopy(self.settings)
        modified["executable_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.resolve(modified)
        modified = deepcopy(self.settings)
        modified["pseudopotentials"]["O"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.resolve(modified)
        modified = deepcopy(self.settings)
        modified["pseudopotentials"]["Unobtainium"] = modified["pseudopotentials"].pop("O")
        with self.assertRaises(ValueError):
            self.resolve(modified)
        self.executable.chmod(0o600)
        with self.assertRaises(ValueError):
            self.resolve()


@unittest.skipUnless(os.name == "posix", "The ABACUS process supervisor requires POSIX")
class AbacusProcessTests(AbacusTestCase):
    def backend(self, mode="success", timeout=2):
        settings = self.resolve()
        settings.pop("engine")
        settings["timeout_seconds"] = timeout
        path = self.directory / f"run-{mode}"
        log = self.directory / f"worker-{mode}.jsonl"
        command = [sys.executable, "-u", "-c", FAKE_WORKER, mode, str(log)]
        replacement = patch("qmdpa.abacus._worker_command", return_value=command)
        replacement.start()
        self.addCleanup(replacement.stop)
        backend = AbacusBackend(**settings, directory=path)
        self.addCleanup(backend.close)
        return backend, path, log

    @staticmethod
    def messages(log):
        return [json.loads(line) for line in log.read_text().splitlines()]

    def test_worker_is_persistent_and_child_environment_is_isolated(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "7"}):
            backend, _, log = self.backend()
            first = backend.evaluate(water())
            second = backend.evaluate(water(displacement=0.02))
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "7")
        self.assertNotEqual(first.energy, second.energy)
        np.testing.assert_allclose(first.forces, -2 * water().positions)
        messages = self.messages(log)
        self.assertEqual(
            [row["request"]["op"] for row in messages], ["initialize", "evaluate", "evaluate"]
        )
        self.assertEqual(len({row["pid"] for row in messages}), 1)
        self.assertEqual({row["omp"] for row in messages}, {"1"})
        ids = [row["request"]["id"] for row in messages]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(backend.statistics["successful_evaluations"], 2)
        self.assertEqual(backend.statistics["failed_evaluations"], 0)
        self.assertGreater(backend.statistics["wall_seconds"], 0)
        backend.close()
        backend.close()
        self.assertEqual(sum(row["request"]["op"] == "close" for row in self.messages(log)), 1)

    def test_explicit_centroid_setting_reaches_worker_with_unmapped_input(self):
        self.settings["coordinate_gauge"] = "arithmetic-centroid"
        backend, _, log = self.backend()
        value = water()
        backend.evaluate(value)
        messages = self.messages(log)
        self.assertEqual(
            messages[0]["request"]["settings"]["coordinate_gauge"], "arithmetic-centroid"
        )
        self.assertEqual(
            messages[1]["request"]["geometry"]["positions_angstrom"], value.positions.tolist()
        )

    def test_periodic_and_unknown_element_inputs_do_not_launch_a_worker(self):
        for label, geometry in (
            ("periodic", water(cell=np.eye(3) * 20)),
            ("unknown-element", water(symbols=("C", "H", "H"))),
        ):
            with self.subTest(label=label):
                backend, path, log = self.backend(mode=label)
                with self.assertRaises(EvaluationError):
                    backend.evaluate(geometry)
                self.assertFalse(log.exists())
                self.assertTrue((path / "failure.json").is_file())

    def test_ids_and_symbols_cannot_change_in_a_session(self):
        for label, changed in (
            ("ids", water(atom_ids=(12, 4, 8))),
            ("symbols", water(symbols=("H", "O", "H"))),
        ):
            with self.subTest(label=label):
                backend, _, log = self.backend(mode=label)
                backend.evaluate(water())
                with self.assertRaises(EvaluationError):
                    backend.evaluate(changed)
                self.assertEqual(
                    sum(row["request"]["op"] == "evaluate" for row in self.messages(log)), 1
                )
                backend.close()

    def test_bad_responses_and_worker_exit_are_permanent_failures(self):
        for mode in ("malformed", "wrong-id", "wrong-atoms", "bad-shape", "nonfinite", "exit"):
            with self.subTest(mode=mode):
                backend, path, log = self.backend(mode)
                with self.assertRaises(EvaluationError):
                    backend.evaluate(water())
                self.assertTrue((path / "failure.json").is_file())
                count = len(self.messages(log))
                with self.assertRaises(EvaluationError):
                    backend.evaluate(water())
                self.assertEqual(len(self.messages(log)), count)
                self.assertEqual(backend.statistics["successful_evaluations"], 0)
                self.assertEqual(backend.statistics["failed_evaluations"], 1)
                self.assertIsNone(backend._process)
                backend.close()

    def test_scf_failure_does_not_return_the_previous_success(self):
        backend, path, _ = self.backend("scf-failure")
        backend.evaluate(water())
        with self.assertRaises(EvaluationError):
            backend.evaluate(water(displacement=0.02))
        self.assertTrue((path / "failure.json").is_file())
        self.assertEqual(backend.statistics["successful_evaluations"], 1)
        self.assertGreaterEqual(backend.statistics["failed_evaluations"], 1)

    def test_evaluation_timeout_is_bounded_and_never_restarts(self):
        backend, path, log = self.backend("timeout", timeout=0.25)
        start = time.monotonic()
        with self.assertRaises(EvaluationError):
            backend.evaluate(water())
        self.assertLess(time.monotonic() - start, 5)
        self.assertTrue((path / "failure.json").is_file())
        count = len(self.messages(log))
        with self.assertRaises(EvaluationError):
            backend.evaluate(water())
        self.assertEqual(len(self.messages(log)), count)
        self.assertEqual(backend.statistics["failed_evaluations"], 1)
        self.assertIsNone(backend._process)

    def test_close_timeout_is_bounded_nonthrowing_and_idempotent(self):
        backend, path, _ = self.backend("close-hang", timeout=0.25)
        backend.evaluate(water())
        start = time.monotonic()
        backend.close()
        backend.close()
        self.assertLess(time.monotonic() - start, 5)
        self.assertTrue((path / "cleanup.json").is_file())
        self.assertIsNone(backend._process)

    def test_group_signals_precede_reaping_on_success_and_failure(self):
        for mode in ("success", "scf-failure"):
            with self.subTest(mode=mode):
                self._assert_group_signals_precede_reaping(mode)

    def _assert_group_signals_precede_reaping(self, mode):
        backend, _, _ = self.backend(mode)
        backend.evaluate(water())
        process = backend._process
        events = []
        original_killpg = os.killpg
        original_wait = process.wait

        def signal_owned_group(pgid, signum):
            # Check real kernel ownership immediately before each signal.
            # WNOWAIT leaves a dead child waitable and its PID reserved.
            self.assertIsNone(process.returncode)
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            events.append(("signal", signum))
            return original_killpg(pgid, signum)

        def reap_after_signals(*args, **kwargs):
            self.assertEqual(events, [("signal", signal.SIGTERM), ("signal", signal.SIGKILL)])
            events.append(("reap", None))
            return original_wait(*args, **kwargs)

        with (
            patch("qmdpa.abacus.os.killpg", side_effect=signal_owned_group),
            patch.object(process, "wait", side_effect=reap_after_signals),
            patch.object(process, "poll", side_effect=AssertionError("Premature poll")),
        ):
            if mode == "success":
                backend.close()
            else:
                with self.assertRaises(EvaluationError):
                    backend.evaluate(water(displacement=0.02))
            backend.close()
        self.assertEqual(
            events,
            [("signal", signal.SIGTERM), ("signal", signal.SIGKILL), ("reap", None)],
        )
        self.assertIsNone(backend._process)

    def test_already_reaped_worker_is_never_signaled_by_group_id(self):
        backend, path, _ = self.backend()
        backend.evaluate(water())
        process = backend._process
        process.terminate()
        process.wait(timeout=2)
        with patch("qmdpa.abacus.os.killpg") as kill_group:
            backend.close()
        kill_group.assert_not_called()
        self.assertIsNone(backend._process)
        self.assertIn("already reaped", (path / "cleanup.json").read_text())

    def test_missing_nonreaping_wait_support_rejected_before_launch(self):
        with (
            patch("qmdpa.abacus.os.waitid", None),
            self.assertRaisesRegex(ValueError, "waitid WNOWAIT"),
        ):
            self.backend()
        self.assertFalse((self.directory / "run-success").exists())

    def test_failure_cleanup_reaps_worker_and_close_never_signals_old_group(self):
        backend, _, _ = self.backend("scf-failure")
        backend.evaluate(water())
        process = backend._process
        with patch("qmdpa.abacus.os.killpg", wraps=os.killpg) as kill_group:
            with self.assertRaises(EvaluationError):
                backend.evaluate(water(displacement=0.02))
            self.assertIsNone(backend._process)
            self.assertIsNotNone(process.poll())
            signals = list(kill_group.call_args_list)
            self.assertTrue(signals)
            backend.close()
            backend.close()
            self.assertEqual(kill_group.call_args_list, signals)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Uses Linux process state")
    def test_owner_death_kills_worker_native_descendant_and_unique_socket(self):
        self._assert_owner_death("socket")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Uses Linux process state")
    def test_cold_owner_death_kills_worker_and_native_descendant_without_socket(self):
        self._assert_owner_death("fileio")

    def _assert_owner_death(self, execution_mode):
        runtime = self.directory / f"run-owner-death-{execution_mode}"
        script = self.directory / "owner-death-worker.py"
        script.write_text(OWNER_DEATH_WORKER)
        settings = self.resolve()
        settings.pop("engine")
        settings["timeout_seconds"] = 30
        settings["execution_mode"] = execution_mode
        case = self.directory / "owner-death-case.json"
        case.write_text(
            json.dumps(
                {
                    "runtime": str(runtime),
                    "worker_script": str(script),
                    "settings": settings,
                }
            )
        )
        output = self.directory / "owner-death.log"

        def running(pid):
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            except FileNotFoundError:
                return False
            return fields[0] not in ("Z", "X")

        owned = None
        with output.open("w") as log:
            harness = subprocess.Popen(
                [sys.executable, "-u", "-c", OWNER_DEATH_HARNESS, str(case)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                self.assertEqual(harness.wait(timeout=7), 23, output.read_text())
                owned = json.loads((runtime / "owned.json").read_text())
                native_directory = (
                    runtime / "evaluations/000001" if execution_mode == "fileio" else runtime
                )
                native = json.loads((native_directory / "native-ready.json").read_text())
                pids = (owned["worker_pid"], native["pid"])
                socket_path = Path(owned["socket_path"]) if owned["socket_path"] else None
                deadline = time.monotonic() + 5
                while any(running(pid) for pid in pids) or (
                    socket_path is not None and socket_path.exists()
                ):
                    if time.monotonic() >= deadline:
                        self.fail("Owner-death watchdog left a running process or socket")
                    time.sleep(0.02)
            finally:
                # Cleanup stays active on assertion failure, including a broken watchdog.
                if owned is None and (runtime / "owned.json").is_file():
                    owned = json.loads((runtime / "owned.json").read_text())
                if owned is not None:
                    try:
                        os.killpg(owned["worker_pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if owned["socket_path"]:
                        Path(owned["socket_path"]).unlink(missing_ok=True)
                if harness.poll() is None:
                    os.killpg(harness.pid, signal.SIGKILL)
                harness.wait(timeout=2)


class _FakeProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSocketIO:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.last_scf_converged = True
        self.results = {
            "energy": -12.375,
            "forces": np.array([[0.12, 0.23, 0.34], [-0.45, -0.56, -0.67], [0.78, 0.89, 0.91]]),
        }
        self.closed = False

    def calculate(self, **kwargs):
        self.calls.append(kwargs)

    def close(self):
        self.closed = True


class _PolynomialSocketIO(_FakeSocketIO):
    """Independent finite-grid energy with the native equal-weight force projection."""

    def calculate(self, **kwargs):
        super().calculate(**kwargs)
        atoms = kwargs["atoms"]
        x = atoms.positions - [4.6, 5.8, 6.7]
        weight = np.array([{"O": 2.3, "H": 0.7}[symbol] for symbol in atoms.symbols])[:, None]
        quadratic = np.array([0.7, 1.1, 1.6])
        cubic = np.array([0.2, -0.15, 0.1])
        linear = np.array([0.3, -0.2, 0.4])
        energy = np.sum(weight * (quadratic * x**2 / 2 + cubic * x**3 / 3) + linear * x)
        self.raw_forces = -weight * (quadratic * x + cubic * x**2) - linear
        self.results = {
            "energy": float(energy),
            "forces": self.raw_forces - self.raw_forces.mean(axis=0),
        }


class _FakeAtoms:
    def __init__(self, *, symbols, positions, cell, pbc):
        self.symbols = tuple(symbols)
        self.positions = np.array(positions, copy=True)
        self.cell = np.array(cell, copy=True)
        self.pbc = pbc


class AbacusWorkerAdapterTests(AbacusTestCase):
    def setUp(self):
        super().setUp()
        modules = {
            name: ModuleType(name)
            for name in ("abacuslite", "ase", "ase.calculators", "ase.calculators.calculator")
        }
        modules["abacuslite"].AbacusProfile = _FakeProfile
        modules["abacuslite"].AbacusSocketIO = _FakeSocketIO
        modules["ase"].Atoms = _FakeAtoms
        self.all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]
        modules["ase.calculators.calculator"].all_changes = self.all_changes
        replacement = patch.dict(sys.modules, modules)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.runtime = self.directory / "adapter"
        self.runtime.mkdir()
        self.session = _Session(
            {
                "settings": self.resolve(),
                "directory": str(self.runtime),
                "unixsocket": "qmdpa-" + "a" * 32,
            }
        )
        self.addCleanup(self.session.close)

    @staticmethod
    def request(geometry):
        return {
            "geometry": {
                "atom_ids": list(geometry.atom_ids),
                "symbols": list(geometry.symbols),
                "positions_angstrom": geometry.positions.tolist(),
            }
        }

    def centroid_session(self, name):
        runtime = self.directory / name
        runtime.mkdir()
        session = _Session(
            {
                "settings": self.resolve(
                    {**self.settings, "coordinate_gauge": "arithmetic-centroid"}
                ),
                "directory": str(runtime),
                "unixsocket": "qmdpa-" + "c" * 32,
            }
        )
        self.addCleanup(session.close)
        return session

    def test_centroid_maps_equal_weights_and_passes_native_results_unchanged(self):
        session = self.centroid_session("centroid-pass-through")
        value = water(symbols=("H", "O", "H"))
        result = session.evaluate(self.request(value))
        actual = session.calculator.calls[0]["atoms"].positions
        expected = value.positions - value.positions.mean(axis=0) + [5, 6, 7]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            result["forces_ev_per_angstrom"], session.calculator.results["forces"]
        )
        self.assertGreater(np.linalg.norm(np.sum(result["forces_ev_per_angstrom"], axis=0)), 0)
        self.assertEqual(result["energy_ev"], session.calculator.results["energy"])

    def test_centroid_worker_energy_derivative_translation_and_permutation(self):
        with patch("abacuslite.AbacusSocketIO", _PolynomialSocketIO):
            session = self.centroid_session("centroid-polynomial")
            permuted_session = self.centroid_session("permuted-polynomial")
        value = water(symbols=("H", "O", "H"))
        original = value.positions.copy()
        first = session.evaluate(self.request(value))
        forces = np.asarray(first["forces_ev_per_angstrom"])
        self.assertGreater(np.max(np.abs(session.calculator.raw_forces.mean(axis=0))), 0.1)
        for step in (1e-4, 3e-5):
            differences = np.zeros((3, 3))
            for atom in range(3):
                for axis in range(3):
                    energies = []
                    for sign in (1, -1):
                        positions = value.positions.copy()
                        positions[atom, axis] += sign * step
                        displaced = Geometry(value.atom_ids, value.symbols, positions)
                        energies.append(session.evaluate(self.request(displaced))["energy_ev"])
                    differences[atom, axis] = -(energies[0] - energies[1]) / (2 * step)
            with self.subTest(step=step):
                np.testing.assert_allclose(forces, differences, atol=1e-8, rtol=0)
        translated = Geometry(value.atom_ids, value.symbols, value.positions + [1.37, -2.19, 0.43])
        shifted = session.evaluate(self.request(translated))
        self.assertAlmostEqual(first["energy_ev"], shifted["energy_ev"], places=12)
        np.testing.assert_allclose(forces, shifted["forces_ev_per_angstrom"], atol=1e-12, rtol=0)
        reverse = value.select(tuple(reversed(value.atom_ids)))
        permuted = permuted_session.evaluate(self.request(reverse))
        self.assertEqual(permuted["atom_ids"], list(reverse.atom_ids))
        self.assertEqual(permuted["symbols"], list(reverse.symbols))
        self.assertAlmostEqual(first["energy_ev"], permuted["energy_ev"], places=12)
        np.testing.assert_allclose(
            forces, permuted["forces_ev_per_angstrom"][::-1], atol=1e-12, rtol=0
        )
        np.testing.assert_array_equal(value.positions, original)

    def test_official_adapter_receives_fixed_physics_and_native_environment(self):
        options = self.session.calculator.kwargs
        profile = options["profile"].kwargs
        self.assertEqual(
            shlex.split(profile["command"]),
            [
                "env",
                "-u",
                "LD_PRELOAD",
                "-u",
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                str(self.executable),
            ],
        )
        self.assertNotIn("omp_num_threads", profile)
        self.assertEqual(
            options["inp"],
            {
                "calculation": "scf",
                "basis_type": "pw",
                "dft_functional": "PBE",
                "ecutwfc": 60.0,
                "scf_thr": 1e-9,
                "scf_nmax": 100,
                "ks_solver": "cg",
                "nspin": 1,
                "smearing_method": "fixed",
                "symmetry": -1,
                "cal_force": 1,
                "cal_stress": 0,
            },
        )
        self.assertEqual(
            options["kpts"],
            {
                "mode": "mp-sampling",
                "gamma-centered": True,
                "nk": [1, 1, 1],
                "kshift": [0, 0, 0],
            },
        )
        self.assertEqual(options["pseudopotentials"], {"H": "H.upf", "O": "O.upf"})
        for symbol in ("H", "O"):
            self.assertEqual(
                (Path(profile["pseudo_dir"]) / f"{symbol}.upf").read_bytes(),
                (self.directory / f"{symbol}.upf").read_bytes(),
            )

    def test_selected_solver_reaches_the_official_socket(self):
        for solver in ("dav", "dav_subspace"):
            with self.subTest(solver=solver):
                runtime = self.directory / solver
                runtime.mkdir()
                session = _Session(
                    {
                        "settings": self.resolve({**self.settings, "ks_solver": solver}),
                        "directory": str(runtime),
                        "unixsocket": "qmdpa-" + "b" * 32,
                    }
                )
                try:
                    self.assertEqual(session.calculator.kwargs["inp"]["ks_solver"], solver)
                finally:
                    session.close()

    def test_reused_calculator_preserves_offset_cell_order_and_result_units(self):
        calculator = self.session.calculator
        geometries = [
            water(symbols=("H", "O", "H")),
            water(symbols=("H", "O", "H"), displacement=0.03),
        ]
        for geometry in geometries:
            result = self.session.evaluate(self.request(geometry))
            self.assertIs(self.session.calculator, calculator)
            self.assertEqual(result["atom_ids"], list(geometry.atom_ids))
            self.assertEqual(result["symbols"], list(geometry.symbols))
            self.assertEqual(result["energy_ev"], calculator.results["energy"])
            np.testing.assert_array_equal(
                result["forces_ev_per_angstrom"], calculator.results["forces"]
            )
        self.assertEqual(len(calculator.calls), 2)
        self.assertEqual(calculator.calls[0]["system_changes"], self.all_changes)
        self.assertEqual(calculator.calls[1]["system_changes"], ["positions"])
        for call, geometry in zip(calculator.calls, geometries):
            self.assertEqual(call["properties"], ["energy", "forces"])
            self.assertEqual(call["atoms"].symbols, geometry.symbols)
            self.assertIs(call["atoms"].pbc, True)
            np.testing.assert_array_equal(call["atoms"].cell, [20, 21, 22])
            np.testing.assert_array_equal(call["atoms"].positions, geometry.positions + [5, 6, 7])

    def test_unconfirmed_scf_never_returns_cached_results(self):
        request = self.request(water())
        self.session.evaluate(request)
        for status in (False, None, 1):
            with self.subTest(status=status):
                self.session.calculator.last_scf_converged = status
                with self.assertRaisesRegex(RuntimeError, "SCF did not report convergence"):
                    self.session.evaluate(request)


@unittest.skipUnless(os.name == "posix", "The ABACUS process supervisor requires POSIX")
class AbacusFileIOProcessTests(AbacusTestCase):
    backend = AbacusProcessTests.backend
    messages = staticmethod(AbacusProcessTests.messages)

    def setUp(self):
        super().setUp()
        self.settings["execution_mode"] = "fileio"
        (self.directory / "fake-interface.py").write_text("# Stable mock official interface\n")

    def test_every_evaluation_uses_a_fresh_worker_and_checked_manifest(self):
        backend, path, log = self.backend()
        first = backend.evaluate(water())
        repeated = backend.evaluate(water())
        displaced = backend.evaluate(water(displacement=0.02))
        self.assertEqual(first.energy, repeated.energy)
        self.assertNotEqual(first.energy, displaced.energy)
        self.assertIsNone(backend._process)
        messages = self.messages(log)
        self.assertEqual(
            [row["request"]["op"] for row in messages], ["initialize", "evaluate", "close"] * 3
        )
        self.assertEqual([row["request"]["id"] for row in messages], [1, 2, 3] * 3)
        self.assertEqual(len({row["pid"] for row in messages}), 3)
        manifest_bytes = (path / "worker.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertNotIn("pid", manifest)
        self.assertEqual(manifest["execution_mode"], "fileio")
        self.assertEqual(
            [item["directory"] for item in manifest["workers"]],
            [f"evaluations/{i:06d}" for i in (1, 2, 3)],
        )
        for item in manifest["workers"]:
            directory = path / item["directory"]
            for key, name in (("sha256", "worker.json"), ("lifecycle_sha256", "lifecycle.json")):
                self.assertEqual(
                    item[key], hashlib.sha256((directory / name).read_bytes()).hexdigest()
                )
            worker = json.loads((directory / "worker.json").read_text())
            self.assertIsNone(worker["unixsocket"])
            self.assertEqual(worker["interface"], manifest["interface"])
            self.assertEqual(
                json.loads((directory / "lifecycle.json").read_text()),
                {"closed": True, "returncode": 0, "cleanup_error": None},
            )
        backend.close()
        backend.close()
        self.assertEqual((path / "worker.json").read_bytes(), manifest_bytes)
        self.assertEqual(backend.statistics["successful_evaluations"], 3)

    def test_second_scf_failure_keeps_first_result_and_never_restarts(self):
        backend, path, log = self.backend("cold-second-scf-failure")
        backend.evaluate(water())
        with self.assertRaises(EvaluationError) as first:
            backend.evaluate(water(displacement=0.02))
        count = len(self.messages(log))
        with self.assertRaises(EvaluationError) as second:
            backend.evaluate(water())
        self.assertEqual(str(first.exception), str(second.exception))
        self.assertEqual(len(self.messages(log)), count)
        self.assertEqual(backend.statistics["successful_evaluations"], 1)
        self.assertEqual(backend.statistics["failed_evaluations"], 1)
        self.assertTrue((path / "failure.json").is_file())
        workers = json.loads((path / "worker.json").read_text())["workers"]
        self.assertEqual(len(workers), 2)
        lifecycle = json.loads((path / workers[-1]["directory"] / "lifecycle.json").read_text())
        self.assertTrue(lifecycle["closed"])
        self.assertIsNone(lifecycle["cleanup_error"])

    def test_one_deadline_includes_start_calculation_and_close(self):
        for mode in ("combined-deadline", "close-hang"):
            with self.subTest(mode=mode):
                backend, path, _ = self.backend(mode, timeout=0.25)
                started = time.monotonic()
                with self.assertRaises(EvaluationError):
                    backend.evaluate(water())
                self.assertLess(time.monotonic() - started, 4)
                self.assertEqual(backend.statistics["successful_evaluations"], 0)
                self.assertIsNone(backend._process)
                self.assertTrue((path / "failure.json").is_file())
                backend.close()

    def test_changed_interface_stops_before_the_next_calculation(self):
        backend, path, log = self.backend()
        backend.evaluate(water())
        (self.directory / "fake-interface.py").write_text("# Changed interface\n")
        with self.assertRaisesRegex(EvaluationError, "interface changed"):
            backend.evaluate(water())
        self.assertEqual(sum(row["request"]["op"] == "evaluate" for row in self.messages(log)), 1)
        self.assertTrue((path / "failure.json").is_file())

    def test_lifecycle_record_failure_cannot_accept_valid_forces(self):
        backend, path, _ = self.backend()
        original = backend._record

        def fail_lifecycle(name, value, **kwargs):
            if name == "lifecycle.json":
                raise OSError("simulated lifecycle disk failure")
            return original(name, value, **kwargs)

        with (
            patch.object(backend, "_record", side_effect=fail_lifecycle),
            self.assertRaises(EvaluationError),
        ):
            backend.evaluate(water())
        self.assertIsNone(backend._process)
        self.assertEqual(backend.statistics["successful_evaluations"], 0)
        self.assertTrue((path / "cleanup.json").is_file())
        self.assertTrue((path / "evaluations/000001/cleanup.json").is_file())

    def test_existing_evaluation_directory_is_not_overwritten(self):
        backend, path, log = self.backend()
        directory = path / "evaluations/000001"
        directory.mkdir()
        sentinel = directory / "sentinel"
        sentinel.write_text("retained")
        with self.assertRaises(EvaluationError):
            backend.evaluate(water())
        self.assertEqual(sentinel.read_text(), "retained")
        self.assertFalse(log.exists())

    def test_second_directory_collision_preserves_previous_worker_evidence(self):
        backend, path, log = self.backend()
        backend.evaluate(water())
        previous = path / "evaluations/000001"
        evidence = {item: item.read_bytes() for item in previous.rglob("*") if item.is_file()}
        manifest = (path / "worker.json").read_bytes()
        messages = self.messages(log)
        attempted = path / "evaluations/000002"
        attempted.mkdir()
        sentinel = attempted / "sentinel"
        sentinel.write_text("retained")
        with self.assertRaises(EvaluationError):
            backend.evaluate(water(displacement=0.02))
        failure = json.loads((path / "failure.json").read_text())
        self.assertEqual(
            failure["request"],
            {"op": "create_evaluation_directory", "directory": str(attempted)},
        )
        self.assertEqual((path / "worker.json").read_bytes(), manifest)
        self.assertEqual(self.messages(log), messages)
        self.assertEqual({item: item.read_bytes() for item in evidence}, evidence)
        self.assertEqual(sentinel.read_text(), "retained")
        self.assertEqual(set(attempted.iterdir()), {sentinel})
        self.assertIsNone(backend._last_lifecycle)
        self.assertEqual(backend.statistics["successful_evaluations"], 1)
        self.assertEqual(backend.statistics["failed_evaluations"], 1)


class _FakeFileIO(_FakeSocketIO):
    def calculate(self, **kwargs):
        super().calculate(**kwargs)
        directory = self.kwargs["directory"]
        output = directory / "OUT.ABACUS"
        output.mkdir()
        for name in ("INPUT", "STRU", "KPT"):
            (directory / name).write_text("mock native input\n")
        (output / "running_scf.log").write_text(
            "#ELEC ITER# 3\nElectron density deviation = 1e-12\n#SCF IS CONVERGED#\n"
        )


class AbacusFileIOAdapterTests(AbacusTestCase):
    def setUp(self):
        super().setUp()
        modules = {
            name: ModuleType(name)
            for name in ("abacuslite", "ase", "ase.calculators", "ase.calculators.calculator")
        }
        modules["abacuslite"].AbacusProfile = _FakeProfile
        modules["abacuslite"].Abacus = _FakeFileIO
        modules["ase"].Atoms = _FakeAtoms
        self.all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]
        modules["ase.calculators.calculator"].all_changes = self.all_changes
        replacement = patch.dict(sys.modules, modules)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.settings["execution_mode"] = "fileio"
        self.settings["coordinate_gauge"] = "arithmetic-centroid"

    def session(self, name="fileio"):
        directory = self.directory / name
        directory.mkdir()
        return _FileIOSession(
            {"settings": self.resolve(), "directory": str(directory), "unixsocket": None}
        )

    def test_official_fileio_contract_and_raw_evidence(self):
        session = self.session()
        geometry = water(symbols=("H", "O", "H"))
        response = session.evaluate(AbacusWorkerAdapterTests.request(geometry))
        call = session.calculator.calls[0]
        self.assertEqual(call["system_changes"], self.all_changes)
        self.assertEqual(call["properties"], ["energy", "forces"])
        self.assertNotIn("unixsocket", session.calculator.kwargs)
        self.assertNotIn("socket_driver", session.calculator.kwargs["inp"])
        self.assertEqual(session.calculator.kwargs["inp"]["ks_solver"], "cg")
        np.testing.assert_array_equal(
            call["atoms"].positions,
            geometry.positions - geometry.positions.mean(axis=0) + [5, 6, 7],
        )
        np.testing.assert_array_equal(
            response["forces_ev_per_angstrom"], session.calculator.results["forces"]
        )
        self.assertGreater(np.linalg.norm(np.sum(response["forces_ev_per_angstrom"], axis=0)), 0)
        result = json.loads((session.directory / "result.json").read_text())
        self.assertEqual(result["evidence"]["scf_iterations"], 3)
        self.assertEqual(result["evidence"]["terminal_density_deviation"], 1e-12)
        self.assertEqual(result["atom_ids"], list(geometry.atom_ids))
        self.assertEqual(result["symbols"], list(geometry.symbols))
        self.assertEqual(
            result["evidence"]["scf_log_sha256"],
            hashlib.sha256(
                (session.directory / result["evidence"]["scf_log"]).read_bytes()
            ).hexdigest(),
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            session.evaluate(AbacusWorkerAdapterTests.request(geometry))
        self.assertEqual(len(session.calculator.calls), 1)
        session.close()
        session.close()

    def test_missing_or_unconverged_scf_and_invalid_results_fail_closed(self):
        for index, mode in enumerate(
            (
                "missing-log",
                "unconverged",
                "multiple",
                "residual",
                "iterations",
                "nonfinite",
                "bad-shape",
                "missing-input",
            )
        ):
            with self.subTest(mode=mode):
                session = self.session(str(index))
                original = session.calculator.calculate

                def corrupt(original=original, session=session, mode=mode, **kwargs):
                    original(**kwargs)
                    log = session.calculation / "OUT.ABACUS/running_scf.log"
                    if mode == "missing-log":
                        log.unlink()
                    elif mode == "unconverged":
                        log.write_text(
                            log.read_text().replace(
                                "#SCF IS CONVERGED#", "!!SCF IS NOT CONVERGED!!"
                            )
                        )
                    elif mode == "multiple":
                        log.write_text(log.read_text() + "#SCF IS CONVERGED#\n")
                    elif mode == "residual":
                        log.write_text(log.read_text().replace("1e-12", "1e-4"))
                    elif mode == "iterations":
                        log.write_text(log.read_text().replace("ITER# 3", "ITER# 101"))
                    elif mode == "nonfinite":
                        session.calculator.results["energy"] = float("nan")
                    elif mode == "bad-shape":
                        session.calculator.results["forces"] = [[0, 0, 0]]
                    elif mode == "missing-input":
                        (session.calculation / "STRU").unlink()

                with (
                    patch.object(session.calculator, "calculate", side_effect=corrupt),
                    self.assertRaises((ValueError, RuntimeError, OSError)),
                ):
                    session.evaluate(AbacusWorkerAdapterTests.request(water()))
                self.assertFalse((session.directory / "result.json").exists())
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    session.evaluate(AbacusWorkerAdapterTests.request(water()))


if __name__ == "__main__":
    unittest.main()
