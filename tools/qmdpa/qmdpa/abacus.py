# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Bounded POSIX supervision of explicit ABACUS/ASE socket or cold FileIO sessions.

The molecular input is evaluated in an explicit finite periodic supercell.
Coordinates use an explicit fixed-offset or arithmetic-centroid mapping.
Stress and cell relaxation are excluded. ASE and abacuslite are imported only
in the isolated worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_ELEMENTS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
    "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf "
    "Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_DEFAULTS = {
    "engine": "abacus",
    "execution_mode": "socket",
    "ecutwfc": 60.0,
    "scf_thr": 1e-9,
    "scf_nmax": 100,
    "ks_solver": "cg",
    "coordinate_gauge": "fixed-offset",
    "charge": 0,
    "multiplicity": 1,
    "timeout_seconds": 120.0,
}
_REQUIRED = {
    "executable",
    "executable_sha256",
    "pseudopotentials",
    "supercell_angstrom",
    "coordinate_offset_angstrom",
}
_MAX_MESSAGE = 4 * 1024 * 1024


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def _number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _identity(path, digest, base_dir, name):
    if not isinstance(path, (str, os.PathLike)) or not str(path):
        raise ValueError(f"{name} requires a file path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError(f"{name} requires a full SHA256")
    resolved = (Path(base_dir) / path).resolve()
    if not resolved.is_file() or _digest(resolved) != digest.lower():
        raise ValueError(f"{name} file is missing or its SHA256 differs")
    return str(resolved), digest.lower()


def validate_abacus_settings(settings, base_dir):
    """Resolve and validate the JSON QM schema without importing an engine.

    ``directory`` is supplied separately by the runtime, never by the physics
    configuration. File paths are relative to the configuration's directory.
    """
    if not isinstance(settings, dict):
        raise ValueError("ABACUS settings must be a mapping")
    unknown = set(settings).difference(_REQUIRED | _DEFAULTS.keys())
    missing = _REQUIRED.difference(settings)
    if unknown or missing:
        raise ValueError(
            f"ABACUS settings have unknown {sorted(unknown)} or missing {sorted(missing)} keys"
        )
    result = {**_DEFAULTS, **settings}
    if result["engine"] != "abacus":
        raise ValueError("ABACUS settings require engine='abacus'")
    if not isinstance(result["execution_mode"], str) or result["execution_mode"] not in (
        "socket",
        "fileio",
    ):
        raise ValueError("execution_mode must be socket or fileio")
    for key, expected in (("charge", 0), ("multiplicity", 1)):
        if type(result[key]) is not int or result[key] != expected:
            raise ValueError(
                "ABACUS currently requires neutral singlets (charge=0, multiplicity=1)"
            )
    if not isinstance(result["ks_solver"], str) or result["ks_solver"] not in (
        "cg",
        "dav",
        "dav_subspace",
    ):
        raise ValueError("ks_solver must be cg, dav or dav_subspace")
    if not isinstance(result["coordinate_gauge"], str) or result["coordinate_gauge"] not in (
        "fixed-offset",
        "arithmetic-centroid",
    ):
        raise ValueError("coordinate_gauge must be fixed-offset or arithmetic-centroid")
    for key in ("ecutwfc", "scf_thr", "timeout_seconds"):
        result[key] = _number(result[key], key)
    if type(result["scf_nmax"]) is not int or result["scf_nmax"] <= 0:
        raise ValueError("scf_nmax must be a positive integer")
    for key in ("supercell_angstrom", "coordinate_offset_angstrom"):
        value = result[key]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{key} must contain three numbers")
        result[key] = [_number(x, key, positive=key == "supercell_angstrom") for x in value]
    result["executable"], result["executable_sha256"] = _identity(
        result["executable"], result["executable_sha256"], base_dir, "ABACUS executable"
    )
    if not os.access(result["executable"], os.X_OK):
        raise ValueError("ABACUS executable does not have execute permission")
    potentials = result["pseudopotentials"]
    if not isinstance(potentials, dict) or not potentials:
        raise ValueError("pseudopotentials must map elements to file identities")
    result["pseudopotentials"] = {}
    for symbol, item in potentials.items():
        if symbol not in _ELEMENTS or not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Each pseudopotential requires an element and only path/sha256")
        path, digest = _identity(
            item["path"], item["sha256"], base_dir, f"{symbol} pseudopotential"
        )
        result["pseudopotentials"][symbol] = {"path": path, "sha256": digest}
    return result


def map_abacus_coordinates(geometry, settings):
    """Map the complete finite molecular set using validated ABACUS settings.

    The centroid is the equal-weight arithmetic mean of every supplied atom,
    independent of elements or masses. The offset is the centroid's target
    when centering is explicit. No wrapping or force transformation is done.
    """
    if geometry.periodic:
        raise ValueError("ABACUS coordinate mapping requires nonperiodic input geometry")
    gauge = settings["coordinate_gauge"]
    with np.errstate(over="ignore", invalid="ignore"):
        if gauge == "fixed-offset":
            positions = geometry.positions + settings["coordinate_offset_angstrom"]
        elif gauge == "arithmetic-centroid":
            positions = (
                geometry.positions
                - geometry.positions.mean(axis=0)
                + settings["coordinate_offset_angstrom"]
            )
        else:
            raise ValueError("coordinate_gauge must be fixed-offset or arithmetic-centroid")
    if not np.isfinite(positions).all():
        raise ValueError("Mapped ABACUS coordinates must be finite")
    return positions


def _worker_command(owner_pid, unixsocket):
    # The owner and socket identity must be available before initialization, so
    # an owner dying between Popen and the first request cannot leave a worker.
    command = [sys.executable, "-u", "-m", "qmdpa.abacus_worker", "--owner-pid", str(owner_pid)]
    if unixsocket is not None:
        command += ["--unixsocket", unixsocket]
    return command


class AbacusBackend:
    """Fixed electronic settings, with bounded requests and owned process cleanup.

    POSIX process groups contain the worker and the serial ABACUS executable.
    Any failed evaluation permanently invalidates this instance. ``close`` is
    idempotent and never masks a calculation exception; cleanup diagnostics and
    the last failed request remain in the fresh runtime directory.
    """

    def __init__(
        self,
        *,
        executable,
        executable_sha256,
        pseudopotentials,
        supercell_angstrom,
        coordinate_offset_angstrom,
        coordinate_gauge="fixed-offset",
        execution_mode="socket",
        ecutwfc=60,
        scf_thr=1e-9,
        scf_nmax=100,
        ks_solver="cg",
        charge=0,
        multiplicity=1,
        timeout_seconds=120,
        directory,
    ):
        waitid_flags = ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
        if (
            os.name != "posix"
            or not callable(getattr(os, "waitid", None))
            or not all(hasattr(os, name) for name in waitid_flags)
        ):
            raise ValueError("The ABACUS worker requires POSIX process groups and waitid WNOWAIT")
        self._settings = validate_abacus_settings(
            {
                "executable": executable,
                "executable_sha256": executable_sha256,
                "pseudopotentials": pseudopotentials,
                "supercell_angstrom": supercell_angstrom,
                "coordinate_offset_angstrom": coordinate_offset_angstrom,
                "coordinate_gauge": coordinate_gauge,
                "execution_mode": execution_mode,
                "ecutwfc": ecutwfc,
                "scf_thr": scf_thr,
                "scf_nmax": scf_nmax,
                "ks_solver": ks_solver,
                "charge": charge,
                "multiplicity": multiplicity,
                "timeout_seconds": timeout_seconds,
            },
            Path.cwd(),
        )
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self._process = None
        self._log = None
        self._closed = False
        self._failure = None
        self._identity = None
        self._request_id = 0
        self._buffer = b""
        self._last_request = None
        self._socket_name = "qmdpa-" + uuid.uuid4().hex if execution_mode == "socket" else None
        self._active_directory = self.directory
        self._workers = []
        self._common_interface = None
        self._last_lifecycle = None
        if execution_mode == "fileio":
            (self.directory / "evaluations").mkdir()
        self._statistics = {
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "wall_seconds": 0.0,
        }
        self._record("settings.json", self._settings)

    @property
    def statistics(self):
        """Wall time includes worker initialization and cold per-evaluation cleanup."""
        return dict(self._statistics)

    def _record(self, name, value, *, directory=None):
        ((directory or self.directory) / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )

    def _diagnostic(self, name, value):
        try:
            self._record(name, value)
            if name == "cleanup.json" and self._active_directory != self.directory:
                self._record(name, value, directory=self._active_directory)
        except (OSError, TypeError, ValueError):
            pass  # Preserve the original calculation error even on a full filesystem.

    def _record_workers(self):
        self._record(
            "worker.json",
            {
                "execution_mode": "fileio",
                "interface": self._common_interface,
                "workers": self._workers,
            },
        )

    def _begin_fileio(self):
        if self._process is not None:
            raise RuntimeError("Previous FileIO worker has not been closed")
        relative = f"evaluations/{len(self._workers) + 1:06d}"
        directory = self.directory / relative
        # Allocation failure belongs to this attempt, never the previous worker.
        self._active_directory = self.directory
        self._request_id = 0
        self._buffer = b""
        self._last_request = {"op": "create_evaluation_directory", "directory": str(directory)}
        self._last_lifecycle = None
        directory.mkdir(exist_ok=False)
        self._active_directory = directory
        self._workers.append({"directory": relative, "sha256": None, "lifecycle_sha256": None})
        self._record_workers()

    def _finish_fileio(self, deadline):
        response = self._exchange({"op": "close"}, deadline)
        if response.get("status") != "closed":
            raise ValueError("ABACUS FileIO worker did not confirm closure")
        self._wait_without_reaping(self._process, deadline)
        self._stop_process()
        if self._last_lifecycle != {"closed": True, "returncode": 0, "cleanup_error": None}:
            raise RuntimeError("ABACUS FileIO worker did not exit and clean up successfully")
        if time.monotonic() >= deadline:
            raise TimeoutError("ABACUS FileIO request deadline expired during cleanup")

    def _exchange(self, request, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("ABACUS request deadline has expired")
        self._request_id += 1
        request = {**request, "id": self._request_id}
        self._last_request = request
        data = json.dumps(request, allow_nan=False).encode() + b"\n"
        if len(data) > _MAX_MESSAGE:
            raise ValueError("ABACUS request exceeds the supported message size")
        stream = self._process.stdin
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_WRITE)
            offset = 0
            while offset < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("ABACUS request timed out while sending")
                offset += os.write(stream.fileno(), data[offset:])
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("ABACUS request timed out while waiting for a result")
                data = os.read(self._process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("ABACUS worker exited without a complete response")
                self._buffer += data
                if len(self._buffer) > _MAX_MESSAGE:
                    raise ValueError("ABACUS worker response is too large")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(line)
        if (
            not isinstance(response, dict)
            or type(response.get("id")) is not int
            or response["id"] != request["id"]
        ):
            raise ValueError("ABACUS worker response has the wrong request ID")
        if response.get("status") == "error":
            raise RuntimeError(f"ABACUS worker failed: {response.get('error', 'no diagnostic')}")
        return response

    def _start(self, deadline):
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self._log = (self._active_directory / "worker.stderr.log").open("xb")
        self._process = subprocess.Popen(
            _worker_command(os.getpid(), self._socket_name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            bufsize=0,
            start_new_session=True,
            env=env,
        )
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        response = self._exchange(
            {
                "op": "initialize",
                "settings": self._settings,
                "directory": str(self._active_directory),
                "unixsocket": self._socket_name,
            },
            deadline,
        )
        if response.get("status") != "ready":
            raise ValueError("ABACUS worker did not confirm initialization")
        interface = response.get("interface")
        self._record(
            "worker.json",
            {"pid": self._process.pid, "unixsocket": self._socket_name, "interface": interface},
            directory=self._active_directory,
        )
        if self._settings["execution_mode"] == "fileio":
            self._workers[-1]["sha256"] = _digest(self._active_directory / "worker.json")
            self._record_workers()
            if response.get("execution_mode") != "fileio" or not isinstance(interface, dict):
                raise ValueError(
                    "ABACUS FileIO worker did not identify its execution mode/interface"
                )
            _identity(
                interface.get("path"), interface.get("sha256"), self.directory, "ABACUS interface"
            )
            if self._common_interface is not None and self._common_interface != interface:
                raise ValueError("ABACUS FileIO interface changed between evaluations")
            self._common_interface = interface
            self._record_workers()

    @staticmethod
    def _exited_without_reaping(process):
        """Observe the owned child while keeping its PID reserved for cleanup."""
        if process.returncode is not None:
            raise ChildProcessError("ABACUS worker has already been reaped")
        status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        return status is not None and status.si_pid == process.pid

    def _wait_without_reaping(self, process, deadline):
        while not self._exited_without_reaping(process):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ABACUS worker did not exit before the deadline")
            time.sleep(min(0.02, remaining))

    def _stop_process(self):
        """Never let cleanup replace a calculation exception."""
        process, errors = self._process, []
        try:
            errors = self._terminate_group()
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
            self._diagnostic("cleanup.json", {"errors": errors})
        finally:
            # Do not poll here: it can release ownership before group signaling.
            if self._process is not None and self._process.returncode is not None:
                self._process = None
            if self._settings["execution_mode"] == "fileio" and process is not None:
                lifecycle = {
                    "closed": process.returncode is not None,
                    "returncode": process.returncode,
                    "cleanup_error": "; ".join(errors) if errors else None,
                }
                self._last_lifecycle = lifecycle
                try:
                    self._record("lifecycle.json", lifecycle, directory=self._active_directory)
                    self._workers[-1]["lifecycle_sha256"] = _digest(
                        self._active_directory / "lifecycle.json"
                    )
                    self._record_workers()
                except Exception as exc:  # noqa: BLE001 - preserve the first calculation error
                    lifecycle["cleanup_error"] = f"Failed to record worker lifecycle: {exc}"
                    self._diagnostic("cleanup.json", {"error": lifecycle["cleanup_error"]})

    def _terminate_group(self):
        errors = []
        process = self._process
        if process is not None:
            owned = True
            for signum in (signal.SIGTERM, signal.SIGKILL):
                try:
                    # An exited, unreaped child still reserves its PID/PGID.
                    # Never use the numerical group ID once ownership is lost.
                    self._exited_without_reaping(process)
                except ChildProcessError:
                    errors.append("Worker already reaped; process-group signaling skipped")
                    owned = False
                    self._process = None
                    break
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    errors.append(str(exc))
                if signum == signal.SIGTERM:
                    deadline = time.monotonic() + min(1.0, self._settings["timeout_seconds"])
                    try:
                        self._wait_without_reaping(process, deadline)
                    except TimeoutError:
                        pass
                    except ChildProcessError:
                        # The next ownership check will prevent any further signal.
                        pass
            if owned:
                # This is the first reaping operation, after the final group
                # signal. No code path may signal this group after it succeeds.
                try:
                    process.wait(timeout=min(1.0, self._settings["timeout_seconds"]))
                except subprocess.TimeoutExpired:
                    errors.append("Worker did not exit after group SIGKILL")
                if process.returncode is not None:
                    self._process = None
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError as exc:
                    errors.append(str(exc))
        if self._log is not None:
            self._log.close()
        if self._socket_name is not None:
            try:
                Path("/tmp/ipi_" + self._socket_name).unlink(missing_ok=True)
            except OSError as exc:
                errors.append(str(exc))
        if errors:
            self._diagnostic("cleanup.json", {"errors": errors})
        return errors

    def evaluate(self, geometry):
        from .core import Evaluation, EvaluationError

        if self._failure is not None:
            raise EvaluationError(self._failure)
        if self._closed:
            raise EvaluationError("ABACUS backend is closed")
        started = time.monotonic()
        try:
            if geometry.periodic:
                raise ValueError("ABACUS correction requires nonperiodic input geometry")
            identity = (geometry.atom_ids, geometry.symbols)
            if self._identity is not None and identity != self._identity:
                raise ValueError("ABACUS atom IDs and symbols must stay fixed")
            if any(s not in self._settings["pseudopotentials"] for s in geometry.symbols):
                raise ValueError("An input element has no declared pseudopotential")
            self._identity = identity
            deadline = started + self._settings["timeout_seconds"]
            if self._settings["execution_mode"] == "fileio":
                self._begin_fileio()
            if self._process is None:
                self._start(deadline)
            response = self._exchange(
                {
                    "op": "evaluate",
                    "geometry": {
                        "atom_ids": list(geometry.atom_ids),
                        "symbols": list(geometry.symbols),
                        "positions_angstrom": geometry.positions.tolist(),
                    },
                },
                deadline,
            )
            if response.get("status") != "ok" or response.get("scf_converged") is not True:
                raise ValueError("ABACUS did not provide a converged SCF result")
            if response.get("atom_ids") != list(geometry.atom_ids) or response.get(
                "symbols"
            ) != list(geometry.symbols):
                raise ValueError("ABACUS response changed atom identities or order")
            result = Evaluation(response["energy_ev"], response["forces_ev_per_angstrom"])
            if result.forces.shape != geometry.positions.shape:
                raise ValueError("ABACUS force count differs from the input atoms")
            if self._settings["execution_mode"] == "fileio":
                self._finish_fileio(deadline)
            self._statistics["successful_evaluations"] += 1
            return result
        except Exception as exc:
            self._statistics["failed_evaluations"] += 1
            self._failure = f"ABACUS evaluation failed: {type(exc).__name__}: {exc}"
            self._diagnostic(
                "failure.json", {"error": self._failure, "request": self._last_request}
            )
            self._stop_process()
            raise EvaluationError(self._failure) from exc
        finally:
            self._statistics["wall_seconds"] += time.monotonic() - started

    def close(self):
        """Close once, killing the owned group if official socket shutdown stalls."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None and self._failure is None:
                deadline = time.monotonic() + min(2.0, self._settings["timeout_seconds"])
                response = self._exchange({"op": "close"}, deadline)
                if response.get("status") != "closed":
                    raise ValueError("ABACUS worker did not confirm closure")
                self._wait_without_reaping(self._process, deadline)
        except Exception as exc:
            self._diagnostic("cleanup.json", {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._stop_process()
