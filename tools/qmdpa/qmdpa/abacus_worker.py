# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Private JSON-lines worker around official ABACUS socket and FileIO calculators.

The supervising process owns the deadline and process group. In particular,
third-party socket accept and close operations cannot block the parent. This
module is an implementation detail, not an independent simulation entry point.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import sys
import threading
import traceback
from pathlib import Path

from .abacus import _MAX_MESSAGE, _digest, map_abacus_coordinates, validate_abacus_settings


def _calculator_arguments(request):
    """Freeze native inputs and share the same physical settings in both modes."""
    from abacuslite import AbacusProfile

    directory = Path(request["directory"])
    settings = validate_abacus_settings(request["settings"], directory)
    pp_directory = directory / "pseudopotentials"
    pp_directory.mkdir()
    potentials = {}
    for symbol, item in settings["pseudopotentials"].items():
        target = pp_directory / f"{symbol}.upf"
        shutil.copyfile(item["path"], target)
        if _digest(target) != item["sha256"]:
            raise ValueError("Pseudopotential changed while preparing the worker")
        potentials[symbol] = target.name
    # Native ABACUS resolves its own libraries, independently of LAMMPS/Torch.
    command = shlex.join(
        [
            "env",
            "-u",
            "LD_PRELOAD",
            "-u",
            "LD_LIBRARY_PATH",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            settings["executable"],
        ]
    )
    return settings, {
        "profile": AbacusProfile(command=command, pseudo_dir=str(pp_directory)),
        "directory": directory / "calculation",
        "pseudopotentials": potentials,
        "kpts": {
            "mode": "mp-sampling",
            "gamma-centered": True,
            "nk": [1, 1, 1],
            "kshift": [0, 0, 0],
        },
        "inp": {
            "calculation": "scf",
            "basis_type": "pw",
            "dft_functional": "PBE",
            "ecutwfc": settings["ecutwfc"],
            "scf_thr": settings["scf_thr"],
            "scf_nmax": settings["scf_nmax"],
            "ks_solver": settings["ks_solver"],
            "nspin": 1,
            "smearing_method": "fixed",
            "symmetry": -1,
            "cal_force": 1,
            "cal_stress": 0,
        },
    }


def _geometry_and_atoms(request, settings):
    from ase import Atoms

    from .core import Geometry

    data = request["geometry"]
    geometry = Geometry(tuple(data["atom_ids"]), tuple(data["symbols"]),
                        data["positions_angstrom"], data.get("cell_angstrom"))
    if any(symbol not in settings["pseudopotentials"] for symbol in geometry.symbols):
        raise ValueError("Missing element pseudopotential")
    atoms = Atoms(
        symbols=geometry.symbols,
        positions=map_abacus_coordinates(geometry, settings),
        cell=settings["supercell_angstrom"],
        pbc=True,
    )
    return geometry, atoms


def _evaluation_response(geometry, calculator):
    from .core import Evaluation

    # Official ASE supplies eV and eV/Angstrom in the original atom order.
    result = Evaluation(calculator.results["energy"], calculator.results["forces"])
    if result.forces.shape != geometry.positions.shape:
        raise ValueError("ABACUS force count differs from the requested geometry")
    return {
        "status": "ok",
        "atom_ids": list(geometry.atom_ids),
        "symbols": list(geometry.symbols),
        "scf_converged": True,
        "energy_ev": result.energy,
        "forces_ev_per_angstrom": result.forces.tolist(),
    }


def _interface(calculator_type):
    source = Path(inspect.getfile(calculator_type)).resolve()
    return {"path": str(source), "sha256": _digest(source)}


class _Session:
    """Default persistent native socket session."""

    def __init__(self, request):
        from abacuslite import AbacusSocketIO

        name = request["unixsocket"]
        if not isinstance(name, str) or re.fullmatch(r"qmdpa-[0-9a-f]{32}", name) is None:
            raise ValueError("Invalid private ABACUS socket name")
        self.settings, arguments = _calculator_arguments(request)
        self.log = (Path(request["directory"]) / "socket.log").open("w")
        self.calculator = AbacusSocketIO(
            **arguments, unixsocket=name, timeout=self.settings["timeout_seconds"], log=self.log
        )
        self.identity = None
        self.interface = _interface(AbacusSocketIO)

    def evaluate(self, request):
        from ase.calculators.calculator import all_changes

        geometry, atoms = _geometry_and_atoms(request, self.settings)
        identity = (geometry.atom_ids, geometry.symbols)
        if self.identity is not None and identity != self.identity:
            raise ValueError("ABACUS atom IDs and symbols changed during the session")
        changes = all_changes if self.identity is None else ["positions"]
        self.identity = identity
        self.calculator.calculate(
            atoms=atoms, properties=["energy", "forces"], system_changes=changes
        )
        if self.calculator.last_scf_converged is not True:
            raise RuntimeError("ABACUS SCF did not report convergence for this frame")
        return _evaluation_response(geometry, self.calculator)

    def close(self):
        try:
            self.calculator.close()
        finally:
            self.log.close()


class _FileIOSession:
    """Exactly one cold native calculation in a fresh isolated worker."""

    def __init__(self, request):
        from abacuslite import Abacus

        if request.get("unixsocket") is not None:
            raise ValueError("ABACUS FileIO must not create a socket")
        self.settings, arguments = _calculator_arguments(request)
        self.directory = Path(request["directory"])
        self.calculation = arguments["directory"]
        self.calculation.mkdir(exist_ok=False)
        self.calculator = Abacus(**arguments)
        self.interface = _interface(Abacus)
        self.evaluated = False

    def evaluate(self, request):
        from ase.calculators.calculator import all_changes

        if self.evaluated:
            raise RuntimeError("A FileIO worker accepts exactly one evaluation")
        self.evaluated = True
        geometry, atoms = _geometry_and_atoms(request, self.settings)
        self.calculator.calculate(
            atoms=atoms, properties=["energy", "forces"], system_changes=all_changes
        )
        logs = list(self.calculation.glob("OUT.*/running_scf.log"))
        if len(logs) != 1:
            raise RuntimeError("Expected exactly one fresh ABACUS FileIO SCF log")
        log = logs[0].read_text()
        if log.count("#SCF IS CONVERGED#") != 1 or "!!SCF IS NOT CONVERGED!!" in log:
            raise RuntimeError("ABACUS SCF did not report convergence for this frame")
        residuals = re.findall(r"Electron density deviation\s*=\s*([-+\d.eE]+)", log)
        iterations = re.findall(r"#ELEC ITER#\s*(\d+)", log)
        if not residuals or not iterations:
            raise RuntimeError("ABACUS FileIO log omitted SCF residual or iteration evidence")
        residual, iteration = float(residuals[-1]), int(iterations[-1])
        if (
            not 0 <= residual <= self.settings["scf_thr"]
            or not 0 < iteration <= self.settings["scf_nmax"]
        ):
            raise RuntimeError("ABACUS FileIO SCF evidence exceeds the declared limits")
        response = _evaluation_response(geometry, self.calculator)
        evidence = {
            "positions_angstrom": geometry.positions.tolist(),
            "native_positions_angstrom": atoms.positions.tolist(),
            "scf_log": str(logs[0].relative_to(self.directory)),
            "scf_log_sha256": _digest(logs[0]),
            "terminal_density_deviation": residual,
            "scf_iterations": iteration,
            "input_sha256": {
                name: _digest(self.calculation / name) for name in ("INPUT", "STRU", "KPT")
            },
        }
        (self.directory / "result.json").write_text(
            json.dumps({**response, "evidence": evidence}, indent=2, allow_nan=False) + "\n"
        )
        return response

    def close(self):
        # ASE FileIO waits for its native subprocess. The parent supervises
        # the whole process group when calculation or shutdown raises/blocks.
        pass


def _start_owner_watchdog(owner_pid, unixsocket):
    """Bound orphan lifetime even while third-party accept/SCF/close blocks."""
    pid = os.getpid()
    if owner_pid <= 0 or owner_pid == pid:
        raise ValueError("ABACUS worker requires its original parent PID")
    if unixsocket is not None and re.fullmatch(r"qmdpa-[0-9a-f]{32}", unixsocket) is None:
        raise ValueError("Invalid private ABACUS socket name")
    if os.getpgrp() != pid or os.getsid(0) != pid:
        raise RuntimeError("ABACUS worker must own a separate process group and session")
    socket_path = Path("/tmp/ipi_" + unixsocket) if unixsocket is not None else None

    pause = threading.Event()

    def watch():
        while True:
            # Reparenting detects death even if the old PID has been reused.
            if os.getppid() != owner_pid:
                try:
                    if socket_path is not None:
                        socket_path.unlink(missing_ok=True)
                except OSError:
                    pass
                # Never signal a group unless this process is still its leader
                # and the leader of the session created by its owner.
                try:
                    if os.getpgrp() == pid and os.getsid(0) == pid:
                        os.killpg(pid, signal.SIGKILL)
                finally:
                    os._exit(1)
            pause.wait(0.2)

    thread = threading.Thread(target=watch, name="abacus-owner-watchdog", daemon=True)
    thread.start()
    return thread


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--owner-pid", required=True, type=int)
    cli.add_argument("--unixsocket")
    args = cli.parse_args()
    _start_owner_watchdog(args.owner_pid, args.unixsocket)
    # Reserve the protocol descriptor before importing third-party libraries.
    # Their Python prints and native stdout go to the retained worker log.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        session = None
        previous_id = 0
        try:
            while True:
                line = sys.stdin.readline(_MAX_MESSAGE + 1)
                if not line:
                    break
                request = None
                try:
                    if len(line) > _MAX_MESSAGE or not line.endswith("\n"):
                        raise ValueError("ABACUS worker request is incomplete or too large")
                    request = json.loads(line)
                    if not isinstance(request, dict) or type(request.get("id")) is not int:
                        raise ValueError("ABACUS worker requires an integer request ID")
                    request_id = request["id"]
                    if request_id != previous_id + 1:
                        raise ValueError("ABACUS worker request IDs must increase by one")
                    previous_id = request_id
                    operation = request["op"]
                    if operation == "initialize" and session is None:
                        if request.get("unixsocket") != args.unixsocket:
                            raise ValueError("ABACUS initialization changed the owned socket")
                        mode = request["settings"].get("execution_mode", "socket")
                        if mode == "socket":
                            session = _Session(request)
                        elif mode == "fileio":
                            session = _FileIOSession(request)
                        else:
                            raise ValueError("Unknown ABACUS execution mode")
                        response = {"status": "ready", "interface": session.interface}
                        if mode == "fileio":
                            response["execution_mode"] = mode
                    elif operation == "evaluate" and session is not None:
                        response = session.evaluate(request)
                    elif operation == "close" and session is not None:
                        session.close()
                        session = None
                        protocol.write(json.dumps({"id": request_id, "status": "closed"}) + "\n")
                        return 0
                    else:
                        raise ValueError("Invalid ABACUS worker operation or session state")
                    protocol.write(
                        json.dumps({"id": request_id, **response}, allow_nan=False) + "\n"
                    )
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    request_id = request.get("id") if isinstance(request, dict) else None
                    protocol.write(
                        json.dumps(
                            {
                                "id": request_id,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        + "\n"
                    )
                    return 1
        finally:
            if session is not None:
                session.close()  # Supervision bounds even a third-party close hang.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
