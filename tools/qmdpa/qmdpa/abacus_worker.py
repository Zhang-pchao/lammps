# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Private JSON-lines worker around official abacuslite.AbacusSocketIO.

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


class _Session:
    def __init__(self, request):
        from abacuslite import AbacusProfile, AbacusSocketIO

        directory = Path(request["directory"])
        self.settings = validate_abacus_settings(request["settings"], directory)
        name = request["unixsocket"]
        if re.fullmatch(r"qmdpa-[0-9a-f]{32}", name) is None:
            raise ValueError("Invalid private ABACUS socket name")
        # Freeze the bytes actually passed to ABACUS, without changing originals.
        pp_directory = directory / "pseudopotentials"
        pp_directory.mkdir()
        potentials = {}
        for symbol, item in self.settings["pseudopotentials"].items():
            target = pp_directory / f"{symbol}.upf"
            shutil.copyfile(item["path"], target)
            if _digest(target) != item["sha256"]:
                raise ValueError("Pseudopotential changed while preparing the worker")
            potentials[symbol] = target.name
        # The native binary has its own validated runtime search path. Do not
        # inject the parent LAMMPS/Torch compiler libraries into ABACUS.
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
                self.settings["executable"],
            ]
        )
        profile = AbacusProfile(command=command, pseudo_dir=str(pp_directory))
        self.log = (directory / "socket.log").open("w")
        self.calculator = AbacusSocketIO(
            profile=profile,
            directory=directory / "calculation",
            unixsocket=name,
            timeout=self.settings["timeout_seconds"],
            log=self.log,
            pseudopotentials=potentials,
            kpts={
                "mode": "mp-sampling",
                "gamma-centered": True,
                "nk": [1, 1, 1],
                "kshift": [0, 0, 0],
            },
            inp={
                "calculation": "scf",
                "basis_type": "pw",
                "dft_functional": "PBE",
                "ecutwfc": self.settings["ecutwfc"],
                "scf_thr": self.settings["scf_thr"],
                "scf_nmax": self.settings["scf_nmax"],
                "ks_solver": self.settings["ks_solver"],
                "nspin": 1,
                "smearing_method": "fixed",
                "symmetry": -1,
                "cal_force": 1,
                "cal_stress": 0,
            },
        )
        self.identity = None
        source = Path(inspect.getfile(AbacusSocketIO)).resolve()
        self.interface = {"path": str(source), "sha256": _digest(source)}

    def evaluate(self, request):
        from ase import Atoms
        from ase.calculators.calculator import all_changes

        from .core import Evaluation, Geometry

        data = request["geometry"]
        geometry = Geometry(
            tuple(data["atom_ids"]), tuple(data["symbols"]), data["positions_angstrom"]
        )
        identity = (geometry.atom_ids, geometry.symbols)
        if self.identity is not None and identity != self.identity:
            raise ValueError("ABACUS atom IDs and symbols changed during the session")
        if any(symbol not in self.settings["pseudopotentials"] for symbol in geometry.symbols):
            raise ValueError("Missing element pseudopotential")
        changes = all_changes if self.identity is None else ["positions"]
        self.identity = identity
        positions = map_abacus_coordinates(geometry, self.settings)
        atoms = Atoms(
            symbols=geometry.symbols,
            positions=positions,
            cell=self.settings["supercell_angstrom"],
            pbc=True,
        )
        self.calculator.calculate(
            atoms=atoms, properties=["energy", "forces"], system_changes=changes
        )
        if self.calculator.last_scf_converged is not True:
            raise RuntimeError("ABACUS SCF did not report convergence for this frame")
        # Official ASE socket conversion supplies eV and eV/Angstrom and maps
        # grouped species back to the original input order. Do not convert twice.
        result = Evaluation(self.calculator.results["energy"], self.calculator.results["forces"])
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

    def close(self):
        try:
            self.calculator.close()
        finally:
            self.log.close()


def _start_owner_watchdog(owner_pid, unixsocket):
    """Bound orphan lifetime even while third-party accept/SCF/close blocks."""
    pid = os.getpid()
    if owner_pid <= 0 or owner_pid == pid:
        raise ValueError("ABACUS worker requires its original parent PID")
    if re.fullmatch(r"qmdpa-[0-9a-f]{32}", unixsocket) is None:
        raise ValueError("Invalid private ABACUS socket name")
    if os.getpgrp() != pid or os.getsid(0) != pid:
        raise RuntimeError("ABACUS worker must own a separate process group and session")
    socket_path = Path("/tmp/ipi_" + unixsocket)

    pause = threading.Event()

    def watch():
        while True:
            # Reparenting detects death even if the old PID has been reused.
            if os.getppid() != owner_pid:
                try:
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
    cli.add_argument("--unixsocket", required=True)
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
                        session = _Session(request)
                        response = {"status": "ready", "interface": session.interface}
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
