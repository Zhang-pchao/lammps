# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Collective I/O, accepted-state snapshots, and restart verification."""

import json
from pathlib import Path

import numpy as np

from .config import _positive_integer, file_digest


def _root_call(comm, function):
    """Broadcast root I/O/backend errors before another collective is entered."""
    result = None
    if comm is None or comm.rank == 0:
        try:
            result = (None, function())
        except Exception as exc:
            result = (f"{type(exc).__name__}: {exc}", None)
    if comm is not None:
        result = comm.bcast(result, root=0)
    error, value = result
    if error is not None:
        raise RuntimeError(error)
    return value


def _quoted(path):
    value = str(path)
    if any(character in value for character in ('"', "$", "\n", "\r")):
        raise ValueError("Paths must not contain quotes, dollar signs, or line breaks")
    return f'"{value}"'


def _validate_frame(frame):
    _positive_integer(frame["step"], "frame.step", allow_zero=True)
    for name in ("potential_eV", "kinetic_eV", "temperature_K"):
        if name in frame:
            value = frame[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"Frame {name} must be a finite scalar")
    atoms = frame["atoms"]
    ids = []
    for atom in atoms:
        ids.append(_positive_integer(atom["id"], "frame atom ID"))
        for name in ("x", "v", "f", "image"):
            if name == "image" and any(type(value) is not int for value in atom[name]):
                raise ValueError("Frame image flags must be integers, not booleans")
            values = np.asarray(atom[name])
            if (
                values.shape != (3,)
                or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Frame {name} must be a finite three-vector")
            if name == "image" and values.dtype.kind not in "iu":
                raise ValueError("Frame image flags must be integers")
    if not ids or ids != sorted(set(ids)):
        raise ValueError("Frame atom IDs must be unique and sorted")


def _frame(lmp, comm):
    try:
        count = lmp.extract_setting("nlocal")
        rows = []
        if count:
            arrays = {
                name: lmp.numpy.extract_atom(name)[:count].copy()
                for name in ("id", "x", "v", "f", "image")
            }
            rows = [
                {
                    "id": int(arrays["id"][i]),
                    "x": arrays["x"][i].tolist(),
                    "v": arrays["v"][i].tolist(),
                    "f": arrays["f"][i].tolist(),
                    "image": list(lmp.decode_image_flags(int(arrays["image"][i]))),
                }
                for i in range(count)
            ]
        packet = (None, rows)
    except Exception as exc:
        packet = (f"{type(exc).__name__}: {exc}", None)
    packets = [packet] if comm is None else comm.gather(packet, root=0)
    frame = None

    def assemble():
        nonlocal frame
        errors = [error for error, _ in packets if error is not None]
        if errors:
            raise ValueError("; ".join(errors))
        atoms = sorted((atom for _, rows in packets for atom in rows), key=lambda atom: atom["id"])
        if len(atoms) != lmp.get_natoms():
            raise ValueError("Accepted frame has missing atoms")
        frame = {
            "step": int(lmp.extract_global("ntimestep")),
            "potential_eV": lmp.get_thermo("pe"),
            "kinetic_eV": lmp.get_thermo("ke"),
            "temperature_K": lmp.get_thermo("temp"),
            "atoms": atoms,
        }
        _validate_frame(frame)

    _root_call(comm, assemble)
    return frame


def _validate_restart(checkpoint_path, config):
    checkpoint = json.loads(Path(checkpoint_path).read_text())
    if checkpoint["physics_sha256"] != config["physics_sha256"]:
        raise ValueError("Restart physics configuration differs from this run")
    restart = Path(checkpoint_path).resolve().parent / checkpoint["restart_file"]
    if file_digest(restart) != checkpoint["restart_sha256"]:
        raise ValueError("Restart binary checksum mismatch")
    checkpoint["restart_path"] = str(restart)
    return checkpoint


def _compare_restart(frame, checkpoint):
    reference = checkpoint["frame"]
    _validate_frame(frame)
    _validate_frame(reference)
    if frame["step"] != reference["step"]:
        raise ValueError("Restart timestep mismatch")
    if [atom["id"] for atom in frame["atoms"]] != [atom["id"] for atom in reference["atoms"]]:
        raise ValueError("Restart atom identity mismatch")
    if not np.isclose(frame["potential_eV"], reference["potential_eV"], atol=1e-6, rtol=0):
        raise ValueError("Recomputed restart potential differs by more than 1e-6 eV")
    for name, tolerance in (("x", 1e-12), ("v", 1e-12), ("f", 1e-5), ("image", 0)):
        actual = [atom[name] for atom in frame["atoms"]]
        expected = [atom[name] for atom in reference["atoms"]]
        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            raise ValueError(f"Restart {name} differs beyond tolerance {tolerance}")
