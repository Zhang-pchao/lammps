#!/usr/bin/env python3
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
        except Exception as exc:  # noqa: BLE001 - Preserve collective or top-level failure reporting.
            result = (f"{type(exc).__name__}: {exc}", None)
    if comm is not None:
        result = comm.bcast(result, root=0)
    error, value = result
    if error is not None:
        raise RuntimeError(error)
    return value


class ProviderResources:
    """Owned external providers for one run; no calculation or automatic retry."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._providers = []
        self._closed = False
        self._errors = []

    def register(self, provider):
        if self._closed:
            raise RuntimeError("Cannot add a provider after run cleanup")
        if not callable(getattr(provider, "close_checked", None)):
            raise TypeError("External providers require checked resource cleanup")
        self._providers.append(provider)
        return provider

    def close(self):
        if not self._closed:
            self._closed = True
            for provider in reversed(self._providers):
                try:
                    provider.close_checked()
                except Exception as exc:  # noqa: BLE001 - Preserve collective or top-level failure reporting.
                    self._errors.append(f"{type(exc).__name__}: {exc}")
        return list(self._errors)

    def statistics(self):
        return {"closed": self._closed, "cleanup_errors": list(self._errors),
                "providers": [dict(provider.statistics) for provider in self._providers]}


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


def _canonical_periodic_frame(frame, box):
    """Keep wrapped coordinates and image counts equivalent across restart I/O.

    LAMMPS may leave accepted coordinates outside the box until write_restart
    remaps and migrates atoms. Transform the recorded representation only;
    never mutate native coordinates, forces or velocities after a callback.
    """
    from .region import box_matrix

    if len(box[5]) != 3 or not all(box[5]):
        raise ValueError("Periodic accepted frames require full three-dimensional PBC")
    cell = box_matrix(box)
    xyz = np.array([atom["x"] for atom in frame["atoms"]], dtype=float)
    images = np.array([atom["image"] for atom in frame["atoms"]], dtype=np.int64)
    fractional = np.linalg.solve(cell.T, (xyz - np.asarray(box[0])).T).T
    shifts = np.floor(fractional)
    if not np.isfinite(shifts).all() or np.max(np.abs(shifts)) >= 2**52:
        raise ValueError("Periodic frame image shifts exceed exact integer representation")
    shifts = shifts.astype(np.int64)
    wrapped = xyz - shifts @ cell
    for atom, position, image in zip(frame["atoms"], wrapped, images + shifts):
        atom["x"] = position.tolist()
        atom["image"] = image.tolist()


def _frame(lmp, comm, *, periodic=False):
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
    except Exception as exc:  # noqa: BLE001 - Preserve collective or top-level failure reporting.
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
        if periodic:
            _canonical_periodic_frame(frame, lmp.extract_box())
            _validate_frame(frame)

    _root_call(comm, assemble)
    return frame


def _validate_restart(checkpoint_path, config):
    checkpoint = json.loads(Path(checkpoint_path).read_text())
    if checkpoint["physics_sha256"] != config["physics_sha256"]:
        raise ValueError("Restart physics configuration differs from this run")
    from .state_records import validate_frame_state

    validate_frame_state(checkpoint["frame"], config)
    from .bias import validate_bias_frame

    validate_bias_frame(checkpoint["frame"], config)
    from .dynamics import validate_thermostat_frame
    from .sampling import validate_sampling_frame

    validate_thermostat_frame(checkpoint["frame"], config)
    validate_sampling_frame(checkpoint["frame"], config)
    from .ehrenfest_runtime import validate_frame as validate_ehrenfest_frame

    validate_ehrenfest_frame(checkpoint["frame"], config)
    from .water_buffer import validate_buffer_frame

    validate_buffer_frame(checkpoint["frame"], config)
    from .reactive_config import validate_reactive_frame

    validate_reactive_frame(checkpoint["frame"], config)
    from .sampling import validate_checkpoint_history

    validate_checkpoint_history(checkpoint, checkpoint_path, config)
    restart = Path(checkpoint_path).resolve().parent / checkpoint["restart_file"]
    if file_digest(restart) != checkpoint["restart_sha256"]:
        raise ValueError("Restart binary checksum mismatch")
    checkpoint["restart_path"] = str(restart)
    return checkpoint


def preflight_restart(config_path, checkpoint_path, *, config_loader, comm=None, output=None):
    """Reject invalid restart inputs without initializing a native engine.

    The native driver repeats validation before reading the restart, so a
    successful preflight cannot bypass changed-input or recomputed-state checks.
    Supply the launch communicator when called collectively.
    """
    if checkpoint_path is None:
        return None
    try:
        return _root_call(
            comm, lambda: _validate_restart(checkpoint_path, config_loader(config_path))
        )
    except Exception as exc:
        failure_message = f"{type(exc).__name__}: {exc}"
        if output is not None:
            def save_failure():
                directory = Path(output).resolve()
                directory.mkdir(parents=True, exist_ok=False)
                (directory / "failure.json").write_text(json.dumps({
                    "status": "failed", "stage": "restart-preflight",
                    "error": failure_message, "native_initialized": False,
                    "recovery": "Use the last accepted checkpoint; no state was advanced.",
                }, indent=2) + "\n")
            try:
                _root_call(comm, save_failure)
            except RuntimeError:
                pass  # Never replace old output or hide the original input error.
        raise


def _compare_restart(frame, checkpoint, config=None):
    reference = checkpoint["frame"]
    from .water_buffer import validate_buffer_frame

    if config is not None:
        validate_buffer_frame(frame, config)
        validate_buffer_frame(reference, config)
        from .reactive_config import validate_reactive_frame

        validate_reactive_frame(frame, config)
        validate_reactive_frame(reference, config)
    from .bias import compare_bias_restart

    compare_bias_restart(frame, reference, {} if config is None else config)
    from .dynamics import compare_thermostat_restart
    from .sampling import compare_sampling_restart

    compare_thermostat_restart(frame, reference, {} if config is None else config)
    compare_sampling_restart(frame, reference, {} if config is None else config)
    from .ehrenfest_runtime import compare_restart as compare_ehrenfest_restart

    compare_ehrenfest_restart(frame, reference, {} if config is None else config)
    _validate_frame(frame)
    _validate_frame(reference)
    from .state_records import validate_frame_state

    for item in (frame, reference):
        validate_frame_state(item, {} if config is None else config)
    actual_state, expected_state = frame.get("electronic_state"), reference.get("electronic_state")
    if (actual_state is None) != (expected_state is None):
        raise ValueError("Restart electronic-state record is missing")
    if actual_state is not None and any(
        actual_state[name] != expected_state[name] for name in ("reference_sha256", "selected_root")
    ):
        raise ValueError("Restart recomputation selected a different electronic state")
    for role, actual in frame.get("auxiliary_electronic_states", {}).items():
        expected = reference["auxiliary_electronic_states"][role]
        if any(actual[name] != expected[name] for name in ("reference_sha256", "selected_root")):
            raise ValueError("Restart recomputation selected a different auxiliary electronic state")
        if abs(actual["provider_energy_ev"] - expected["provider_energy_ev"]) > 1e-6:
            raise ValueError("Restart auxiliary provider energy differs by more than 1e-6 eV")
    if frame["step"] != reference["step"]:
        raise ValueError("Restart timestep mismatch")
    if [atom["id"] for atom in frame["atoms"]] != [atom["id"] for atom in reference["atoms"]]:
        raise ValueError("Restart atom identity mismatch")
    from .execution import restart_limits

    energy_limit, force_limit, energy_label = restart_limits({} if config is None else config)
    if not np.isclose(frame["potential_eV"], reference["potential_eV"], atol=energy_limit, rtol=0):
        raise ValueError(f"Recomputed restart potential differs by more than {energy_label} eV")
    for name, tolerance in (("x", 1e-12), ("v", 1e-12), ("f", force_limit), ("image", 0)):
        actual = [atom[name] for atom in frame["atoms"]]
        expected = [atom[name] for atom in reference["atoms"]]
        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            raise ValueError(f"Restart {name} differs beyond tolerance {tolerance}")


def _attach_state(frame, coupling, config):
    """Validate the same successful callback before any accepted-frame write."""
    from .state_records import configured_reference, validate_frame_state
    from .water_buffer import HAMILTONIAN as WATER_BUFFER
    from .water_buffer import frame_record

    if config.get("hamiltonian") == WATER_BUFFER:
        if coupling.failure is not None or coupling.last_step != frame["step"]:
            raise ValueError("Water-buffer frame requires the successful current callback")
        frame["water_buffer"] = frame_record(frame, config)
    from .reactive_config import frame_record as reactive_frame_record
    from .reactive_water import HAMILTONIAN as REACTIVE_WATER

    if config.get("hamiltonian") == REACTIVE_WATER:
        if coupling.failure is not None or coupling.last_step != frame["step"]:
            raise ValueError("Reactive-water frame requires the successful current callback")
        frame["reactive_water"] = reactive_frame_record(frame, config)
    from .ehrenfest_runtime import attach_frame as attach_ehrenfest_frame

    attach_ehrenfest_frame(frame, coupling, config)
    if configured_reference(config) is None:
        return
    if getattr(getattr(coupling, "comm", None), "rank", 0) != 0:
        raise ValueError("Electronic-state frame assembly belongs only to rank zero")
    if coupling.failure is not None or coupling.last_step != frame["step"]:
        raise ValueError("Electronic-state frame requires the successful current callback")
    observer = coupling.state_observer
    if observer is None:
        raise ValueError("RHF/TDA coupling did not retain its electronic-state provider")
    records = {"electronic_state": observer.accepted_record(frame)}
    auxiliary = getattr(observer, "accepted_auxiliary_records", None)
    if auxiliary is not None:
        records["auxiliary_electronic_states"] = auxiliary(frame)
    try:
        validate_frame_state({**frame, **records}, config)
    except BaseException:
        observer.clear()
        raise
    frame.update(records)
