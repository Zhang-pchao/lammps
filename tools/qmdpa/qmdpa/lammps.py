# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Collective LAMMPS external-force coupling with explicit failure handling."""

import numpy as np

from .core import EvaluationError
from .region import box_matrix, unwrap


class _SerialComm:
    rank = 0
    size = 1

    def gather(self, value, root=0):
        return [value]

    def bcast(self, value, root=0):
        return value


class ExternalCoupling:
    """Add only QM-minus-DP corrections through ``fix external``.

    Call :meth:`run` instead of calling LAMMPS ``run`` directly. A callback
    exception cannot cross ctypes safely. Errors are broadcast, stop the run,
    and are re-raised by this wrapper. The in-memory failed step is invalid:
    discard the LAMMPS instance and retain only the last accepted checkpoint.
    Do not configure LAMMPS dumps or automatic restart writes during this run;
    write accepted frames/checkpoints from the controlling driver afterward.

    ``type_symbols`` maps LAMMPS type 1, type 2, ... to element names and is
    checked for every owned QM atom. When omitted, the caller is responsible
    for verifying that the region symbols agree with the LAMMPS type mapping.
    """

    def __init__(self, lmp, region, calculator, *, fix_id="qmdpa", comm=None, type_symbols=None):
        if not fix_id.isidentifier():
            raise ValueError("fix_id must be an identifier")
        if lmp.extract_global("units") != "metal":
            raise ValueError("QM/DPA coupling requires LAMMPS units metal")
        self.lmp = lmp
        self.region = region
        self.calculator = calculator
        self.fix_id = fix_id
        self.comm = comm if comm is not None else lmp.get_mpi_comm()
        world_size = lmp.extract_setting("world_size")
        if self.comm is None:
            if world_size != 1:
                raise ValueError("Parallel QM/DPA requires a matching mpi4py communicator")
            self.comm = _SerialComm()
        if self.comm.size != world_size:
            raise ValueError("The communicator size does not match the LAMMPS instance")
        if self.comm.rank == 0 and calculator is None:
            raise ValueError("rank zero requires a subtractive calculator")
        self.type_symbols = None if type_symbols is None else tuple(type_symbols)
        if self.type_symbols is not None and (
            not self.type_symbols or any(not isinstance(s, str) or not s for s in self.type_symbols)
        ):
            raise ValueError("type_symbols must contain a nonempty element name per LAMMPS type")
        self.failure = None
        self.last_step = None
        self.last_correction = None
        self._box = lmp.extract_box()
        self._cell = box_matrix(self._box)
        self._members = set(region.atom_ids)
        self._symbols = dict(zip(region.atom_ids, region.symbols))
        lmp.command(f"fix {fix_id} all external pf/callback 1 1")
        lmp.command(f"fix_modify {fix_id} energy yes virial no")
        lmp.set_fix_external_callback(fix_id, self._callback)

    def _local_packet(self, nlocal, tags, positions):
        box = self.lmp.extract_box()
        if (
            not np.array_equal(box_matrix(box), self._cell)
            or box[:2] != self._box[:2]
            or box[5] != self._box[5]
        ):
            raise ValueError("QM/DPA currently requires a fixed simulation box")
        if nlocal == 0:
            # LAMMPS may pass NULL arrays on an empty rank.
            return [], np.empty((0, 3))
        ids = np.asarray(tags).reshape(-1)
        xyz = np.asarray(positions)
        if ids.shape != (nlocal,) or ids.dtype.kind not in "iu":
            raise ValueError("Local atom IDs must be an integer array matching nlocal")
        if xyz.shape != (nlocal, 3):
            raise ValueError("Local coordinates must have shape (nlocal, 3)")
        selected = [row for row, atom_id in enumerate(ids) if atom_id in self._members]
        selected_ids = [int(ids[row]) for row in selected]
        if not selected:
            return [], np.empty((0, 3))
        if self.type_symbols is not None:
            atom_types = np.asarray(self.lmp.numpy.extract_atom("type"))
            if (
                atom_types.ndim != 1
                or len(atom_types) < nlocal
                or atom_types.dtype.kind not in "iu"
            ):
                raise ValueError("LAMMPS atom types must be an integer array covering local atoms")
            for row, atom_id in zip(selected, selected_ids):
                type_id = int(atom_types[row])
                if not 1 <= type_id <= len(self.type_symbols):
                    raise ValueError(f"LAMMPS atom type {type_id} has no element mapping")
                if self.type_symbols[type_id - 1] != self._symbols[atom_id]:
                    raise ValueError(f"Element mapping differs for QM atom ID {atom_id}")
        image_flags = self.lmp.numpy.extract_atom("image")
        images = np.array(
            [self.lmp.decode_image_flags(int(image_flags[row])) for row in selected],
            dtype=np.int64,
        ).reshape(-1, 3)
        return selected_ids, unwrap(xyz[selected], images, self._cell)

    def _stop(self, step, error, forces):
        """Invalidate the segment and request return from the native run loop."""
        self.failure = f"QM/DPA evaluation failed at step {step}: {error}"
        if forces is not None:
            try:
                forces[:] = 0.0
            except Exception:
                # A damaged/read-only output array is already a failed segment.
                pass
        try:
            self.lmp.fix_external_set_energy_global(self.fix_id, 0.0)
        except Exception as exc:
            self.failure += f"; could not clear correction energy: {exc}"
        self.lmp.force_timeout()

    def _callback(self, caller, step, nlocal, tags, positions, forces):
        # Every rank participates even when local validation or QM fails.
        try:
            if nlocal:
                if forces is None or np.shape(forces) != (nlocal, 3):
                    raise ValueError("Local correction forces must have shape (nlocal, 3)")
                forces[:] = 0.0
            packet = (None, self._local_packet(nlocal, tags, positions))
        except Exception as exc:
            packet = (f"{type(exc).__name__}: {exc}", None)
        packets = self.comm.gather(packet, root=0)
        result = None
        if self.comm.rank == 0:
            try:
                if self.failure is not None:
                    raise EvaluationError(self.failure)
                errors = [error for error, _ in packets if error is not None]
                if errors:
                    raise EvaluationError("; ".join(errors))
                geometry = self.region.assemble([data for _, data in packets])
                correction = self.calculator.correction(geometry)
                result = (None, correction.energy, correction.forces)
            except Exception as exc:
                result = (f"{type(exc).__name__}: {exc}", 0.0, None)
        error, energy, correction_forces = self.comm.bcast(result, root=0)
        if error is not None:
            self._stop(step, error, forces)
            return
        local_error = None
        try:
            if nlocal:
                ids = np.asarray(tags).reshape(-1)
                forces[:] = self.region.local_forces(ids, correction_forces)
            self.lmp.fix_external_set_energy_global(self.fix_id, energy)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        # Writing the correction can fail on only one rank. Publish that failure
        # before any rank leaves the callback and re-enters the native MD loop.
        errors = self.comm.gather(local_error, root=0)
        error = None
        if self.comm.rank == 0:
            error = "; ".join(item for item in errors if item is not None) or None
        error = self.comm.bcast(error, root=0)
        if error is not None:
            self._stop(step, error, forces)
            return
        self.last_step = int(step)
        self.last_correction = float(energy)

    def run(self, steps):
        """Run a bounded segment, accepting its final state only after success."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if self.failure is not None:
            raise EvaluationError(self.failure)
        start = int(self.lmp.extract_global("ntimestep"))
        self.lmp.command(f"run {steps} post no")
        if self.failure is not None:
            raise EvaluationError(self.failure)
        final = int(self.lmp.extract_global("ntimestep"))
        if final != start + steps:
            self.failure = f"LAMMPS stopped at step {final}; requested step {start + steps}"
            raise EvaluationError(self.failure)
        return self.last_correction
