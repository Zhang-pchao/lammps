#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Official PLUMED EXTRACV forward/reverse bridge for named static restraints.

No electronic provider, native fix, second mixer or OPES implementation is
created here. The calling coupler owns total-force assembly and accepted steps.
"""

import ctypes as ct
import json
import math
from pathlib import Path

import numpy as np

from .config import file_digest, software_identity
from .named_cv import digest


class _Handle(ct.Structure):
    _fields_ = [("p", ct.c_void_p)]


_Callback = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int, ct.c_char_p, ct.c_void_p)


class _ErrorHandler(ct.Structure):
    _fields_ = [("ptr", ct.c_void_p), ("handler", _Callback)]


class _Plumed:
    """Use the documented nonthrowing C wrapper; C++ exceptions never cross ctypes."""

    def __init__(self, kernel, wrapper):
        self.lib = ct.CDLL(str(wrapper))
        self.lib.plumed_create_dlopen.argtypes = [ct.c_char_p]
        self.lib.plumed_create_dlopen.restype = _Handle
        self.lib.plumed_valid.argtypes = [_Handle]
        self.lib.plumed_valid.restype = ct.c_int
        self.lib.plumed_cmd_nothrow.argtypes = [_Handle, ct.c_char_p, ct.c_void_p, _ErrorHandler]
        self.lib.plumed_cmd_nothrow.restype = None
        self.lib.plumed_finalize.argtypes = [_Handle]
        self.lib.plumed_finalize.restype = None
        self.handle = self.lib.plumed_create_dlopen(str(kernel).encode())
        if not self.lib.plumed_valid(self.handle):
            self.close()
            raise ValueError("PLUMED could not load the declared kernel")

    def cmd(self, key, value=None):
        errors = []
        callback = _Callback(lambda ptr, code, message, opt: errors.append(
            f"PLUMED error {code}: {message.decode(errors='replace')}"))
        if isinstance(value, str):
            buffer = ct.create_string_buffer(value.encode())
            pointer = ct.cast(buffer, ct.c_void_p)
        elif isinstance(value, np.ndarray):
            if not value.flags.c_contiguous or value.dtype != np.float64:
                raise ValueError("PLUMED buffers must be contiguous float64 arrays")
            pointer = value.ctypes.data_as(ct.c_void_p)
        else:
            pointer = None if value is None else ct.cast(ct.byref(value), ct.c_void_p)
        self.lib.plumed_cmd_nothrow(self.handle, key.encode(), pointer, _ErrorHandler(None, callback))
        if errors:
            raise ValueError("; ".join(errors))

    def close(self):
        if self.handle is not None:
            self.lib.plumed_finalize(self.handle)
            self.handle = None


class ExternalCVBias:
    """Versioned named scalar harmonic bias through a pinned official kernel.

    Parameters map each name to center, kappa and unit; kappa is eV/unit**2.
    This first bridge has static history. Existing water OPES retains its native
    history owner; external electronic OPES is not admitted by this class.
    """

    def __init__(self, schema, parameters, kernel, wrapper, directory, *, timestep_fs,
                 restart=None):
        names = {s.name for s in schema.variables}
        if not isinstance(parameters, dict) or set(parameters) != names:
            raise ValueError("Restraint parameters must bind every CV by name")
        params = {}
        for spec in schema.variables:
            row = parameters[spec.name]
            if (not isinstance(row, dict) or set(row) != {"center", "kappa", "unit"}
                    or row["unit"] != spec.unit
                    or any(type(row[k]) not in (int, float) or not math.isfinite(row[k])
                           for k in ("center", "kappa")) or row["kappa"] <= 0):
                raise ValueError("Each restraint requires center, positive kappa (eV/unit**2), and the CV unit")
            params[spec.name] = dict(row)
        if type(timestep_fs) not in (int, float) or not math.isfinite(timestep_fs) or timestep_fs <= 0:
            raise ValueError("External CV timestep must be finite and positive")
        paths = {}
        for name, raw in (("kernel", kernel), ("wrapper", wrapper)):
            if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
                raise ValueError("PLUMED kernel and wrapper require explicit path and SHA256")
            path = Path(raw["path"]).resolve()
            if not path.is_file() or file_digest(path) != raw["sha256"]:
                raise ValueError("PLUMED library checksum differs")
            paths[name] = path
        self.schema, self.parameters = schema, params
        self.identity = {"schema": schema.record(), "parameters": params, "software": software_identity(),
                         "kernel_sha256": kernel["sha256"], "wrapper_sha256": wrapper["sha256"],
                         "timestep_fs": timestep_fs, "history_kind": "static-restraint-v1"}
        self.identity_sha256 = digest(self.identity)
        self.last_step, self.last_frame_sha256, self.accepted_count = -1, None, 0
        if restart is not None:
            if (not isinstance(restart, dict) or set(restart) != {
                    "identity_sha256", "identity", "step", "frame_sha256", "accepted_count", "sha256"}
                    or digest({k: v for k, v in restart.items() if k != "sha256"}) != restart["sha256"]
                    or restart["identity"] != self.identity
                    or restart["identity_sha256"] != self.identity_sha256
                    or type(restart["step"]) is not int or restart["step"] < 0
                    or type(restart["accepted_count"]) is not int
                    or restart["accepted_count"] != restart["step"] + 1
                    or not isinstance(restart["frame_sha256"], str) or len(restart["frame_sha256"]) != 64):
                raise ValueError("External CV restart schema, dependencies, history or kernel identity differs")
            self.last_step, self.last_frame_sha256 = restart["step"], restart["frame_sha256"]
            self.accepted_count = restart["accepted_count"]
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.kernel = _Plumed(paths["kernel"], paths["wrapper"])
        try:
            self.kernel.cmd("setRealPrecision", ct.c_int(8))
            self.kernel.cmd("setMDEnergyUnits", ct.c_double(96.48533212331002))
            self.kernel.cmd("setMDLengthUnits", ct.c_double(0.1))
            self.kernel.cmd("setMDTimeUnits", ct.c_double(1.0))
            self.kernel.cmd("setMDEngine", "qmdpa-external-cv")
            self.kernel.cmd("setNatoms", ct.c_int(0))
            self.kernel.cmd("setTimestep", ct.c_double(timestep_fs / 1000))
            self.kernel.cmd("setLogFile", str(self.directory / "plumed.log"))
            self.kernel.cmd("init")
            # Match the explicit MD conversion exactly. Named eV in older PLUMED
            # releases uses a different historical conversion constant.
            rows = ["UNITS LENGTH=0.1 ENERGY=96.48533212331002 TIME=1"]
            for index, spec in enumerate(schema.variables):
                row = params[spec.name]
                rows += [f"{spec.name}: EXTRACV NAME={spec.name}",
                         f"qmdpa_restraint_{index}: RESTRAINT ARG={spec.name} "
                         f"AT={row['center']:.17g} KAPPA={row['kappa']:.17g}"]
            for row in rows:
                self.kernel.cmd("readInputLine", row)
            (self.directory / "plumed.dat").write_text("\n".join(rows) + "\n")
            (self.directory / "identity.json").write_text(json.dumps(self.identity, indent=2) + "\n")
            self.closed = False
        except Exception:
            self.kernel.close()
            raise

    def evaluate(self, frame, step):
        if self.closed:
            raise ValueError("External CV bias is closed")
        if frame.schema.sha256 != self.schema.sha256:
            raise ValueError("External CV frame schema differs")
        if type(step) is not int or step < 0 or step > 2147483647:
            raise ValueError("External CV step must be a nonnegative 32-bit integer")
        self._values = np.array(frame.values, copy=True)
        self._forces = np.zeros_like(self._values)
        self.kernel.cmd("setStep", ct.c_int(step))
        for index, spec in enumerate(self.schema.variables):
            self.kernel.cmd("setExtraCV " + spec.name, self._values[index:index+1])
            self.kernel.cmd("setExtraCVForce " + spec.name, self._forces[index:index+1])
        self.kernel.cmd("prepareCalc")
        self.kernel.cmd("performCalcNoUpdate")
        energy = ct.c_double()
        self.kernel.cmd("getBias", energy)
        derivatives = {s.name: -float(self._forces[i]) for i, s in enumerate(self.schema.variables)}
        forces = frame.pullback(derivatives)
        if not math.isfinite(energy.value) or not np.isfinite(forces).all():
            raise ValueError("PLUMED returned nonfinite external bias results")
        result = {"bias_energy_eV": energy.value, "derivatives_eV_per_unit": derivatives,
                  "forces_eV_per_angstrom": forces.tolist(), "frame": frame.record(),
                  "frame_sha256": frame.sha256, "identity_sha256": self.identity_sha256,
                  "accepted_step": self.last_step, "accepted_count": self.accepted_count}
        return result

    def accept(self, frame, step):
        if step == self.last_step:
            if frame.sha256 != self.last_frame_sha256:
                raise ValueError("An accepted CV step cannot change geometry, values or derivatives")
            return self.evaluate(frame, step)
        if step != self.last_step + 1:
            raise ValueError("Accepted CV steps must be consecutive from zero")
        result = self.evaluate(frame, step)
        self.kernel.cmd("update")
        self.last_step, self.last_frame_sha256 = step, frame.sha256
        self.accepted_count += 1
        result.update(accepted_step=step, accepted_count=self.accepted_count)
        return result

    def checkpoint(self):
        if self.last_step < 0:
            raise ValueError("No accepted CV frame to checkpoint")
        record = {"identity_sha256": self.identity_sha256, "identity": self.identity,
                  "step": self.last_step, "frame_sha256": self.last_frame_sha256,
                  "accepted_count": self.accepted_count}
        return json.loads(json.dumps({**record, "sha256": digest(record)}))

    def close(self):
        if not self.closed:
            self.kernel.close()
            self.closed = True
