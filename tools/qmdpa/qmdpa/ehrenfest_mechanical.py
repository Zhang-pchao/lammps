# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Complete fixed-QM mechanical surfaces and transactional electronic history."""

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from .core import EvaluationError, Geometry, _atom_ids
from .ehrenfest import SurfacePair, amplitudes, electronic_step, mean_field

HAMILTONIAN = "finite-two-state-ehrenfest-mechanical"


class MechanicalPair:
    """Add the same DPA difference to both states, with zero environment NAC."""

    def __init__(self, full_low, region_low, quantum, *, qm_atom_ids):
        self.full_low, self.region_low, self.quantum = full_low, region_low, quantum
        self.qm_atom_ids = _atom_ids(qm_atom_ids)

    def evaluate_pair(self, geometry):
        if not isinstance(geometry, Geometry) or geometry.periodic:
            raise ValueError("Finite two-state ME requires a nonperiodic Geometry")
        qm = geometry.select(self.qm_atom_ids)
        high = self.quantum.evaluate_pair(qm)
        if not isinstance(high, SurfacePair) or high.forces.shape[1] != len(qm.atom_ids):
            raise EvaluationError("Quantum state-pair force count differs from QM atoms")
        full_low = self.full_low.evaluate(geometry)
        region_low = self.region_low.evaluate(qm)
        if (full_low.forces.shape != geometry.positions.shape
                or region_low.forces.shape != qm.positions.shape):
            raise EvaluationError("Low-level force counts differ from declared atoms")
        energies = full_low.energy + high.energies - region_low.energy
        forces = np.repeat(full_low.forces[None], 2, axis=0)
        rows = [geometry.atom_ids.index(atom) for atom in qm.atom_ids]
        forces[:, rows, :] += high.forces - region_low.forces
        nac = np.zeros_like(geometry.positions)
        nac[rows] = high.nac01
        return SurfacePair(energies, forces, nac)


def _digest(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _same_domain(left, right):
    return (left.atom_ids == right.atom_ids and left.symbols == right.symbols
            and not left.periodic and not right.periodic)


def _complex_record(coefficients):
    return np.column_stack((coefficients.real, coefficients.imag)).tolist()


def _complex_from_record(values):
    array = np.asarray(values)
    if array.shape != (2, 2) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Electronic state requires two finite real/imaginary pairs")
    return amplitudes(array[:, 0] + 1j * array[:, 1])


@dataclass(frozen=True)
class ElectronicSnapshot:
    step: int
    geometry: Geometry
    coefficients: np.ndarray
    pair: SurfacePair

    def __post_init__(self):
        if type(self.step) is not int or self.step < 0:
            raise ValueError("Electronic step must be a nonnegative integer")
        if not isinstance(self.geometry, Geometry) or self.geometry.periodic:
            raise ValueError("Electronic history requires a finite Geometry")
        if not isinstance(self.pair, SurfacePair) or self.pair.nac01.shape != self.geometry.positions.shape:
            raise ValueError("Electronic surfaces do not match the stored atoms")
        object.__setattr__(self, "coefficients", amplitudes(self.coefficients))


class EhrenfestStepper:
    """Advance electrons once per native nuclear step; never integrate nuclei.

    A failed instance is permanently invalid. Its last accepted snapshot remains
    inspectable, but further evaluation needs a new instance from an immutable
    checkpoint. Duplicate same-step/geometry callbacks do not propagate history.
    """

    def __init__(self, evaluator, *, timestep_fs, initial_amplitudes, identity_sha256,
                 checkpoint=None):
        if (type(timestep_fs) not in (int, float) or not np.isfinite(timestep_fs)
                or timestep_fs <= 0):
            raise ValueError("Ehrenfest timestep must be a positive finite scalar")
        if not isinstance(identity_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", identity_sha256) is None:
            raise ValueError("Electronic history requires a physics SHA256")
        self.evaluator = evaluator
        self.timestep_fs = float(timestep_fs)
        self.initial_amplitudes = amplitudes(initial_amplitudes)
        self.identity_sha256 = identity_sha256
        self.failure = None
        self.state = None
        self._restore_pending = False
        if checkpoint is not None:
            self.restore(checkpoint)

    def restore(self, checkpoint):
        if self.state is not None or self.failure is not None:
            raise EvaluationError("Electronic history can only restore a fresh instance")
        self.state = self._read_record(checkpoint)
        self._restore_pending = True

    def _read_record(self, record):
        fields = {"schema_version", "model", "physics_sha256", "timestep_fs", "initial_amplitudes",
                  "step", "atom_ids", "symbols", "positions_angstrom", "amplitudes",
                  "energies_ev", "forces_ev_angstrom", "d01_per_angstrom", "sha256"}
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("Electronic history has missing or unknown fields")
        payload = {k: v for k, v in record.items() if k != "sha256"}
        if record["sha256"] != _digest(payload):
            raise ValueError("Electronic history checksum differs")
        if (type(record["schema_version"]) is not int or record["schema_version"] != 1
                or record["model"] != HAMILTONIAN or record["physics_sha256"] != self.identity_sha256
                or type(record["timestep_fs"]) not in (int, float)
                or record["timestep_fs"] != self.timestep_fs
                or not np.array_equal(_complex_from_record(record["initial_amplitudes"]), self.initial_amplitudes)):
            raise ValueError("Electronic history physics or initialization differs")
        geometry = Geometry(record["atom_ids"], record["symbols"], record["positions_angstrom"])
        pair = SurfacePair(record["energies_ev"], record["forces_ev_angstrom"], record["d01_per_angstrom"])
        return ElectronicSnapshot(record["step"], geometry, _complex_from_record(record["amplitudes"]), pair)

    def record(self):
        if self.state is None:
            raise ValueError("No accepted electronic snapshot is available")
        state = self.state
        payload = {"schema_version": 1, "model": HAMILTONIAN,
                   "physics_sha256": self.identity_sha256, "timestep_fs": self.timestep_fs,
                   "initial_amplitudes": _complex_record(self.initial_amplitudes),
                   "step": state.step, "atom_ids": list(state.geometry.atom_ids),
                   "symbols": list(state.geometry.symbols),
                   "positions_angstrom": state.geometry.positions.tolist(),
                   "amplitudes": _complex_record(state.coefficients),
                   "energies_ev": state.pair.energies.tolist(),
                   "forces_ev_angstrom": state.pair.forces.tolist(),
                   "d01_per_angstrom": state.pair.nac01.tolist()}
        return {**payload, "sha256": _digest(payload)}

    def evaluate(self, geometry, step):
        if self.failure is not None:
            raise EvaluationError(self.failure)
        try:
            result = self._evaluate(geometry, step)
        except Exception as error:
            self.failure = f"Electronic history failed at step {step}: {error}"
            raise
        return result

    def _evaluate(self, geometry, step):
        if type(step) is not int or step < 0 or not isinstance(geometry, Geometry) or geometry.periodic:
            raise ValueError("A nonnegative step and finite geometry are required")
        previous = self.state
        if previous is None:
            if step != 0:
                raise EvaluationError("Fresh electronic history must start at step zero")
            pair = self.evaluator.evaluate_pair(geometry)
            candidate = ElectronicSnapshot(step, geometry, self.initial_amplitudes, pair)
        else:
            if not _same_domain(previous.geometry, geometry):
                raise EvaluationError("Electronic history atom order or element identity changed")
            if step == previous.step:
                if not np.array_equal(previous.geometry.positions, geometry.positions):
                    raise EvaluationError("A repeated electronic step has different coordinates")
                if self._restore_pending:
                    current = self.evaluator.evaluate_pair(geometry)
                    for name, limit in (("energies", 1e-7), ("forces", 1e-6), ("nac01", 1e-7)):
                        if np.max(np.abs(getattr(current, name) - getattr(previous.pair, name))) > limit:
                            raise EvaluationError(f"Restored electronic {name} differs from a fresh solution")
                    # Keep the verified endpoint history for bitwise reproducible
                    # continuation. No unverified or failed provider data is used.
                    self._restore_pending = False
                return mean_field(previous.pair, previous.coefficients)
            if self._restore_pending or step != previous.step + 1:
                raise EvaluationError("Electronic history skipped a step or was not restored")
            current = self.evaluator.evaluate_pair(geometry)
            delta = geometry.positions - previous.geometry.positions
            coefficients = electronic_step(previous.coefficients, previous.pair, current,
                                           delta, self.timestep_fs)
            candidate = ElectronicSnapshot(step, geometry, coefficients, current)
        result = mean_field(candidate.pair, candidate.coefficients)
        self.state = candidate
        return result
