# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Accepted native-frame checks for explicit coherent electronic history."""

import numpy as np

from .ehrenfest import mean_field
from .ehrenfest_mechanical import HAMILTONIAN, EhrenfestStepper, _complex_from_record


def validate_frame(frame, config):
    record = frame.get("ehrenfest")
    if config.get("hamiltonian") != HAMILTONIAN:
        if record is not None:
            raise ValueError("Unexpected Ehrenfest history in this Hamiltonian")
        return
    if not isinstance(record, dict):
        raise ValueError("An accepted Ehrenfest frame requires its complete electronic history")  # noqa: TRY004 - Checkpoint API.
    history = EhrenfestStepper(None, timestep_fs=config["timestep_fs"],
                               initial_amplitudes=_complex_from_record(config["initial_amplitudes"]),
                               identity_sha256=config["physics_sha256"], checkpoint=record)
    state = history.state
    full = config["full_region"]
    if (state.step != frame["step"] or state.geometry.atom_ids != tuple(full["atom_ids"])
            or state.geometry.symbols != tuple(full["symbols"])):
        raise ValueError("Electronic history step or full-region identities differ")
    atoms = {a["id"]: a for a in frame["atoms"]}
    if len(atoms) != len(frame["atoms"]) or set(atoms) != set(state.geometry.atom_ids):
        raise ValueError("Electronic history does not cover native atom ownership")
    ordered = [atoms[i] for i in state.geometry.atom_ids]
    if any(a["image"] != [0, 0, 0] for a in ordered):
        raise ValueError("Finite Ehrenfest atoms cannot carry periodic images")
    if not np.array_equal([a["x"] for a in ordered], state.geometry.positions):
        raise ValueError("Electronic history coordinates differ from the native frame")
    value = mean_field(state.pair, state.coefficients)
    if (abs(value.energy - frame["potential_eV"]) > 1e-7
            or not np.allclose([a["f"] for a in ordered], value.forces, atol=1e-6, rtol=0)):
        raise ValueError("Electronic history and native complete energy/forces differ")


def attach_frame(frame, coupling, config):
    if config.get("hamiltonian") != HAMILTONIAN:
        return
    if coupling.failure is not None or coupling.last_step != frame["step"]:
        raise ValueError("Electronic history requires a successful current callback")
    if coupling.comm.rank != 0:
        raise ValueError("Electronic history belongs to the provider owner rank")
    frame["ehrenfest"] = coupling.calculator.record()
    validate_frame(frame, config)


def restore_frame(previous, coupling, config):
    if config.get("hamiltonian") == HAMILTONIAN and previous is not None:
        coupling.calculator.restore(previous["frame"]["ehrenfest"])


def compare_restart(frame, reference, config):
    validate_frame(frame, config)
    validate_frame(reference, config)
    if config.get("hamiltonian") == HAMILTONIAN and frame["ehrenfest"] != reference["ehrenfest"]:
        raise ValueError("Restored coherent electronic history differs")
