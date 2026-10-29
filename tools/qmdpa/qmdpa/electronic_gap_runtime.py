#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Accepted-step history for the finite ground/TDA external restraint."""

from copy import deepcopy

import numpy as np

from .core import Evaluation
from .electronic_gap import gap_frame, native_geometry, schema_for
from .named_cv import digest


def identity(config):
    from .config import software_identity

    ext = config["electronic_gap"]
    return {"schema": schema_for(config).record(),
            "parameters": {ext["name"]: ext["restraint"]}, "software": software_identity(),
            "kernel_sha256": ext["kernel"]["sha256"],
            "wrapper_sha256": ext["wrapper"]["sha256"],
            "timestep_fs": config["timestep_fs"], "history_kind": "static-restraint-v1"}


def initialize(coupling, config, output, previous):
    if "electronic_gap" not in config:
        return
    from .external_cv import ExternalCVBias

    calculator = coupling.calculator
    ext = config["electronic_gap"]
    libraries = {name: {"path": config["electronic_gap_libraries"][name], **ext[name]}
                 for name in ("kernel", "wrapper")}
    calculator.bias = ExternalCVBias(
        calculator.schema, {ext["name"]: ext["restraint"]}, libraries["kernel"], libraries["wrapper"],
        output/"electronic-gap-plumed", timestep_fs=config["timestep_fs"],
        restart=None if previous is None else previous["frame"]["electronic_gap"]["history"])


def close(coupling):
    calculator = getattr(coupling, "calculator", None)
    bias = getattr(calculator, "bias", None)
    if bias is not None:
        bias.close()


def capture(frame, coupling, config, previous=None):
    if "electronic_gap" not in config:
        return
    if coupling.failure is not None or coupling.last_step != frame["step"]:
        raise ValueError("Gap acceptance requires the successful current callback")
    calculator = coupling.calculator
    if calculator.last is None or calculator.bias is None:
        raise ValueError("No successful same-frame gap and bias evaluation")
    geometry, states, cv, bias, total, parity = calculator.last
    current = native_geometry(frame, config["electronic_gap"]["full_region"])
    if (geometry.atom_ids != current.atom_ids
            or not np.array_equal(geometry.positions, current.positions)):
        raise ValueError("Electronic gap evaluation differs from the native frame")
    record = {"schema_version": 1, "schema_sha256": cv.schema.sha256,
              "states": {k: {"energy_eV": v.energy, "forces_eV_per_angstrom": v.forces.tolist()}
                         for k, v in states.items()},
              "cv": cv.record(), "bias_energy_eV": bias["bias_energy_eV"],
              "derivatives_eV_per_unit": bias["derivatives_eV_per_unit"],
              "bias_forces_eV_per_angstrom": bias["forces_eV_per_angstrom"],
              "ground_parity_eV": parity, "history": None}
    record["sha256"] = digest(record)
    candidate = {**frame, "electronic_gap": record}
    validate_frame(candidate, config, require_history=False)
    if previous is not None:
        # Initial continuation queries are pure. Compare new electronic results,
        # then retain the previously accepted history byte-for-byte. Subsequent
        # successful steps commit their newly evaluated CV frame normally.
        compare_restart(candidate, previous["frame"], config, require_history=False)
        frame["electronic_gap"] = deepcopy(previous["frame"]["electronic_gap"])
    else:
        calculator.bias.accept(cv, frame["step"])
        record["history"] = calculator.bias.checkpoint()
        record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
        frame["electronic_gap"] = record
        validate_frame(frame, config)


def _close(actual, expected, tolerance, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=tolerance):
        raise ValueError("Electronic gap " + label + " differs beyond tolerance")


def validate_frame(frame, config, *, require_history=True):
    record, ext = frame.get("electronic_gap"), config.get("electronic_gap")
    if ext is None:
        if record is not None:
            raise ValueError("Frame contains an undeclared electronic gap")
        return
    keys = {"schema_version", "schema_sha256", "states", "cv", "bias_energy_eV",
            "derivatives_eV_per_unit", "bias_forces_eV_per_angstrom", "ground_parity_eV",
            "history", "sha256"}
    if (not isinstance(record, dict) or set(record) != keys
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or digest({k: v for k, v in record.items() if k != "sha256"}) != record["sha256"]):
        raise ValueError("Electronic gap record is missing or corrupt")
    geometry = native_geometry(frame, ext["full_region"])
    schema = schema_for(config)
    if record["schema_sha256"] != schema.sha256 or set(record["states"]) != {"ground", "selected"}:
        raise ValueError("Electronic gap state or schema identity differs")
    states = {k: Evaluation(v["energy_eV"], v["forces_eV_per_angstrom"])
              for k, v in record["states"].items()}
    if any(v.forces.shape != geometry.positions.shape for v in states.values()):
        raise ValueError("Electronic gap requires complete two-state force arrays")
    cv = gap_frame(schema, geometry, states, ext["order"])
    if record["cv"] != cv.record():
        raise ValueError("Electronic gap value or Jacobian differs from both complete states")
    if type(record["ground_parity_eV"]) not in (int, float) or not np.isfinite(record["ground_parity_eV"]) or abs(record["ground_parity_eV"]) > 1e-7:
        raise ValueError("Electronic gap ground parity failed")
    delta = float(cv.values[0])-ext["restraint"]["center"]
    derivative = ext["restraint"]["kappa"]*delta
    _close(record["bias_energy_eV"], .5*derivative*delta, 1e-10, "restraint energy")
    if set(record["derivatives_eV_per_unit"]) != {ext["name"]}:
        raise ValueError("Electronic gap bias derivative name differs")
    _close(record["derivatives_eV_per_unit"][ext["name"]], derivative, 1e-10, "bias derivative")
    force = -derivative*cv.jacobian[0]
    _close(record["bias_forces_eV_per_angstrom"], force, 1e-10, "bias pullback")
    sampled = states[ext["sampled_surface"]]
    _close(frame["potential_eV"], sampled.energy+record["bias_energy_eV"], 1e-7, "native total energy")
    atoms = {a["id"]: a for a in frame["atoms"]}
    _close([atoms[i]["f"] for i in geometry.atom_ids], sampled.forces+force, 1e-5, "native total forces")
    if not require_history:
        return
    history = record["history"]
    expected = identity(config)
    if (not isinstance(history, dict) or set(history) != {
            "identity_sha256", "identity", "step", "frame_sha256", "accepted_count", "sha256"}
            or digest({k: v for k, v in history.items() if k != "sha256"}) != history["sha256"]
            or history["identity"] != expected or history["identity_sha256"] != digest(expected)
            or type(history["step"]) is not int or history["step"] != frame["step"]
            or type(history["accepted_count"]) is not int or history["accepted_count"] != frame["step"]+1
            or history["frame_sha256"] != cv.sha256):
        raise ValueError("Electronic gap accepted history, source or library identity differs")


def compare_restart(frame, reference, config, *, require_history=True):
    validate_frame(frame, config, require_history=require_history)
    validate_frame(reference, config)
    if "electronic_gap" not in config:
        return
    a, b = frame["electronic_gap"], reference["electronic_gap"]
    for name in ("ground", "selected"):
        _close(a["states"][name]["energy_eV"], b["states"][name]["energy_eV"], 1e-7, "restart state energy")
        _close(a["states"][name]["forces_eV_per_angstrom"], b["states"][name]["forces_eV_per_angstrom"], 1e-5, "restart state forces")
    _close(a["cv"]["values"], b["cv"]["values"], 1e-7, "restart CV")
    _close(a["cv"]["jacobian_per_angstrom"], b["cv"]["jacobian_per_angstrom"], 1e-5, "restart Jacobian")
