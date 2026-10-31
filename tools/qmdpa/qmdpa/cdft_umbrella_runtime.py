#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulation
# SPDX-License-Identifier: GPL-2.0-only
"""Accepted-step history for the finite constrained-gap umbrella restraint.

The umbrella reuses the existing official PLUMED EXTRACV bridge and the
existing cDFT pair record. It adds no mixer, no solver and no second owner of
the bias energy: the umbrella force is the pullback of the single restraint
derivative through the already validated constrained-gap Jacobian.
"""

from copy import deepcopy

import numpy as np

from .cdft_umbrella import GAP_NAME, bias_energy, enabled, full_region, schema
from .named_cv import digest


def identity(config):
    from .config import software_identity

    umb = config["cdft_umbrella"]
    return {"schema": schema(config).record(),
            "parameters": {GAP_NAME: {"center": umb["center_eV"],
                                      "kappa": umb["kappa_per_eV"],
                                      "unit": "eV"}},
            "software": software_identity(),
            "kernel_sha256": umb["kernel"]["sha256"],
            "wrapper_sha256": umb["wrapper"]["sha256"],
            "timestep_fs": config["timestep_fs"], "history_kind": "static-restraint-v1"}


def initialize(coupling, config, output, previous):
    if not enabled(config):
        return
    from .external_cv import ExternalCVBias

    from .cdft_umbrella import restraint

    calculator = coupling.calculator
    umb = config["cdft_umbrella"]
    libraries = {name: {"path": config["cdft_umbrella_libraries"][name], **umb[name]}
                 for name in ("kernel", "wrapper")}
    calculator.bias = ExternalCVBias(
        schema(config), restraint(config), libraries["kernel"], libraries["wrapper"],
        output/"cdft-umbrella-plumed", timestep_fs=config["timestep_fs"],
        restart=None if previous is None else previous["frame"]["cdft_umbrella"]["history"])


def close(coupling):
    calculator = getattr(coupling, "calculator", None)
    bias = getattr(calculator, "bias", None)
    if bias is not None:
        bias.close()


def capture(frame, coupling, config, previous=None):
    if not enabled(config):
        return
    if coupling.failure is not None or coupling.last_step != frame["step"]:
        raise ValueError("Umbrella acceptance requires the successful current callback")
    calculator = coupling.calculator
    if calculator.last is None or calculator.bias is None:
        raise ValueError("No successful same-frame gap and umbrella evaluation")
    geometry, terms, quantum, cv = calculator.last
    current = _native_geometry(frame, full_region(config))
    if (geometry.atom_ids != current.atom_ids
            or not np.array_equal(geometry.positions, current.positions)):
        raise ValueError("Umbrella evaluation differs from the native frame")

    gap = float(cv.values[0])
    energy, derivative = bias_energy(gap, config)
    bias = calculator.bias.accept(cv, frame["step"])
    if abs(bias["bias_energy_eV"]-energy) > 1e-10:
        raise ValueError("Umbrella restraint energy differs from the admitted window")
    if abs(bias["derivatives_eV_per_unit"][GAP_NAME]-derivative) > 1e-10:
        raise ValueError("Umbrella restraint derivative differs from the admitted window")
    force = -derivative*cv.jacobian[0]
    record = {"schema_version": 1, "schema_sha256": cv.schema.sha256,
              "cv_sha256": cv.sha256,
              "gap_eV": gap,
              "window_center_eV": config["cdft_umbrella"]["center_eV"],
              "kappa_per_eV": config["cdft_umbrella"]["kappa_per_eV"],
              "bias_energy_eV": bias["bias_energy_eV"],
              "derivatives_eV_per_unit": bias["derivatives_eV_per_unit"],
              "bias_forces_eV_per_angstrom": force.tolist(),
              "cv": cv.record(),
              "history": calculator.bias.checkpoint()}
    record["sha256"] = digest(record)
    candidate = {**frame, "cdft_umbrella": record}
    validate_frame(candidate, config, require_history=False)
    if previous is not None:
        # Initial continuation queries are pure. Compare the new electronic
        # results, then retain the previously accepted history byte-for-byte.
        compare_restart(candidate, previous["frame"], config, require_history=False)
        frame["cdft_umbrella"] = deepcopy(previous["frame"]["cdft_umbrella"])
    else:
        frame["cdft_umbrella"] = record
        validate_frame(frame, config)


def _native_geometry(frame, region):
    from .electronic_gap import native_geometry

    return native_geometry(frame, region)


def _close(actual, expected, tolerance, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.allclose(a, b, rtol=0, atol=tolerance)):
        raise ValueError("cDFT umbrella " + label + " differs beyond tolerance")


def validate_frame(frame, config, *, require_history=True):
    record, umb = frame.get("cdft_umbrella"), config.get("cdft_umbrella")
    if umb is None:
        if record is not None:
            raise ValueError("Frame contains an undeclared cDFT umbrella")
        return
    keys = {"schema_version", "schema_sha256", "cv_sha256", "gap_eV", "window_center_eV",
            "kappa_per_eV", "bias_energy_eV", "derivatives_eV_per_unit",
            "bias_forces_eV_per_angstrom", "cv", "history", "sha256"}
    if (not isinstance(record, dict) or set(record) != keys
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or digest({k: v for k, v in record.items() if k != "sha256"}) != record["sha256"]):
        raise ValueError("cDFT umbrella record is missing or corrupt")
    geometry = _native_geometry(frame, full_region(config))
    cv_schema = schema(config)
    if record["schema_sha256"] != cv_schema.sha256:
        raise ValueError("cDFT umbrella schema identity differs")
    if (record["window_center_eV"] != umb["center_eV"]
            or record["kappa_per_eV"] != umb["kappa_per_eV"]):
        raise ValueError("cDFT umbrella window differs from the admitted configuration")
    if (type(record["gap_eV"]) not in (int, float) or not np.isfinite(record["gap_eV"])):
        raise ValueError("cDFT umbrella gap must be a finite scalar")
    energy, derivative = bias_energy(record["gap_eV"], config)
    _close(record["bias_energy_eV"], energy, 1e-10, "restraint energy")
    if set(record["derivatives_eV_per_unit"]) != {GAP_NAME}:
        raise ValueError("cDFT umbrella bias derivative name differs")
    _close(record["derivatives_eV_per_unit"][GAP_NAME], derivative, 1e-10, "bias derivative")
    jacobian = np.asarray(record["cv"]["jacobian_per_angstrom"], dtype=float)
    if jacobian.shape != (1, len(geometry.atom_ids), 3):
        raise ValueError("cDFT umbrella Jacobian shape differs from the atom identity")
    force = -derivative*jacobian[0]
    _close(record["bias_forces_eV_per_angstrom"], force, 1e-10, "bias pullback")
    if record["cv"]["values"] != [record["gap_eV"]]:
        raise ValueError("cDFT umbrella gap differs from the accepted CV value")
    _validate_native_total(frame, config, record)
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
            or history["frame_sha256"] != record["cv_sha256"]):
        raise ValueError("cDFT umbrella accepted history, source or library identity differs")


def _validate_native_total(frame, config, record):
    """Assert that the native total is the sampled surface plus this restraint.

    The cDFT provider owns the sampled surface and deliberately does not
    assert the native total while a bias owner is active. The umbrella is
    therefore the single owner of that assembly, and it must check it here:
    the native total energy is the sampled state energy plus the restraint
    energy, and the native total force is the sampled state force plus the
    restraint pullback.
    """
    from .cdft_runtime import ROLES

    state = frame.get("cdft_states")
    if not isinstance(state, dict) or "states" not in state:
        raise ValueError("cDFT umbrella requires the accepted constrained-state record")
    reference = config["provider"]["state"]["reference"]
    sampled = state["states"][reference["sampled_state"]]
    energy = float(sampled["energy_eV"])
    forces = np.asarray(sampled["forces_eV_per_angstrom"], dtype=float)
    bias_energy = float(record["bias_energy_eV"])
    bias_force = np.asarray(record["bias_forces_eV_per_angstrom"], dtype=float)
    _close(frame["potential_eV"], energy+bias_energy, 1e-7, "native total energy")
    atoms = {a["id"]: a for a in frame["atoms"]}
    geometry = _native_geometry(frame, full_region(config))
    _close([atoms[i]["f"] for i in geometry.atom_ids], forces+bias_force, 1e-5,
           "native total forces")


def compare_restart(frame, reference, config, *, require_history=True):
    validate_frame(frame, config, require_history=require_history)
    validate_frame(reference, config)
    if not enabled(config):
        return
    a, b = frame["cdft_umbrella"], reference["cdft_umbrella"]
    _close(a["gap_eV"], b["gap_eV"], 1e-7, "restart gap")
    _close(a["bias_energy_eV"], b["bias_energy_eV"], 1e-10, "restart bias energy")
    _close(a["bias_forces_eV_per_angstrom"], b["bias_forces_eV_per_angstrom"], 1e-10,
           "restart bias forces")
