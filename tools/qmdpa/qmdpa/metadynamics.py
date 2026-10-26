#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Immutable history contracts for one finite, fixed-width PLUMED WTMetaD CV.

PLUMED owns the bias and derivatives. This module validates and snapshots its
HILLS records; it does not implement a second production potential.
"""

import hashlib
import math
from pathlib import Path

from .dynamics import positive_scalar

FORM = "well-tempered-metadynamics"


def active(config):
    return config.get("sampling", {}).get("bias", {}).get("form") == FORM


def validate_bias(raw, config):
    if set(raw) != {"form", "sigma_angstrom", "height_ev", "pace_steps", "bias_factor"}:
        raise ValueError("WTMetaD requires form, sigma_angstrom, height_ev, pace_steps and bias_factor")
    if "thermostat" not in config:
        raise ValueError("WTMetaD requires the declared constant-temperature thermostat")
    pace = raw["pace_steps"]
    if type(pace) is not int or pace < 1:
        raise ValueError("WTMetaD pace_steps must be a positive integer")
    factor = positive_scalar(raw["bias_factor"], "sampling.bias.bias_factor")
    if factor <= 1:
        raise ValueError("WTMetaD bias_factor must exceed one")
    return {
        "form": FORM,
        "sigma_angstrom": positive_scalar(raw["sigma_angstrom"], "sampling.bias.sigma_angstrom"),
        "height_ev": positive_scalar(raw["height_ev"], "sampling.bias.height_ev"),
        "pace_steps": pace, "bias_factor": factor,
        "temperature_K": config["thermostat"]["temperature_K"],
    }


def history_record(path, config, step):
    """Validate every deposited hill and derive its cumulative deposition work."""
    payload = Path(path).read_bytes()
    bias = config["sampling"]["bias"]
    rows = []
    fields = False
    for line in payload.decode("ascii").splitlines():
        if not line.strip():
            continue
        if line.startswith("#! FIELDS "):
            if line.split()[2:] != ["time", "qmdpa_cv", "sigma_qmdpa_cv", "height", "biasf"]:
                raise ValueError("WTMetaD history fields differ")
            fields = True
        elif line.startswith("#! SET "):
            if line.split()[2:] not in [
                    ["multivariate", "false"], ["kerneltype", "stretched-gaussian"]]:
                raise ValueError("Unsupported WTMetaD history metadata")
        elif line.startswith("#"):
            raise ValueError("Unsupported WTMetaD history header")
        else:
            if not fields:
                raise ValueError("WTMetaD history is missing its fields")
            row = [float(x) for x in line.split()]
            if len(row) != 5 or any(not math.isfinite(x) for x in row):
                raise ValueError("WTMetaD history requires five finite columns")
            rows.append(row)
    if len(rows) != step // bias["pace_steps"]:
        raise ValueError("WTMetaD history has missing, duplicate or future hills")
    work = []
    for i, (time, _cv, sigma, height, factor) in enumerate(rows, 1):
        expected = i * bias["pace_steps"] * config["timestep_fs"] / 1000
        if not math.isclose(time, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("WTMetaD hill time differs from the deposition schedule")
        if not math.isclose(sigma, bias["sigma_angstrom"], rel_tol=1e-14, abs_tol=0):
            raise ValueError("WTMetaD history Gaussian width differs")
        if not math.isclose(factor, bias["bias_factor"], rel_tol=1e-14, abs_tol=0):
            raise ValueError("WTMetaD history bias factor differs")
        # PLUMED writes the asymptotic free-energy hill height in WT HILLS.
        # Convert it back to deposited potential, not an extra force or energy.
        deposited = height * (factor - 1) / factor
        if not 0 < deposited <= bias["height_ev"] * (1 + 1e-12):
            raise ValueError("WTMetaD hill height is outside its declared bound")
        work.append(deposited)
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hill_count": len(rows),
        "last_deposition_step": len(rows) * bias["pace_steps"],
        "deposition_work_eV": math.fsum(work),
    }


def validate_frame_history(frame, config):
    record = frame["sampling"].get("history")
    if not isinstance(record, dict) or set(record) != {
            "sha256", "hill_count", "last_deposition_step", "deposition_work_eV"}:
        raise ValueError("Missing or invalid WTMetaD frame history")
    digest = record["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Invalid WTMetaD history checksum")
    bias = config["sampling"]["bias"]
    count = frame["step"] // bias["pace_steps"]
    if (type(record["hill_count"]) is not int or record["hill_count"] != count
            or type(record["last_deposition_step"]) is not int
            or record["last_deposition_step"] != count * bias["pace_steps"]):
        raise ValueError("WTMetaD frame hill count differs from its step")
    work = record["deposition_work_eV"]
    if (type(work) not in (int, float) or not math.isfinite(work)
            or not 0 <= work <= count * bias["height_ev"] * (1 + 1e-12)
            or (count > 0 and work <= 0)):
        raise ValueError("Invalid WTMetaD deposition work")


def snapshot_history(checkpoint, output, live_path, config):
    """Write a new immutable history before the accepted checkpoint is published."""
    frame = checkpoint["frame"]
    record = history_record(live_path, config, frame["step"])
    if record != frame["sampling"]["history"]:
        raise ValueError("WTMetaD history changed after its frame was captured")
    name = f"sampling-history-{frame['step']:012d}.HILLS"
    payload = Path(live_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("WTMetaD history changed while taking its snapshot")
    with (Path(output) / name).open("xb") as stream:
        stream.write(payload)
    checkpoint["sampling_history"] = {"file": name, **record}


def validate_checkpoint_history(checkpoint, checkpoint_path, config):
    record = checkpoint.get("sampling_history")
    if not active(config):
        if record is not None:
            raise ValueError("Unexpected WTMetaD checkpoint history")
        return
    frame = checkpoint["frame"]
    expected = frame["sampling"]["history"]
    name = f"sampling-history-{frame['step']:012d}.HILLS"
    if record != {"file": name, **expected}:
        raise ValueError("WTMetaD checkpoint history differs from its accepted frame")
    path = Path(checkpoint_path).resolve().parent / name
    if history_record(path, config, frame["step"]) != expected:
        raise ValueError("WTMetaD checkpoint history checksum or contents differ")
    checkpoint["sampling_history_path"] = str(path)
