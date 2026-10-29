#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Exact single-walker OPES state contracts for the finite water CV profile.

The official PLUMED action owns the KDE and all bias derivatives. KERNELS
are diagnostic output only; immutable STATE snapshots own continuation.
"""

import hashlib
import math
from pathlib import Path

from .dynamics import positive_scalar
from .water_cv import LABELS

WIDTH_NAMES = ("logdistance", "ionization")

FORM = "opes-metad"
HISTORY_FILE = "STATE"


def active(config):
    return config.get("sampling", {}).get("bias", {}).get("form") == FORM


def validate_bias(raw, config):
    if set(raw) != {"form", "sigma", "sigma_min", "barrier_ev", "pace_steps", "bias_factor"}:
        raise ValueError("OPES requires sigma, sigma_min, barrier_ev, pace_steps and bias_factor")
    if "thermostat" not in config:
        raise ValueError("OPES requires the declared constant-temperature thermostat")
    pace = raw["pace_steps"]
    if type(pace) is not int or pace < 1:
        raise ValueError("OPES pace_steps must be a positive integer")
    if config["sample_interval"] % pace or config["steps"] % pace:
        raise ValueError("OPES samples and final step must align with its STATE/deposition cadence")
    widths = {}
    for name in ("sigma", "sigma_min"):
        supplied = raw[name]
        if isinstance(supplied, dict):
            if set(supplied) != set(WIDTH_NAMES):
                raise ValueError("Named water OPES widths require logdistance and ionization")
            supplied = [supplied[key] for key in WIDTH_NAMES]
        if not isinstance(supplied, list) or len(supplied) != len(LABELS):
            raise ValueError("OPES widths require log-distance and ionization values, in that order")
        widths[name] = [positive_scalar(v, "sampling.bias." + name) for v in supplied]
    if any(a > b for a, b in zip(widths["sigma_min"], widths["sigma"], strict=True)):
        raise ValueError("OPES sigma_min cannot exceed its initial sigma")
    factor = positive_scalar(raw["bias_factor"], "sampling.bias.bias_factor")
    if factor <= 1:
        raise ValueError("OPES bias_factor must exceed one")
    return {"form": FORM, **widths,
            "barrier_ev": positive_scalar(raw["barrier_ev"], "sampling.bias.barrier_ev"),
            "pace_steps": pace, "bias_factor": factor,
            "temperature_K": config["thermostat"]["temperature_K"]}


def render(bias, directory, path_token, *, restart=False):
    args = ",".join(LABELS)
    widths = " ".join(name.upper() + "=" + ",".join(f"{v:.17g}" for v in bias[name])
                      for name in ("sigma", "sigma_min"))
    row = (f"qmdpa_bias: OPES_METAD ARG={args} {widths} "
           f"PACE={bias['pace_steps']} BARRIER={bias['barrier_ev']:.17g} "
           f"BIASFACTOR={bias['bias_factor']:.17g} TEMP={bias['temperature_K']:.17g} "
           f"FILE={path_token(directory / 'KERNELS')} "
           f"STATE_WFILE={path_token(directory / HISTORY_FILE)} "
           f"STATE_WSTRIDE={bias['pace_steps']} FMT={{ %.17g}}")
    if restart:
        row += f" STATE_RFILE={path_token(directory / 'PARENT_STATE')} RESTART=YES"
    return row


def history_record(path, config, step):
    bias = config["sampling"]["bias"]
    if type(step) is not int or step < 0 or step % bias["pace_steps"]:
        raise ValueError("OPES history requires a nonnegative aligned step")
    payload = Path(path).read_bytes()
    if step == 0:
        if payload.strip():
            raise ValueError("Fresh OPES must not inherit a preexisting STATE")
        return {"sha256": hashlib.sha256(payload).hexdigest(), "step": 0,
                "counter": 1, "kernel_count": 0}
    expected_fields = ["time", *LABELS, *("sigma_" + v for v in LABELS), "height"]
    metadata, rows, fields_seen = {}, [], False
    for line in payload.decode("ascii").splitlines():
        if not line.strip():
            continue
        if line.startswith("#! FIELDS "):
            if fields_seen or line.split()[2:] != expected_fields:
                raise ValueError("OPES STATE fields differ or contain multiple snapshots")
            fields_seen = True
        elif line.startswith("#! SET "):
            tokens = line.split()
            if len(tokens) != 4 or tokens[2] in metadata:
                raise ValueError("OPES STATE metadata is duplicate or invalid")
            metadata[tokens[2]] = tokens[3]
        elif line.startswith("#"):
            raise ValueError("Unsupported OPES STATE header")
        else:
            row = [float(value) for value in line.split()]
            if not fields_seen or len(row) != 6 or any(not math.isfinite(v) for v in row):
                raise ValueError("OPES STATE requires six finite columns")
            expected_time = step * config["timestep_fs"] / 1000
            if not math.isclose(row[0], expected_time, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("OPES STATE time differs from the accepted step")
            if min(row[3:]) <= 0:
                raise ValueError("OPES STATE widths and kernel height must be positive")
            rows.append(row)
    required = {"action", "biasfactor", "epsilon", "kernel_cutoff", "compression_threshold",
                "zed", "sum_weights", "sum_weights2", "counter"}
    if not fields_seen or not rows or set(metadata) != required:
        raise ValueError("OPES STATE lacks the exact single-walker fixed-input-width metadata")
    if metadata["action"] != "OPES_METAD_state":
        raise ValueError("OPES STATE action differs")
    numeric = {key: float(value) for key, value in metadata.items() if key != "action"}
    if any(not math.isfinite(value) or value <= 0 for value in numeric.values()):
        raise ValueError("OPES STATE metadata must be finite and positive")
    if not math.isclose(numeric["biasfactor"], bias["bias_factor"], rel_tol=1e-14):
        raise ValueError("OPES STATE bias factor differs")
    count = 1 + step // bias["pace_steps"]
    if numeric["counter"] != count or len(rows) > count - 1:
        raise ValueError("OPES STATE counter has missing, duplicate or future depositions")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "step": step,
            "counter": count, "kernel_count": len(rows)}


def validate_frame_history(frame, config):
    record = frame["sampling"].get("history")
    if not isinstance(record, dict) or set(record) != {"sha256", "step", "counter", "kernel_count"}:
        raise ValueError("Missing or invalid OPES frame history")
    digest = record["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Invalid OPES STATE checksum")
    step = frame["step"]
    pace = config["sampling"]["bias"]["pace_steps"]
    if (type(step) is not int or step < 0 or step % pace
            or any(type(record[key]) is not int for key in ("step", "counter", "kernel_count"))
            or record["step"] != step or record["counter"] != 1 + step // pace
            or not (0 if step == 0 else 1) <= record["kernel_count"] <= step // pace):
        raise ValueError("OPES frame history differs from its accepted step")


def snapshot_history(checkpoint, output, live_path, config):
    frame = checkpoint["frame"]
    record = history_record(live_path, config, frame["step"])
    if record != frame["sampling"]["history"]:
        raise ValueError("OPES STATE changed after its accepted frame")
    payload = Path(live_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("OPES STATE changed while taking its snapshot")
    name = f"sampling-state-{frame['step']:012d}.STATE"
    with (Path(output) / name).open("xb") as stream:
        stream.write(payload)
    checkpoint["sampling_history"] = {"file": name, **record}


def validate_checkpoint_history(checkpoint, checkpoint_path, config):
    frame = checkpoint["frame"]
    expected = frame["sampling"]["history"]
    name = f"sampling-state-{frame['step']:012d}.STATE"
    if checkpoint.get("sampling_history") != {"file": name, **expected}:
        raise ValueError("OPES checkpoint STATE differs from its accepted frame")
    path = Path(checkpoint_path).resolve().parent / name
    if history_record(path, config, frame["step"]) != expected:
        raise ValueError("OPES checkpoint STATE checksum or contents differ")
    checkpoint["sampling_history_path"] = str(path)
