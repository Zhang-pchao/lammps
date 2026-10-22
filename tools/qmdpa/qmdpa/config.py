# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Portable run configuration, file identity, and physical-setting validation."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .region import FixedRegion


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def load_config(path):
    """Resolve declared files and validate the deliberately small run schema."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    allowed = {
        "schema_version",
        "data_file",
        "model",
        "model_sha256",
        "type_symbols",
        "periodic",
        "region",
        "full_charge_spin",
        "region_charge_spin",
        "qm",
        "timestep_fs",
        "steps",
        "sample_interval",
        "temperature_K",
        "velocity_seed",
    }
    unknown = set(config).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key in ("data_file", "model"):
        config[key] = str((path.parent / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: {config[key]}")
    actual = file_digest(config["model"])
    if actual != config["model_sha256"]:
        raise ValueError("Model SHA256 does not match the declared model")
    config["data_sha256"] = file_digest(config["data_file"])
    if not isinstance(config.get("periodic"), bool):
        raise ValueError("periodic must be true or false (partial PBC is unsupported)")
    symbols = config["type_symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(
            not isinstance(value, str) or re.fullmatch(r"[A-Z][a-z]?", value) is None
            for value in symbols
        )
    ):
        raise ValueError("type_symbols must map each LAMMPS atom type to an element")
    region = config["region"]
    if set(region) != {"atom_ids", "symbols", "molecules"}:
        raise ValueError("region requires atom_ids, symbols, and complete molecules")
    FixedRegion(tuple(region["atom_ids"]), tuple(region["symbols"]))
    molecules = region["molecules"]
    if (
        not isinstance(molecules, list)
        or not molecules
        or any(not isinstance(molecule, list) or not molecule for molecule in molecules)
    ):
        raise ValueError("region.molecules must contain nonempty lists of atom IDs")
    members = [atom for molecule in molecules for atom in molecule]
    if any(type(atom) is not int or atom <= 0 for atom in members):
        raise ValueError("molecule atom IDs must be positive integers")
    if sorted(members) != sorted(region["atom_ids"]):
        raise ValueError("region.molecules must partition the declared region IDs exactly")
    for key in ("full_charge_spin", "region_charge_spin"):
        value = config.get(key)
        if value is not None and (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) is not int for item in value)
            or value[1] < 1
        ):
            raise ValueError(f"{key} requires integer charge and positive multiplicity")
    qm_defaults = {
        "xc": "pbe",
        "basis": "def2-svp",
        "charge": 0,
        "multiplicity": 1,
        "conv_tol": 1e-9,
        "max_cycle": 100,
        "grid_level": 3,
        "threads": None,
    }
    qm = config["qm"]
    if not isinstance(qm, dict) or set(qm).difference(qm_defaults):
        raise ValueError("qm contains unsupported PySCF configuration keys")
    qm = {**qm_defaults, **qm}
    for key in ("xc", "basis"):
        if not isinstance(qm[key], str) or not qm[key].strip():
            raise ValueError(f"qm.{key} must be a nonempty string")
    if type(qm["charge"]) is not int:
        raise ValueError("qm.charge must be an integer")
    for key in ("multiplicity", "max_cycle"):
        _positive_integer(qm[key], f"qm.{key}")
    _positive_integer(qm["grid_level"], "qm.grid_level", allow_zero=True)
    if qm["threads"] is not None:
        _positive_integer(qm["threads"], "qm.threads")
    if isinstance(qm["conv_tol"], bool) or not np.isfinite(qm["conv_tol"]) or qm["conv_tol"] <= 0:
        raise ValueError("qm.conv_tol must be finite and positive")
    config["qm"] = qm
    state = config.get("region_charge_spin")
    if state is not None and state != [qm["charge"], qm["multiplicity"]]:
        raise ValueError("Region DP and QM electronic states must match in this driver")
    config["steps"] = _positive_integer(config["steps"], "steps", allow_zero=True)
    config["sample_interval"] = _positive_integer(
        config.get("sample_interval", 1), "sample_interval"
    )
    if isinstance(config["timestep_fs"], bool):
        raise ValueError("timestep_fs must be a number")
    step = float(config["timestep_fs"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError("timestep_fs must be finite and positive")
    config["timestep_fs"] = step
    if "temperature_K" in config:
        if isinstance(config["temperature_K"], bool):
            raise ValueError("temperature_K must be a number")
        temperature = float(config["temperature_K"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature_K must be finite and positive")
        config["temperature_K"] = temperature
        _positive_integer(config["velocity_seed"], "velocity_seed")
    physics = {
        key: config.get(key)
        for key in (
            "model_sha256",
            "data_sha256",
            "type_symbols",
            "periodic",
            "region",
            "qm",
            "full_charge_spin",
            "region_charge_spin",
            "timestep_fs",
        )
    }
    config["physics_sha256"] = hashlib.sha256(
        json.dumps(physics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config
