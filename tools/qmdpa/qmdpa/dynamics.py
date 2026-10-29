#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Native constant-temperature integration for admitted finite ground models."""

import hashlib
import json
import math


def require_ground_sampling(config):
    if (config["hamiltonian"] not in ("fixed-region-mechanical",
                                    "finite-reactive-water-mechanical",
                                    "finite-electrostatic-additive") or config["periodic"]
            or config["provider"]["engine"] != "pyscf"
            or config["provider"]["device"] != "cpu"
            or config["provider"]["state"]["method"] != "ground"):
        raise ValueError("Sampling requires finite CPU PySCF ground ME or fixed additive EE")


def require_ground_me(config):
    require_ground_sampling(config)
    if config["hamiltonian"] == "finite-electrostatic-additive":
        raise ValueError("This execution profile requires ground ME")


def fixed_sampling_region(config):
    """Return the fixed QM selection without changing the Hamiltonian schema."""
    key = "qm_region" if config["hamiltonian"] == "finite-electrostatic-additive" else "region"
    return config[key]


def positive_scalar(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def extend_identity(config, name, value):
    base = config["physics_sha256"]
    config[name] = value
    config["physics_sha256"] = hashlib.sha256(json.dumps(
        {"base_physics_sha256": base, name: value},
        sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def bind_thermostat(config, raw):
    require_ground_sampling(config)
    if not isinstance(raw, dict) or set(raw) != {
            "form", "temperature_K", "damping_fs", "chain_length"}:
        raise ValueError("thermostat requires form, temperature_K, damping_fs and chain_length")
    if raw["form"] != "nose-hoover-chain":
        raise ValueError("Only the native nose-hoover-chain thermostat is supported")
    chain = raw["chain_length"]
    if type(chain) is not int or not 1 <= chain <= 10:
        raise ValueError("thermostat.chain_length must be an integer from 1 to 10")
    value = {
        "form": raw["form"],
        "temperature_K": positive_scalar(raw["temperature_K"], "thermostat.temperature_K"),
        "damping_fs": positive_scalar(raw["damping_fs"], "thermostat.damping_fs"),
        "chain_length": chain,
    }
    extend_identity(config, "thermostat", value)
    return config


def integrator_command(config):
    value = config.get("thermostat")
    if value is None:
        return "fix qmdpa_integrator all nve"
    return (
        "fix qmdpa_integrator all nvt temp "
        f"{value['temperature_K']:.17g} {value['temperature_K']:.17g} "
        f"{value['damping_fs'] / 1000:.17g} tchain {value['chain_length']}"
    )


def capture_thermostat(lmp, frame, comm, config):
    if "thermostat" not in config:
        return
    # The fix scalar is collective; all ranks must evaluate it before owner I/O.
    energy = float(lmp.extract_fix("qmdpa_integrator", 0, 0))
    from .runtime import _root_call

    def attach():
        if not math.isfinite(energy):
            raise ValueError("Native thermostat energy is not finite")
        frame["thermostat"] = {
            "form": config["thermostat"]["form"],
            "energy_eV": energy,
            "extended_energy_eV": frame["potential_eV"] + frame["kinetic_eV"] + energy,
        }
        validate_thermostat_frame(frame, config)
    _root_call(comm, attach)


def validate_thermostat_frame(frame, config):
    record = frame.get("thermostat")
    if "thermostat" not in config:
        if record is not None:
            raise ValueError("Unexpected thermostat frame record")
        return
    if not isinstance(record, dict) or set(record) != {
            "form", "energy_eV", "extended_energy_eV"}:
        raise ValueError("Missing or invalid thermostat frame record")
    if record["form"] != config["thermostat"]["form"]:
        raise ValueError("Thermostat frame form differs")
    for name in ("energy_eV", "extended_energy_eV"):
        if type(record[name]) not in (int, float) or not math.isfinite(record[name]):
            raise ValueError("Thermostat frame energies must be finite scalars")
    expected = frame["potential_eV"] + frame["kinetic_eV"] + record["energy_eV"]
    if abs(expected - record["extended_energy_eV"]) > 1e-9:
        raise ValueError("Thermostat extended-energy ledger differs")


def compare_thermostat_restart(frame, reference, config):
    for item in (frame, reference):
        validate_thermostat_frame(item, config)
    if ("thermostat" in config and abs(frame["thermostat"]["energy_eV"] -
                                      reference["thermostat"]["energy_eV"]) > 1e-9):
        raise ValueError("Restart thermostat state energy differs")
