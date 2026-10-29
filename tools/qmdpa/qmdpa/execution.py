#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit ordinary/Kokkos execution and bounded float32 restart policy."""

import atexit
import json
import zipfile
from pathlib import Path

from .dynamics import extend_identity, require_ground_me

PRACTICAL = "float32-practical-v1"
_CLEANUP_HANDLES = set()


def bind_execution(config, raw):
    domain = config["hamiltonian"] == "finite-reactive-water-additive"
    required = {"pair_style", "numerical_profile"} | ({"domain"} if domain else set())
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("execution requires pair_style, numerical_profile and an explicit environment domain for reactive EE")
    if domain:
        from .inference import COMPRESSED_PROFILE

        if config["inference_profile"] != COMPRESSED_PROFILE or raw["domain"] != "environment":
            raise ValueError("Reactive native execution requires the compressed environment-domain profile")
        if config["buffer_reference"]["low_backend"] != raw["pair_style"]:
            raise ValueError("The water reference low backend differs from native execution")
        if config["environment_charge_spin"] != [0, 1]:
            raise ValueError("The practical profile requires a neutral-singlet environment")
    else:
        require_ground_me(config)
        if any(key in config for key in ("sampling", "thermostat", "bias")):
            raise ValueError("The practical float32 profile currently admits NVE without sampling")
        if (config.get("full_charge_spin") != [0, 1]
                or config.get("region_charge_spin") != [0, 1]):
            raise ValueError("The practical profile requires neutral singlet full and QM systems")
    if raw["pair_style"] not in ("deepmd", "deepmd/kk"):
        raise ValueError("execution.pair_style must be deepmd or deepmd/kk")
    if raw["numerical_profile"] != PRACTICAL:
        raise ValueError("The declared execution profile must be float32-practical-v1")
    validate_dpa4c_archive(config["model"])
    extend_identity(config, "execution", dict(raw))
    return config


def validate_dpa4c_archive(path):
    """Inspect the official archive without importing an inference engine."""
    path = Path(path)
    if path.suffix != ".pt2":
        raise ValueError("The practical profile requires an identified DPA4C .pt2 archive")
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("model/extra/metadata.json"))
            model = json.loads(archive.read("model/extra/model_def_script.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("Cannot read the declared DPA4C archive metadata") from exc
    if not isinstance(model, dict) or not isinstance(metadata, dict):
        raise TypeError("DPA4C archive metadata must contain objects")
    descriptor = model.get("descriptor", {})
    fitting = model.get("fitting_net", {})
    if not isinstance(descriptor, dict) or not isinstance(fitting, dict):
        raise TypeError("DPA4C descriptor and fitting metadata must contain objects")
    if (descriptor.get("type") != "dpa4c"
            or descriptor.get("precision") != "float32"
            or fitting.get("precision") != "float32"
            or descriptor.get("use_amp", False)):
        raise ValueError("The practical profile requires float32 DPA4C without AMP")
    if metadata.get("lower_input_kind") not in ("graph", "dpa4c_canonical"):
        raise ValueError("The practical profile requires an official graph/canonical archive")


def uses_native_domains(config):
    return (config.get("hamiltonian") == "finite-reactive-water-additive"
            and config.get("execution", {}).get("domain") == "environment")


def native_arguments(config):
    arguments = ["-log", "none", "-screen", "none", "-nocite"]
    if config.get("execution", {}).get("pair_style") == "deepmd/kk" and not uses_native_domains(config):
        arguments += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                      "newton", "off", "neigh", "full", "gpu/aware", "off"]
    return arguments


def pair_style(config):
    return config.get("execution", {}).get("pair_style", "deepmd")


def verify_native(lmp, config):
    expected = "zero" if uses_native_domains(config) else pair_style(config)
    if "execution" in config and lmp.extract_global("pair_style") != expected:
        raise ValueError("The actual native pair differs from the declared execution profile")


def restart_limits(config):
    """Only E/F arithmetic may differ; coordinates, IDs and history remain strict."""
    if config.get("execution", {}).get("numerical_profile") == PRACTICAL:
        return 1e-4, 1e-3, "1e-4"
    return 1e-6, 1e-5, "1e-6"

def register_native_cleanup(lmp, config):
    """Finalize the owned Kokkos runtime before process-level CUDA unloading.

    LAMMPS close() intentionally permits another instance in the same process.
    Its finalize() also terminates MPI/Python and is inappropriate here. The
    public Kokkos-only C API is registered once per library for process exit.
    """
    if pair_style(config) != "deepmd/kk" or uses_native_domains(config):
        return
    handle = lmp.lib._handle
    if handle not in _CLEANUP_HANDLES:
        finalize = lmp.lib.lammps_kokkos_finalize
        finalize.argtypes = []
        finalize.restype = None
        atexit.register(finalize)
        _CLEANUP_HANDLES.add(handle)
