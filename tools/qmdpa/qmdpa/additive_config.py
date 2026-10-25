# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Configuration for finite additive dynamics, separate from mechanical embedding."""

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

from .additive import CrossLennardJones
from .charges import _invalid_constant, _unique_object, resolve_fixed_charges
from .config import _positive_integer, file_digest, software_identity, validate_qm, validate_region
from .embedding import FixedEmbeddingCharges
from .inference import RAW_PROFILE
from .providers import provider_contract, validate_provider, validate_provider_regions
from .state_inputs import resolve_provider_reference


def cross_parameters(cross):
    """Parse each explicit directional pair, rejecting duplicates and hidden defaults."""
    if not isinstance(cross, dict) or set(cross) != {"form", "source", "cutoff", "mixing", "pairs"}:
        raise ValueError("cross requires form, source, cutoff, mixing and pairs")
    if cross["form"] != "lj12-6" or cross["cutoff"] is not None or cross["mixing"] != "none":
        raise ValueError("Require untruncated cross LJ with no mixing rules")
    if not isinstance(cross["pairs"], list):
        raise TypeError("cross.pairs must be a list")
    pairs = {}
    for pair in cross["pairs"]:
        if not isinstance(pair, dict) or set(pair) != {
            "qm_symbol",
            "environment_symbol",
            "epsilon_ev",
            "sigma_angstrom",
        }:
            raise ValueError("Each cross pair needs two symbols, epsilon_ev and sigma_angstrom")
        key = (pair["qm_symbol"], pair["environment_symbol"])
        if any(
            not isinstance(symbol, str) or re.fullmatch(r"[A-Z][a-z]?", symbol) is None
            for symbol in key
        ):
            raise ValueError("Cross symbols must be element names")
        if key in pairs:
            raise ValueError("Duplicate directional cross pair")
        pairs[key] = (pair["epsilon_ev"], pair["sigma_angstrom"])
    CrossLennardJones(pairs, source=cross["source"])
    return pairs


def _positive_real(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def load_additive_config(path):
    """Validate every Hamiltonian input and include it in the restart identity."""
    path = Path(path).resolve()
    raw = path.read_text()
    config = json.loads(raw)
    provider = config.get("provider") if isinstance(config, dict) else None
    state = provider.get("state") if isinstance(provider, dict) else None
    tda = isinstance(state, dict) and state.get("method") in ("rhf-tda", "pbe0-tda")
    if isinstance(config, dict) and (config.get("schema_version") == 2 or tda):
        # Preserve legacy ground parsing; new charge/state inputs must be unambiguous.
        config = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return _validate_config(config, path)


def _validate_config(config, path):
    """Validate a parsed request using its original file location for relative paths."""
    from copy import deepcopy

    config = deepcopy(config)
    path = Path(path).resolve()
    required = {
        "schema_version",
        "hamiltonian",
        "data_file",
        "model",
        "model_sha256",
        "type_symbols",
        "periodic",
        "qm_region",
        "environment_region",
        "environment_charge_spin",
        "qm",
        "embedding",
        "cross",
        "inference_profile",
        "timestep_fs",
        "steps",
    }
    optional = {"sample_interval", "temperature_K", "velocity_seed", "provider"}
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or set(config) - required - optional
    ):
        raise ValueError("Additive configuration has missing or unsupported fields")
    if type(config["schema_version"]) is not int or config["schema_version"] not in (1, 2):
        raise ValueError("schema_version must be 1 or 2")
    if config["hamiltonian"] != "finite-electrostatic-additive":
        raise ValueError("Require the finite-electrostatic-additive Hamiltonian")
    if config["periodic"] is not False:
        raise ValueError("Additive dynamics currently requires a nonperiodic fixed box")
    if config["inference_profile"] != RAW_PROFILE:
        raise ValueError(f"Only the explicit {RAW_PROFILE} inference profile is supported")
    for key in ("data_file", "model"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a file path")
        config[key] = str((path.parent / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}")
    if Path(config["model"]).suffix != ".pt":
        raise ValueError("The raw inference profile requires an original .pt checkpoint")
    if file_digest(config["model"]) != config["model_sha256"]:
        raise ValueError("Model SHA256 differs from its declared identity")
    config["data_sha256"] = file_digest(config["data_file"])
    symbols = config["type_symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(s, str) or re.fullmatch(r"[A-Z][a-z]?", s) is None for s in symbols)
    ):
        raise ValueError("type_symbols must contain element names")
    for key in ("qm_region", "environment_region"):
        validate_region(config[key])
        if not set(config[key]["symbols"]) <= set(symbols):
            raise ValueError(f"{key} symbols must be present in type_symbols")
    qm_region, env_region = config["qm_region"], config["environment_region"]
    if set(qm_region["atom_ids"]) & set(env_region["atom_ids"]):
        raise ValueError("QM and environment regions must be disjoint")
    state = config["environment_charge_spin"]
    if (
        not isinstance(state, list)
        or len(state) != 2
        or any(type(item) is not int for item in state)
        or state[1] < 1
    ):
        raise ValueError("environment_charge_spin requires integer charge and multiplicity")
    embedding = config["embedding"]
    if config["schema_version"] == 2:
        if (
            not isinstance(embedding, dict)
            or "charge_request" not in embedding
            or set(embedding) - {"charge_request", "radii_angstrom"}
        ):
            raise ValueError(
                "Schema 2 embedding requires only charge_request and optional radii_angstrom"
            )
        resolved = resolve_fixed_charges(
            embedding["charge_request"],
            env_region["atom_ids"],
            env_region["symbols"],
            base_directory=path.parent,
        )
        normalized = resolved["embedding"]
        if "radii_angstrom" in embedding:
            normalized["radii_angstrom"] = embedding["radii_angstrom"]
        config["charge_resolution"] = resolved["provenance"]
        embedding = config["embedding"] = normalized
    if (
        not isinstance(embedding, dict)
        or not {"atom_ids", "charges_e", "source"} <= set(embedding)
        or set(embedding) - {"atom_ids", "charges_e", "source", "radii_angstrom"}
    ):
        raise ValueError(
            "Embedding needs explicit atom IDs, charges and source, with optional radii"
        )
    charges = FixedEmbeddingCharges(**embedding)
    if charges.atom_ids != tuple(env_region["atom_ids"]):
        raise ValueError("Charge IDs must equal environment atom IDs in order")
    total_charge = (
        math.fsum(charges.charges_e)
        if config["schema_version"] == 2
        else float(sum(charges.charges_e))
    )
    if not math.isclose(total_charge, state[0], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Total fixed environment charge must match the DPA charge")
    pairs = cross_parameters(config["cross"])
    required_pairs = {(q, m) for q in qm_region["symbols"] for m in env_region["symbols"]}
    if set(pairs) != required_pairs:
        raise ValueError("Every cross element pair needs exactly one explicit parameter")
    config["provider"] = validate_provider(
        resolve_provider_reference(config.get("provider"), path), electrostatic=True)
    config["provider_contract"] = provider_contract(config["provider"], electrostatic=True)
    config["qm"] = validate_qm(config["qm"], electrostatic=True, provider=config["provider"])
    validate_provider_regions(
        config["provider"],
        qm_region,
        periodic=config["periodic"],
        environment_region=env_region,
        charges=charges,
    )
    config["steps"] = _positive_integer(config["steps"], "steps", allow_zero=True)
    config["sample_interval"] = _positive_integer(
        config.get("sample_interval", 1), "sample_interval"
    )
    config["timestep_fs"] = _positive_real(config["timestep_fs"], "timestep_fs")
    if ("temperature_K" in config) != ("velocity_seed" in config):
        raise ValueError("temperature_K and velocity_seed must be supplied together")
    if "temperature_K" in config:
        config["temperature_K"] = _positive_real(config["temperature_K"], "temperature_K")
        _positive_integer(config["velocity_seed"], "velocity_seed")
    config["software"] = software_identity(config["provider"])
    config["software"]["packages"]["torch"] = importlib.metadata.version("torch")
    try:
        config["software"]["packages"]["mpi4py"] = importlib.metadata.version("mpi4py")
    except importlib.metadata.PackageNotFoundError:
        config["software"]["packages"]["mpi4py"] = None
    physics = {
        key: value
        for key, value in config.items()
        if key
        not in {"model", "data_file", "steps", "sample_interval", "temperature_K", "velocity_seed"}
    }
    config["physics_sha256"] = hashlib.sha256(
        json.dumps(physics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config
