# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Strict configuration for the distinct finite gap-anchor native candidate."""

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

from .additive_config import _positive_real
from .charges import _invalid_constant, _unique_object
from .config import _positive_integer, file_digest, software_identity, validate_qm, validate_region
from .embedding import FixedEmbeddingCharges
from .fragment_response_config import FRAGMENT_GAP_HAMILTONIAN, validate_fragment_response
from .gap_anchor import GapAnchorTerms
from .interaction_delta_config import CORRECTED_GAP_HAMILTONIAN, validate_interaction_delta
from .providers import provider_contract, validate_provider, validate_provider_regions
from .state_inputs import resolve_provider_reference


def load_gap_anchor_config(path):
    """Pin every physical input and require an explicit supported neutral-singlet route."""
    path = Path(path).resolve()
    config = json.loads(
        path.read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
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
        "embedding",
        "full_charge_spin",
        "region_charge_spin",
        "qm",
        "provider",
        "timestep_fs",
        "steps",
    }
    optional = {
        "sample_interval",
        "temperature_K",
        "velocity_seed",
        "interaction_delta",
        "fragment_response",
        "reuse_ground_terms",
        "reduce_fragment_terms",
    }
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or set(config) - required - optional
    ):
        raise ValueError("Gap-anchor configuration has missing or unsupported fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Gap-anchor schema_version must be 1")
    if config["hamiltonian"] not in (
        GapAnchorTerms.hamiltonian,
        CORRECTED_GAP_HAMILTONIAN,
        FRAGMENT_GAP_HAMILTONIAN,
    ):
        raise ValueError("Require an explicit supported gap-anchor Hamiltonian")
    corrected = config["hamiltonian"] == CORRECTED_GAP_HAMILTONIAN
    if corrected != ("interaction_delta" in config):
        raise ValueError(
            "The corrected Hamiltonian and interaction_delta must be selected together"
        )
    fragment_response = config["hamiltonian"] == FRAGMENT_GAP_HAMILTONIAN
    if fragment_response != ("fragment_response" in config):
        raise ValueError(
            "Fragment-response Hamiltonian and configuration must be selected together"
        )
    if "reduce_fragment_terms" in config:
        if (
            type(config["reduce_fragment_terms"]) is not bool
            or not fragment_response
            or (config["reduce_fragment_terms"] and config.get("reuse_ground_terms") is not True)
        ):
            raise ValueError(
                "reduce_fragment_terms requires explicit fragment response and exact ground reuse"
            )
    if config["periodic"] is not False:
        raise ValueError("Gap-anchor dynamics requires nonperiodic boundaries")
    for key in ("data_file", "model"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be an explicit file path")
        config[key] = str((path.parent / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}")
    if Path(config["model"]).suffix != ".pt2":
        raise ValueError("This gap-anchor driver requires an explicit compiled .pt2 model")
    if file_digest(config["model"]) != config["model_sha256"]:
        raise ValueError("Model SHA256 differs from its declared identity")
    config["data_sha256"] = file_digest(config["data_file"])
    symbols = config["type_symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(s, str) or re.fullmatch(r"[A-Z][a-z]?", s) is None for s in symbols)
    ):
        raise ValueError("type_symbols must contain explicit element names")
    for key in ("qm_region", "environment_region"):
        validate_region(config[key])
        if not set(config[key]["symbols"]) <= set(symbols):
            raise ValueError(f"{key} symbols must be present in type_symbols")
    qm, env = config["qm_region"], config["environment_region"]
    if set(qm["atom_ids"]) & set(env["atom_ids"]):
        raise ValueError("QM and environment regions must be disjoint")
    for key in ("full_charge_spin", "region_charge_spin"):
        value = config[key]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(v) is not int for v in value)
            or value != [0, 1]
        ):
            raise ValueError(f"{key} requires the explicit neutral-singlet condition [0,1]")
    embedding = config["embedding"]
    if (
        not isinstance(embedding, dict)
        or not {"atom_ids", "charges_e", "source"} <= set(embedding)
        or set(embedding) - {"atom_ids", "charges_e", "source", "radii_angstrom"}
    ):
        raise ValueError("Embedding requires explicit fixed charges, IDs and provenance")
    charges = FixedEmbeddingCharges(**embedding)
    if charges.atom_ids != tuple(env["atom_ids"]) or charges.radii_angstrom is not None:
        raise ValueError("Point-charge IDs must follow the environment region order")
    if not math.isclose(math.fsum(charges.charges_e), 0.0, abs_tol=1e-8, rel_tol=0):
        raise ValueError(
            "This neutral full-system route requires a neutral fixed-charge environment"
        )
    config["provider"] = validate_provider(
        resolve_provider_reference(config["provider"], path), electrostatic=True
    )
    engine = config["provider"]["engine"]
    method = config["provider"]["state"]["method"]
    if engine == "gpu4pyscf" and (
        method != "rhf-tda" or corrected or fragment_response or config["qm"].get("charge", 0) != 0
    ):
        raise ValueError(
            "GPU supports the plain neutral-singlet RHF gap-anchor; fragment corrections require the matched CPU provider"
        )
    if method not in ("rhf-tda", "pbe0-tda"):
        raise ValueError("Gap-anchor requires a pinned RHF/TDA or PBE0/TDA selected state")
    if "reuse_ground_terms" in config:
        if type(config["reuse_ground_terms"]) is not bool or method != "pbe0-tda":
            raise ValueError(
                "reuse_ground_terms requires an explicit boolean and the matched PBE0 route"
            )
    if corrected:
        config["interaction_delta"] = validate_interaction_delta(
            config["interaction_delta"], qm, env, method
        )
    config["provider_contract"] = provider_contract(config["provider"], electrostatic=True)
    config["qm"] = validate_qm(config["qm"], electrostatic=True, provider=config["provider"])
    validate_provider_regions(
        config["provider"], qm, periodic=False, environment_region=env, charges=charges
    )
    if fragment_response:
        config["fragment_response"] = validate_fragment_response(
            config["fragment_response"], qm, env, config["provider"], charges, config_path=path
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
        config["velocity_seed"] = _positive_integer(config["velocity_seed"], "velocity_seed")
    config["assembly"] = {
        "pair_style": "zero",
        "external_quantity": "total-energy-and-all-forces",
        "model_format": "pt2",
        "ground_method": (
            "rhf"
            if engine == "gpu4pyscf"
            else {"rhf-tda": "rks-xc-hf", "pbe0-tda": "rks-xc-pbe0"}[method]
        ),
    }
    if corrected:
        config["assembly"]["interaction_delta"] = config["interaction_delta"]["model"]
    if fragment_response:
        config["assembly"]["fragment_response"] = config["fragment_response"]["model"]
        config["assembly"]["selected_state_terms"] = ["anchor", "cluster_pair", "cluster_field"]
    if config.get("reuse_ground_terms", False):
        config["assembly"]["ground_result_reuse"] = "last-successful-exact-input"
    if config.get("reduce_fragment_terms", False):
        config["assembly"]["fragment_evaluation"] = "identical-signed-term-reduction"
    config["software"] = software_identity(config["provider"])
    for name in ("torch", "mpi4py"):
        try:
            config["software"]["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "torch":
                raise
            config["software"]["packages"][name] = None
    physics = {
        k: v
        for k, v in config.items()
        if k
        not in {"model", "data_file", "steps", "sample_interval", "temperature_K", "velocity_seed"}
    }
    config["physics_sha256"] = hashlib.sha256(
        json.dumps(physics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config
