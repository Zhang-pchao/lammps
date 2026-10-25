# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Finite ground reference-subtractive configuration and restart identity."""

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from .charges import _invalid_constant, _unique_object
from .config import _validate_config as validate_common
from .config import validate_qm, validate_region
from .embedding import FixedEmbeddingCharges
from .inference import RAW_PROFILE
from .providers import provider_contract, validate_provider_regions
from .reference import FixedChargeCrossCoulomb

HAMILTONIAN = "finite-electrostatic-reference-subtractive"


def load_reference_config(path):
    path = Path(path).resolve()
    return _validate_config(
        json.loads(
            path.read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        ),
        path,
    )


def _validate_config(config, path):
    """Reuse common file, region, QM and dynamics checks before binding the field."""
    if not isinstance(config, dict):
        raise TypeError("Reference configuration must be a JSON object")
    config = copy.deepcopy(config)
    extra = {"qm_region", "environment_region", "embedding", "reference", "inference_profile"}
    if not extra <= set(config) or "region" in config:
        raise ValueError(
            "Reference configuration requires QM/environment regions, embedding and reference"
        )
    if config.get("hamiltonian") != HAMILTONIAN:
        raise ValueError("Require the finite-electrostatic-reference-subtractive Hamiltonian")
    if config.get("periodic") is not False:
        raise ValueError("Reference dynamics requires finite nonperiodic boundaries")
    if config["inference_profile"] != RAW_PROFILE or Path(config["model"]).suffix != ".pt":
        raise ValueError("Reference dynamics requires the explicit deterministic raw .pt profile")
    if ("temperature_K" in config) != ("velocity_seed" in config):
        raise ValueError("temperature_K and velocity_seed must be supplied together")
    provider = config.get(
        "provider", {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}
    )
    if provider != {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}:
        raise ValueError("Reference dynamics currently requires CPU PySCF ground state")
    common = {k: v for k, v in config.items() if k not in extra}
    common.update(hamiltonian="fixed-region-mechanical", region=config["qm_region"])
    normalized = validate_common(common, path)
    normalized.pop("region")
    normalized["qm"] = validate_qm(config["qm"], electrostatic=True, provider=provider)
    normalized.update({k: config[k] for k in extra})
    normalized["hamiltonian"] = HAMILTONIAN
    qm, env = normalized["qm_region"], normalized["environment_region"]
    validate_region(env)
    if set(qm["atom_ids"]) & set(env["atom_ids"]):
        raise ValueError("QM and environment IDs must be disjoint")
    if not set(qm["symbols"] + env["symbols"]) <= set(normalized["type_symbols"]):
        raise ValueError("Every region element must occur in type_symbols")
    for key in ("full_charge_spin", "region_charge_spin"):
        if normalized.get(key) is None:
            raise ValueError("Reference dynamics requires explicit full and region charge_spin")
    embedding = normalized["embedding"]
    if not isinstance(embedding, dict) or set(embedding) != {"atom_ids", "charges_e", "source"}:
        raise ValueError("Reference embedding requires explicit point charges and source")
    charges = FixedEmbeddingCharges(**embedding)
    if charges.atom_ids != tuple(env["atom_ids"]):
        raise ValueError("Embedding charge order must match environment IDs")
    reference = normalized["reference"]
    if (
        not isinstance(reference, dict)
        or set(reference) != {"form", "qm_charges"}
        or reference["form"] != "fixed-point-charge-cross-coulomb"
    ):
        raise ValueError("Reference requires an explicit fixed-point-charge-cross-coulomb model")
    qspec = reference["qm_charges"]
    if not isinstance(qspec, dict) or set(qspec) != {"atom_ids", "charges_e", "source"}:
        raise ValueError("Reference QM point charges require IDs, values and source")
    qcharges = FixedEmbeddingCharges(**qspec)
    if qcharges.atom_ids != tuple(qm["atom_ids"]):
        raise ValueError("Reference charge order must match QM IDs")
    FixedChargeCrossCoulomb(qcharges, charges)
    if not math.isclose(
        math.fsum(qcharges.charges_e), normalized["qm"]["charge"], rel_tol=0, abs_tol=1e-8
    ):
        raise ValueError("Reference QM charges must sum to the QM total charge")
    if not math.isclose(
        math.fsum(charges.charges_e) + normalized["qm"]["charge"],
        normalized["full_charge_spin"][0],
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError("Embedding and QM charges must sum to the full DPA charge")
    normalized["provider_contract"] = provider_contract(provider, electrostatic=True)
    validate_provider_regions(provider, qm, periodic=False, environment_region=env, charges=charges)
    for package in ("torch", "mpi4py"):
        try:
            normalized["software"]["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            normalized["software"]["packages"][package] = None
    physics = {
        k: v
        for k, v in normalized.items()
        if k
        not in {
            "model",
            "data_file",
            "steps",
            "sample_interval",
            "temperature_K",
            "velocity_seed",
            "physics_sha256",
        }
    }
    normalized["physics_sha256"] = hashlib.sha256(
        json.dumps(physics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return normalized
