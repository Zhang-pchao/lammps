# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Admission and exact restart identity for the bounded periodic EE route."""

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .abacus import validate_abacus_settings
from .charges import _invalid_constant, _unique_object
from .config import _validate_config as validate_common
from .config import validate_region
from .embedding import FixedEmbeddingCharges
from .inference import RAW_PROFILE
from .periodic_field import PeriodicGaussianField, PeriodicGaussianReference

HAMILTONIAN = "periodic-electrostatic-reference-subtractive"


def load_periodic_ee_config(path):
    path = Path(path).resolve()
    return _validate_config(json.loads(path.read_text(), object_pairs_hook=_unique_object,
                                       parse_constant=_invalid_constant), path)


def _validate_config(config, path):
    if not isinstance(config, dict):
        raise TypeError("Periodic EE configuration must be a JSON object")
    config = copy.deepcopy(config)
    extra = {"qm_region", "environment_region", "reference", "inference_profile"}
    if not extra <= set(config) or "region" in config:
        raise ValueError("Periodic EE requires explicit QM/environment regions and reference")
    if config.get("hamiltonian") != HAMILTONIAN or config.get("periodic") is not True:
        raise ValueError("Periodic EE requires its explicit Hamiltonian and full periodic boundaries")
    provider = {"engine": "abacus", "device": "cpu", "state": {"method": "ground"}}
    if config.get("provider") != provider:
        raise ValueError("Periodic EE currently requires CPU ABACUS ground state")
    if config["inference_profile"] != RAW_PROFILE or Path(config["model"]).suffix != ".pt":
        raise ValueError("Periodic EE requires explicit deterministic raw .pt inference")
    if ("temperature_K" in config) != ("velocity_seed" in config):
        raise ValueError("temperature_K and velocity_seed must be supplied together")
    qm_settings = validate_abacus_settings(config["qm"], Path(path).parent)
    if "periodic_field" not in qm_settings:
        raise ValueError("Periodic EE requires an explicit periodic_field")
    common = {k: v for k, v in config.items() if k not in extra}
    common.update(hamiltonian="fixed-region-mechanical", region=config["qm_region"],
                  qm={k: v for k, v in qm_settings.items() if k != "periodic_field"})
    normalized = validate_common(common, path)
    normalized.pop("region")
    normalized.update({k: config[k] for k in extra})
    normalized.update(hamiltonian=HAMILTONIAN, qm=qm_settings)
    qm, env = normalized["qm_region"], normalized["environment_region"]
    validate_region(env)
    if set(qm["atom_ids"]) & set(env["atom_ids"]):
        raise ValueError("QM and environment IDs must be disjoint")
    if not set(qm["symbols"] + env["symbols"]) <= set(normalized["type_symbols"]):
        raise ValueError("Every region element must occur in type_symbols")
    for key in ("full_charge_spin", "region_charge_spin"):
        if normalized.get(key) != [0, 1]:
            raise ValueError("Periodic EE requires explicit neutral singlet DPA states")
    field = PeriodicGaussianField(**qm_settings["periodic_field"])
    if field.atom_ids != tuple(env["atom_ids"]):
        raise ValueError("Field charge order must match environment IDs")
    reference = normalized["reference"]
    if (not isinstance(reference, dict) or set(reference) != {"form", "qm_charges"}
            or reference["form"] != PeriodicGaussianReference.model):
        raise ValueError("Periodic EE requires a periodic-gaussian-charge-cross-reference")
    spec = reference["qm_charges"]
    if not isinstance(spec, dict) or set(spec) != {"atom_ids", "charges_e", "source"}:
        raise ValueError("QM reference point charges require IDs, values and provenance")
    charges = FixedEmbeddingCharges(**spec)
    if charges.atom_ids != tuple(qm["atom_ids"]):
        raise ValueError("Reference charge order must match QM IDs")
    PeriodicGaussianReference(charges, field)
    normalized["provider_contract"].update(
        environment="neutral-fixed-gaussian-3d-periodic",
        force_coordinates=["qm", "environment"],
        zero_mode="omitted-neutral-3d-mean-zero",
        environment_self_and_pair="excluded",
    )
    for package in ("torch", "mpi4py"):
        try:
            normalized["software"]["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            normalized["software"]["packages"][package] = None
    physics = {k: v for k, v in normalized.items() if k not in {
        "model", "data_file", "steps", "sample_interval", "temperature_K", "velocity_seed",
        "physics_sha256",
    }}
    normalized["physics_sha256"] = hashlib.sha256(
        json.dumps(physics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return normalized
