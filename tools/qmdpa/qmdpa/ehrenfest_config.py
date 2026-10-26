# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit fixed-QM, finite, CPU two-state Ehrenfest run admission."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .casscf_pair import CASSCFPairReference, CASSCFPairSettings
from .charges import _invalid_constant, _unique_object
from .config import _validate_config as validate_common
from .config import file_digest, validate_region
from .ehrenfest_mechanical import HAMILTONIAN, _complex_from_record, _complex_record
from .inference import RAW_PROFILE


def load_ehrenfest_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return _validate_config(raw, path)


def _validate_config(config, path):
    if not isinstance(config, dict):
        raise TypeError("Ehrenfest configuration must be an object")
    config = copy.deepcopy(config)
    extra = {"full_region", "qm_region", "pair_reference", "initial_amplitudes", "inference_profile"}
    if not extra <= set(config) or "region" in config:
        raise ValueError("Ehrenfest requires explicit full/QM regions, pair reference and amplitudes")
    if config.get("hamiltonian") != HAMILTONIAN or config.get("periodic") is not False:
        raise ValueError("Two-state Ehrenfest requires its finite mechanical Hamiltonian")
    provider = {"engine": "pyscf", "device": "cpu", "state": {"method": "sa-casscf22"}}
    if config.get("provider") != provider:
        raise ValueError("Ehrenfest requires the explicit CPU SA-CASSCF(2,2) state pair")
    if config["inference_profile"] != RAW_PROFILE or Path(config["model"]).suffix != ".pt":
        raise ValueError("Ehrenfest requires deterministic raw .pt DPA inference")
    specification = config["pair_reference"]
    if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
        raise ValueError("pair_reference requires an immutable file path and SHA256")
    reference_path = (Path(path).parent / specification["path"]).resolve()
    if file_digest(reference_path) != specification["sha256"]:
        raise ValueError("CASSCF pair reference file identity changed")
    reference = CASSCFPairReference.from_record(json.loads(
        reference_path.read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant))
    settings = CASSCFPairSettings(**config["qm"])
    if settings != reference.settings:
        raise ValueError("QM controls differ from the prepared CASSCF pair reference")
    for name in ("qm_region", "full_region"):
        validate_region(config[name])
    qm, full = config["qm_region"], config["full_region"]
    identities = dict(zip(full["atom_ids"], full["symbols"]))
    if any(identities.get(atom) != symbol for atom, symbol in zip(qm["atom_ids"], qm["symbols"])):
        raise ValueError("QM atoms/elements must be members of the full region")
    if not set(map(frozenset, qm["molecules"])) <= set(map(frozenset, full["molecules"])):
        raise ValueError("QM membership must select complete declared molecules")
    geometry = reference.wavefunction.geometry
    if geometry.atom_ids != tuple(qm["atom_ids"]) or geometry.symbols != tuple(qm["symbols"]):
        raise ValueError("Prepared CASSCF reference atoms differ from the QM region")
    coefficients = _complex_from_record(config["initial_amplitudes"])
    if ("temperature_K" in config) != ("velocity_seed" in config):
        raise ValueError("Initial temperature and seed must be supplied together")
    common = {k: v for k, v in config.items() if k not in extra}
    common.update(hamiltonian="fixed-region-mechanical", region=qm,
                  provider={"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}},
                  qm={"xc": "hf", "basis": settings.basis, "charge": 0, "multiplicity": 1})
    normalized = validate_common(common, path)
    normalized.pop("region")
    if not set(full["symbols"]) <= set(normalized["type_symbols"]):
        raise ValueError("Full-region elements are absent from native type_symbols")
    for name in ("full_charge_spin", "region_charge_spin"):
        if normalized.get(name) != [0, 1]:
            raise ValueError("Ehrenfest requires explicit neutral singlet DPA conditions")
    normalized.update(hamiltonian=HAMILTONIAN, provider=provider, qm=asdict(settings),
                      full_region=full, qm_region=qm, pair_reference=reference.record(),
                      pair_reference_file_sha256=specification["sha256"],
                      initial_amplitudes=_complex_record(coefficients), inference_profile=RAW_PROFILE,
                      provider_contract={"engine": "pyscf", "device": "cpu", "states": 2,
                                         "method": "equal-weight-sa-casscf22-neutral-singlets",
                                         "boundary": "finite", "embedding": "mechanical",
                                         "derivative_coupling": "d01=<0|grad 1>; no ETF",
                                         "nuclear_algorithm": "native-velocity-verlet",
                                         "electronic_algorithm": "endpoint-averaged-unitary",
                                         "force_coordinates": ["qm", "environment"]})
    physics = {k: v for k, v in normalized.items() if k not in {
        "physics_sha256", "model", "data_file", "steps", "sample_interval", "temperature_K", "velocity_seed"}}
    normalized["physics_sha256"] = hashlib.sha256(json.dumps(
        physics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return normalized
