# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Strict finite two-partition ground-state dynamics and restart identities."""

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from .adaptive import ConservativeMechanicalMixer, MechanicalPartition, QuinticDistanceExchange
from .additive import AdditiveCalculator, CrossLennardJones
from .additive_config import cross_parameters
from .config import _positive_integer, file_digest, software_identity, validate_qm, validate_region
from .core import SubtractiveCalculator
from .embedding import FixedEmbeddingCharges
from .inference import RAW_PROFILE
from .molecular_adaptive import MolecularMechanicalMixer, QuinticMolecularWeights
from .molecular_additive import AdditivePartition, MolecularAdditiveMixer, subset_charges
from .providers import provider_contract, validate_provider
from .selection_config import FixedGroup, read_json
from .water_buffer import HAMILTONIAN as WATER_BUFFER
from .water_buffer import SWITCH as WATER_SWITCH
from .water_buffer import QuinticWaterBufferWeights, WaterBoundaryGuard, WaterBufferMechanicalMixer
from .water_reference import validate_reference

HAMILTONIAN = "finite-two-partition-mechanical"
MOLECULAR_HAMILTONIAN = "finite-molecular-mechanical"
MOLECULAR_ADDITIVE = "finite-molecular-additive"
MOLECULAR_SWITCH = "quintic-molecular-occupancy"
SWITCH_METHOD = "quintic-distance-exchange"


def _fields(value, required, optional=(), *, name):
    if (
        not isinstance(value, dict)
        or not set(required) <= set(value)
        or set(value) - set(required) - set(optional)
    ):
        raise ValueError(f"{name} has missing or unsupported fields")


def _list(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    return value


def _positive_real(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def molecular_environment_charges(config):
    """Keep each partition's environment as an exact subset of one fixed table."""
    charges = FixedEmbeddingCharges(**config["embedding"])
    groups = {g["group_id"]: g["atom_ids"] for g in config["groups"]}
    result = []
    for partition in config["partitions"]:
        try:
            qm_ids = {a for g in partition["qm_group_ids"] for a in groups[g]}
        except KeyError as exc:
            raise ValueError("Every QM group must be explicitly declared") from exc
        result.append(subset_charges(charges, sorted(set(charges.atom_ids) - qm_ids)))
    return tuple(result)


def build_mixer(config, low, high_providers):
    """Use one shared low provider and the existing complete energy derivative.

    The same neutral-singlet conditioning applies to full and isolated inputs.
    Providers receive sorted atom IDs and no per-atom model parameters. This
    function also checks configuration using metadata-only objects; validation
    never evaluates a provider or imports an electronic-structure engine.
    """
    providers = tuple(high_providers)
    if len(providers) != len(config["partitions"]):
        raise ValueError("One ground-state provider per declared partition is required")
    groups = tuple(FixedGroup(**group) for group in config["groups"])
    if config["hamiltonian"] == MOLECULAR_ADDITIVE:
        fields = molecular_environment_charges(config)
        full = tuple(sorted(a for group in groups for a in group.atom_ids))
        cross = CrossLennardJones(cross_parameters(config["cross"]), source=config["cross"]["source"])
        symbols = {a: s for group in groups for a, s in zip(group.atom_ids, group.symbols)}
        partitions = []
        required = set()
        for partition, high, field in zip(config["partitions"], providers, fields, strict=True):
            qm_ids = tuple(a for a in full if a not in field.atom_ids)
            required.update((symbols[q], symbols[m]) for q in qm_ids for m in field.atom_ids)
            partitions.append(AdditivePartition(
                calculator=AdditiveCalculator(low, high, cross, qm_atom_ids=qm_ids,
                                               environment_atom_ids=field.atom_ids),
                **partition))
        if set(cross.parameters) != required:
            raise ValueError("Every molecular cross element pair needs exactly one explicit parameter")
        return MolecularAdditiveMixer(
            groups=groups, core_group_ids=config["core_group_ids"], partitions=partitions,
            switch=QuinticMolecularWeights(**{k: v for k, v in config["switch"].items() if k != "method"}),
            charges=FixedEmbeddingCharges(**config["embedding"]),
        )
    specifications = config["partitions"]
    water = config["hamiltonian"] == WATER_BUFFER
    if water:
        _, delta, source = validate_reference(
            config["buffer_reference"], model_sha256=config["model_sha256"],
            qm=config["qm"], provider=config["provider"])
        specifications = []
        for raw in config["partitions"]:
            partition = dict(raw)
            count = len(set(partition["qm_group_ids"]) - set(config["core_group_ids"]))
            expected = -count*delta
            if (("energy_offset_ev" in partition and partition["energy_offset_ev"] != expected)
                    or ("offset_source" in partition and partition["offset_source"] != source)):
                raise ValueError("Declared partition offset differs from the water reference")
            partition.update(energy_offset_ev=expected, offset_source=source)
            specifications.append(partition)
    partitions = tuple(
        MechanicalPartition(calculator=SubtractiveCalculator(low, high), **partition)
        for partition, high in zip(specifications, providers, strict=True)
    )
    switch = {key: value for key, value in config["switch"].items() if key != "method"}
    if water:
        return WaterBufferMechanicalMixer(
            low, groups=groups, core_group_ids=config["core_group_ids"],
            partitions=partitions, switch=QuinticWaterBufferWeights(**switch),
            guard=WaterBoundaryGuard(**config["water_guard"]),
            reference_delta_ev=delta, reference_source=source)
    if config["hamiltonian"] == MOLECULAR_HAMILTONIAN:
        return MolecularMechanicalMixer(
            low, groups=groups, core_group_ids=config["core_group_ids"],
            partitions=partitions, switch=QuinticMolecularWeights(**switch),
        )
    return ConservativeMechanicalMixer(
        low, groups=groups, partitions=partitions, switch=QuinticDistanceExchange(**switch)
    )


def load_adaptive_config(path):
    """Reject unsupported physics before constructing a scientific backend."""
    path = Path(path).resolve()
    return _validate_config(read_json(path.read_text()), path)


def _validate_config(config, path):
    """Validate the same explicit route for file and unified selector inputs."""
    path = Path(path).resolve()
    additive = config.get("hamiltonian") == MOLECULAR_ADDITIVE if isinstance(config, dict) else False
    water = isinstance(config, dict) and config.get("hamiltonian") == WATER_BUFFER
    molecular = isinstance(config, dict) and config.get("hamiltonian") in (
        MOLECULAR_HAMILTONIAN, MOLECULAR_ADDITIVE, WATER_BUFFER)
    _fields(
        config,
        (
            "schema_version",
            "hamiltonian",
            "data_file",
            "model",
            "model_sha256",
            "type_symbols",
            "periodic",
            "full_region",
            "groups",
            "partitions",
            "switch",
            "provider",
            "qm",
            "inference_profile",
            "timestep_fs",
            "steps",
        ) + (("core_group_ids",) if molecular else ())
          + (("buffer_reference", "water_guard") if water else ())
          + (("environment_charge_spin", "embedding", "cross") if additive
             else ("full_charge_spin", "region_charge_spin")),
        ("sample_interval", "temperature_K", "velocity_seed"),
        name="Adaptive configuration",
    )
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if config["hamiltonian"] not in (HAMILTONIAN, MOLECULAR_HAMILTONIAN, MOLECULAR_ADDITIVE, WATER_BUFFER):
        raise ValueError("Unsupported adaptive Hamiltonian")
    if config["periodic"] is not False:
        raise ValueError("Adaptive molecular dynamics requires a nonperiodic fixed box")
    if config["inference_profile"] != RAW_PROFILE:
        raise ValueError(f"inference_profile must be {RAW_PROFILE}")
    for name in (("environment_charge_spin",) if additive else ("full_charge_spin", "region_charge_spin")):
        state = config[name]
        if not isinstance(state, list) or state != [0, 1] or any(type(x) is not int for x in state):
            raise ValueError(f"{name} must explicitly be the supported neutral singlet [0, 1]")
    if not isinstance(config["provider"], dict):
        raise TypeError("provider must explicitly declare engine, device and state")
    config["provider"] = validate_provider(config["provider"], electrostatic=additive)
    if config["provider"]["engine"] != "pyscf":
        raise ValueError("Adaptive molecular dynamics currently requires the CPU PySCF provider")
    if config["provider"]["state"]["method"] != "ground":
        raise ValueError("Adaptive molecular dynamics requires the ground electronic state")
    config["provider_contract"] = provider_contract(config["provider"], electrostatic=additive)
    if not isinstance(config["qm"], dict) or not {"charge", "multiplicity"} <= set(config["qm"]):
        raise ValueError("qm must explicitly declare charge and multiplicity")
    if "nlc_grid_level" in config["qm"]:
        raise ValueError("Adaptive molecular dynamics does not support an NLC grid setting")
    qm = config["qm"] = validate_qm(config["qm"], electrostatic=additive, provider=config["provider"])
    if [qm["charge"], qm["multiplicity"]] != [0, 1] or qm["grid_response"] is not True:
        raise ValueError("QM requires a neutral singlet and the complete integration-grid response")
    symbols = _list(config["type_symbols"], "type_symbols")
    if any(not isinstance(s, str) or re.fullmatch(r"[A-Z][a-z]?", s) is None for s in symbols):
        raise ValueError("type_symbols must contain element names")
    full = config["full_region"]
    _fields(full, ("atom_ids", "symbols", "molecules"), name="full_region")
    _list(full["atom_ids"], "full_region atom_ids")
    _list(full["symbols"], "full_region symbols")
    validate_region(full)
    if not set(full["symbols"]) <= set(symbols):
        raise ValueError("Every full_region symbol must occur in type_symbols")
    for group in _list(config["groups"], "groups"):
        _fields(group, ("group_id", "atom_ids", "symbols"), name="Group")
        _list(group["atom_ids"], "Group atom_ids")
        _list(group["symbols"], "Group symbols")
    partitions = _list(config["partitions"], "partitions")
    if not molecular and len(partitions) != 2:
        raise ValueError("Exactly two partitions are required")
    for partition in partitions:
        _fields(
            partition,
            (
                "partition_id",
                "qm_group_ids",
                "qm_charge",
                "qm_multiplicity",
                "method_identity",
                "energy_reference",
                "electronic_state",
            ),
            ("energy_offset_ev", "offset_source"),
            name="Partition",
        )
        _list(partition["qm_group_ids"], "qm_group_ids")
    switch = config["switch"]
    if molecular:
        _list(config["core_group_ids"], "core_group_ids")
        _fields(switch, ("method", "center_atom_id", "candidate_anchors",
                        "inner_radius_angstrom", "outer_radius_angstrom")
                + (() if water else ("occupancy",)),
                ("max_partitions",), name="Molecular switch")
        _list(switch["candidate_anchors"], "candidate_anchors")
        expected_switch = WATER_SWITCH if water else MOLECULAR_SWITCH
        if switch["method"] != expected_switch:
            raise ValueError(f"switch.method must be {expected_switch}")
        if water:
            _fields(config["water_guard"], ("min_oh_angstrom", "max_oh_angstrom",
                                           "min_foreign_oh_angstrom"), name="Water guard")
            config["buffer_reference"], _, _ = validate_reference(
                config["buffer_reference"], model_sha256=config["model_sha256"],
                qm=qm, provider=config["provider"])
    else:
        _fields(
            switch,
            ("method", "center_atom_id", "first_atom_id", "second_atom_id", "half_width_angstrom"),
            name="Switch",
        )
        if switch["method"] != SWITCH_METHOD:
            raise ValueError(f"switch.method must be {SWITCH_METHOD}")
    # Metadata-only construction reuses every mixer state/group/family check.
    # No evaluate method or engine construction is involved in this validation.
    state = {"charge": qm["charge"], "multiplicity": qm["multiplicity"],
             "xc": qm["xc"], "basis": qm["basis"]}
    if additive:
        _fields(config["embedding"], ("atom_ids", "charges_e", "source"),
                ("radii_angstrom",), name="Full-system embedding table")
        charges = FixedEmbeddingCharges(**config["embedding"])
        if set(charges.atom_ids) != set(full["atom_ids"]):
            raise ValueError("Embedding table must cover every full_region atom exactly")
        charges = subset_charges(charges, sorted(charges.atom_ids))
        config["embedding"] = {
            "atom_ids": list(charges.atom_ids), "charges_e": charges.charges_e.tolist(),
            "source": charges.source,
        }
        if charges.radii_angstrom is not None:
            config["embedding"]["radii_angstrom"] = charges.radii_angstrom.tolist()
        high = tuple(SimpleNamespace(**state, charges=field)
                     for field in molecular_environment_charges(config))
        low_state = config["environment_charge_spin"]
    else:
        high = (SimpleNamespace(**state),) * len(partitions)
        low_state = config["region_charge_spin"]
    validated = build_mixer(config, SimpleNamespace(charge_spin=low_state), high)
    if water:
        config["water_guard"] = vars(validated.guard).copy()
    mapping = sorted(
        (atom, symbol)
        for group in validated.groups
        for atom, symbol in zip(group.atom_ids, group.symbols, strict=True)
    )
    if mapping != sorted(zip(full["atom_ids"], full["symbols"], strict=True)):
        raise ValueError("Groups must exactly cover every full_region atom ID and element")
    if sorted(tuple(sorted(group)) for group in full["molecules"]) != sorted(
        group.atom_ids for group in validated.groups
    ):
        raise ValueError("full_region molecules must match the declared complete groups")
    config["full_region"] = {
        "atom_ids": [atom for atom, _ in mapping],
        "symbols": [symbol for _, symbol in mapping],
        "molecules": [
            list(group) for group in sorted(group.atom_ids for group in validated.groups)
        ],
    }
    config["groups"] = [
        {
            "group_id": group.group_id,
            "atom_ids": list(group.atom_ids),
            "symbols": list(group.symbols),
        }
        for group in validated.groups
    ]
    config["partitions"] = [
        {
            "partition_id": p.partition_id,
            "qm_group_ids": list(p.qm_group_ids),
            "qm_charge": p.qm_charge,
            "qm_multiplicity": p.qm_multiplicity,
            "method_identity": p.method_identity,
            "energy_reference": p.energy_reference,
            "electronic_state": p.electronic_state,
            "energy_offset_ev": p.energy_offset_ev,
            "offset_source": p.offset_source,
        }
        for p in validated.partitions
    ]
    if molecular:
        config["core_group_ids"] = list(validated.core_group_ids)
        config["switch"] = {
            "method": WATER_SWITCH if water else MOLECULAR_SWITCH,
            "center_atom_id": validated.switch.center_atom_id,
            "candidate_anchors": [list(x) for x in validated.switch.candidate_anchors],
            **({} if water else {"occupancy": validated.switch.occupancy}),
            "inner_radius_angstrom": validated.switch.inner_radius_angstrom,
            "outer_radius_angstrom": validated.switch.outer_radius_angstrom,
            "max_partitions": validated.switch.max_partitions,
        }
    else:
        config["switch"] = {
            "method": SWITCH_METHOD,
            "center_atom_id": validated.switch.center_atom_id,
            "first_atom_id": validated.switch.first_atom_id,
            "second_atom_id": validated.switch.second_atom_id,
            "half_width_angstrom": validated.switch.half_width_angstrom,
        }
    for name in ("model", "data_file"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"{name} must be a file path")
        config[name] = str((path.parent / config[name]).resolve())
        if not Path(config[name]).is_file():
            raise ValueError(f"Missing {name}")
    if Path(config["model"]).suffix != ".pt":
        raise ValueError("The raw profile requires an original .pt model")
    expected = config["model_sha256"]
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("model_sha256 must be a complete lowercase SHA256 digest")
    if file_digest(config["model"]) != expected:
        raise ValueError("Model SHA256 differs from the declared identity")
    config["data_sha256"] = file_digest(config["data_file"])
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
    config["adaptive_contract"] = {
        "schema_version": 1,
         "energy": ("weighted-complete-additive-partition-totals" if additive
                   else "shared-full-baseline-plus-weighted-corrections"),
        "weight_derivatives": "all-Cartesian",
        "history": "none",
        "force_scope": "all-atoms",
        "native_pair": "zero",
        "virial": "unsupported",
        "proton_reassignment": "unsupported",
    }
    config["software"] = software_identity()
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
        json.dumps(physics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return config
