#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit finite-water reassignment configuration and integer-state records."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .config import _validate_config as validate_base
from .config import validate_qm, validate_region
from .core import Geometry, SubtractiveCalculator
from .dynamics import extend_identity
from .inference import COMPRESSED_PROFILE, RAW_PROFILE
from .reactive_mechanical import ReactiveWaterGuard, ReactiveWaterMechanicalMixer
from .reactive_water import (
    ADDITIVE_HAMILTONIAN, HAMILTONIAN, REACTIVE_HAMILTONIANS, ProtonChannel, ReactiveWaterFamily,
)
from .selection_config import FixedGroup, read_json
from .water_reference import reference_identity, validate_reference


def build_family(config):
    settings = config["reactive_region"]
    return ReactiveWaterFamily(
        groups=tuple(FixedGroup(**g) for g in config["groups"]),
        **{key: value for key, value in settings.items() if key != "channels"},
        channels=tuple(ProtonChannel(**channel) for channel in settings["channels"]))


def build_mixer(config, low, high):
    reference, delta, identity = validate_reference(
        config["buffer_reference"], model_sha256=config["model_sha256"],
        qm=config["qm"], provider=config["provider"], inference_profile=config["inference_profile"])
    if config["hamiltonian"] == ADDITIVE_HAMILTONIAN:
        if (config["inference_profile"] == COMPRESSED_PROFILE
                and getattr(low, "pair_style", None) != reference["low_backend"]):
            raise ValueError("The environment provider differs from the calibrated native backend")
        from .additive import CrossLennardJones
        from .additive_config import cross_parameters
        from .embedding import FixedEmbeddingCharges
        from .reactive_additive import ReactiveWaterAdditiveMixer

        return ReactiveWaterAdditiveMixer(
            low, high, CrossLennardJones(cross_parameters(config["cross"]), source=config["cross"]["source"]),
            family=build_family(config), guard=ReactiveWaterGuard(**config["water_guard"]),
            charges=FixedEmbeddingCharges(**config["embedding"]), reference_delta_ev=delta,
            reference_source=identity, method_identity="neutral-ground:"+reference_identity(reference),
            energy_reference="isolated-water-difference")
    return ReactiveWaterMechanicalMixer(
        low, SubtractiveCalculator(low, high), family=build_family(config),
        guard=ReactiveWaterGuard(**config["water_guard"]), reference_delta_ev=delta,
        reference_source=identity, method_identity="neutral-ground:"+reference_identity(reference),
        energy_reference="isolated-water-difference")


def load_reactive_config(path):
    path = Path(path).resolve()
    return _validate_config(read_json(path.read_text()), path)


def _validate_config(raw, path):
    if not isinstance(raw, dict) or raw.get("hamiltonian") not in REACTIVE_HAMILTONIANS:
        raise ValueError("Reactive water requires its explicitly selected Hamiltonian")
    config = deepcopy(raw)
    hamiltonian = config["hamiltonian"]
    additive = hamiltonian == ADDITIVE_HAMILTONIAN
    ee = {}
    if additive:
        fields = {"environment_charge_spin", "embedding", "cross"}
        if not fields <= set(config) or {"full_charge_spin", "region_charge_spin"} & set(config):
            raise ValueError("Reactive EE requires environment_charge_spin, embedding and cross")
        ee = {key: config.pop(key) for key in fields}
        config["full_charge_spin"] = config["region_charge_spin"] = ee["environment_charge_spin"]
    extras = {"full_region", "groups", "reactive_region", "water_guard", "buffer_reference", "inference_profile"}
    if not extras <= set(config) or "region" in config:
        raise ValueError("Reactive water requires full_region, groups, reactive_region, water_guard, buffer_reference and inference_profile")
    values = {key: config.pop(key) for key in extras}
    compressed = additive and values["inference_profile"] == COMPRESSED_PROFILE
    if not (compressed and Path(config["model"]).suffix == ".pt2"
            or values["inference_profile"] == RAW_PROFILE and Path(config["model"]).suffix == ".pt"):
        raise ValueError("Reactive water requires raw .pt or the declared additive native .pt2 profile")
    if config.get("periodic") is not False:
        raise ValueError("Reactive water currently requires finite boundaries")
    for key in ("full_charge_spin", "region_charge_spin"):
        state = config.get(key)
        if not isinstance(state, list) or state != [0, 1] or any(type(v) is not int for v in state):
            raise ValueError("Reactive full and fragment DPA states must be explicit neutral singlets")
    full = values["full_region"]
    validate_region(full)
    pairs = sorted(zip(full["atom_ids"], full["symbols"], strict=True))
    values["full_region"] = {
        "atom_ids": [a for a, _ in pairs], "symbols": [s for _, s in pairs],
        "molecules": sorted(sorted(m) for m in full["molecules"])}
    config["region"] = values["full_region"]
    config["hamiltonian"] = "fixed-region-mechanical"
    config = validate_base(config, path)
    if (config["provider"] != {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}
            or [config["qm"]["charge"], config["qm"]["multiplicity"]] != [0, 1]
            or config["qm"]["grid_response"] is not True):
        raise ValueError("Reactive water requires CPU ground neutral-singlet QM with full grid response")
    settings = values["reactive_region"]
    required = {"pool_group_ids", "seed_group_id", "channels", "inner_radius_angstrom",
                "outer_radius_angstrom", "transfer_inner_angstrom", "transfer_outer_angstrom"}
    if not isinstance(settings, dict) or not required <= set(settings) or set(settings)-required-{"max_partitions"}:
        raise ValueError("reactive_region must explicitly declare its pool, seed, channels and switching intervals")
    if not isinstance(settings["pool_group_ids"], list) or not isinstance(settings["channels"], list):
        raise TypeError("Reactive pool and channels must be lists")
    for channel in settings["channels"]:
        if not isinstance(channel, dict) or set(channel) != {"hydrogen_id", "donor_group_id", "acceptor_group_id"}:
            raise ValueError("Each proton channel requires hydrogen_id, donor_group_id and acceptor_group_id")
    if not isinstance(values["groups"], list) or any(
            not isinstance(g, dict) or set(g) != {"group_id", "atom_ids", "symbols"}
            for g in values["groups"]):
        raise ValueError("Declare every initial water group explicitly")
    guard = values["water_guard"]
    if not isinstance(guard, dict) or set(guard) != {"min_oh_angstrom", "max_oh_angstrom", "min_foreign_oh_angstrom"}:
        raise ValueError("An explicit reactive water guard is required")
    values["water_guard"] = asdict(ReactiveWaterGuard(**guard))
    values["buffer_reference"], _, _ = validate_reference(
        values["buffer_reference"], model_sha256=config["model_sha256"],
        qm=config["qm"], provider=config["provider"], inference_profile=values["inference_profile"])
    candidate = {**config, **values}
    family = build_family(candidate)
    full = config.pop("region")
    symbols = dict(zip(full["atom_ids"], full["symbols"], strict=True))
    if (set(symbols) != set(family.atom_ids)
            or any(symbols[a] != s for g in family.groups for a, s in zip(g.atom_ids, g.symbols, strict=True))
            or {tuple(sorted(m)) for m in full["molecules"]} != {tuple(sorted(g.atom_ids)) for g in family.groups}):
        raise ValueError("Initial groups must cover full_region atoms, species and molecules exactly")
    values["full_region"] = full
    values["groups"] = [asdict(g) for g in family.groups]
    values["reactive_region"] = {
        "pool_group_ids": list(family.pool), "seed_group_id": family.seed_group_id,
        "channels": [asdict(c) for c in family.channels],
        **{key: getattr(family, key) for key in
           ("inner_radius_angstrom", "outer_radius_angstrom", "transfer_inner_angstrom",
            "transfer_outer_angstrom", "max_partitions")}}
    values["hamiltonian"] = hamiltonian
    if additive:
        from types import SimpleNamespace
        from .additive_config import cross_parameters
        from .embedding import FixedEmbeddingCharges
        from .molecular_additive import subset_charges
        from .providers import provider_contract, validate_provider

        table = ee["embedding"]
        if (not isinstance(table, dict) or not {"atom_ids", "charges_e", "source"} <= set(table)
                or set(table)-{"atom_ids", "charges_e", "source", "radii_angstrom"}):
            raise ValueError("Reactive EE requires an explicit full-system fixed charge table")
        charges = FixedEmbeddingCharges(**table)
        if set(charges.atom_ids) != set(family.atom_ids):
            raise ValueError("Embedding table must cover every full_region atom exactly")
        charges = subset_charges(charges, family.atom_ids)
        ee["embedding"] = {"atom_ids": list(charges.atom_ids), "charges_e": charges.charges_e.tolist(),
                           "source": charges.source}
        if charges.radii_angstrom is not None:
            ee["embedding"]["radii_angstrom"] = charges.radii_angstrom.tolist()
        pairs = cross_parameters(ee["cross"])
        if set(pairs) != {("O", "O"), ("O", "H"), ("H", "O"), ("H", "H")}:
            raise ValueError("Every water cross element pair needs exactly one explicit parameter")
        validate_provider(config["provider"], electrostatic=True)
        # Check the embedded solver restrictions without adding EE-only defaults
        # to the isolated-water calibration settings.
        validate_qm(config["qm"], electrostatic=True, provider=config["provider"])
        config["provider_contract"] = provider_contract(config["provider"], electrostatic=True)
        config.pop("full_charge_spin")
        config.pop("region_charge_spin")
        values.update(ee)
        # The same constructor checks every topology using metadata only.
        build_mixer({**config, **values}, SimpleNamespace(charge_spin=ee["environment_charge_spin"],
                                             pair_style=values["buffer_reference"].get("low_backend")),
                    lambda field: SimpleNamespace(charge=0, multiplicity=1, charges=field))
    # Reuse common file/method/dependency validation, then bind the complete new
    # model to a distinct identity. The validation cover is not an all-QM run.
    extend_identity(config, "reactive_model_identity", values)
    config.update(values)
    config.pop("reactive_model_identity")
    config["partitions"] = [asdict(p) for p in family.partitions]
    return config


def frame_record(frame, config):
    full = config["full_region"]
    symbols = dict(zip(full["atom_ids"], full["symbols"], strict=True))
    ids = tuple(atom["id"] for atom in frame["atoms"])
    if set(ids) != set(symbols):
        raise ValueError("Reactive frame does not cover every declared atom")
    geometry = Geometry(ids, tuple(symbols[a] for a in ids), [atom["x"] for atom in frame["atoms"]])
    family = build_family(config)
    ReactiveWaterGuard(**config["water_guard"]).validate(geometry, family)
    weights = family.evaluate(geometry)
    return {
        "hamiltonian": config["hamiltonian"],
        "reference_sha256": reference_identity(config["buffer_reference"]),
        "ownership_probabilities": weights.ownership_probabilities.tolist(),
        "channel_gaps_angstrom": weights.channel_gaps_angstrom.tolist(),
        "partitions": [{"id": p.partition_id, "weight": float(w),
                        "qm_atom_ids": list(p.qm_atom_ids), "qm_charge": 0,
                        "qm_electrons": p.electron_count, "multiplicity": 1,
                        "group_hydrogens": [[g, list(h)] for g, h in p.group_hydrogens],
                        "group_formal_charges": [[g, q] for g, q in p.group_charges]}
                       for p, w in zip(family.partitions, weights.weights, strict=True)],
    }


def validate_reactive_frame(frame, config):
    record = frame.get("reactive_water")
    if config.get("hamiltonian") not in REACTIVE_HAMILTONIANS:
        if record is not None:
            raise ValueError("Unexpected reactive water record")
        return
    if record != frame_record(frame, config):
        raise ValueError("Reactive water grouping, integer states or weights differ from the accepted geometry")
