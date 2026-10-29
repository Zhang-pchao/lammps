#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Prepare and bind a fixed isolated-water energy reference for variable buffers."""
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np

from .adaptive import _real, _text
from .config import file_digest, software_identity, validate_qm
from .core import Geometry, _evaluate
from .inference import COMPRESSED_PROFILE, RAW_PROFILE
from .providers import validate_provider
from .selection_config import read_json

KIND = "isolated-water-qm-minus-dpa-v1"
FIELDS = {
    "schema_version", "kind", "model_sha256", "qm", "provider", "inference_profile",
    "geometry_angstrom", "high_energy_ev", "low_energy_ev", "source", "software",
}


def reference_software():
    value = software_identity()
    value["packages"]["torch"] = importlib.metadata.version("torch")
    return value


def water_geometry(positions):
    geometry = Geometry((1, 2, 3), ("O", "H", "H"), positions)
    if any(np.linalg.norm(geometry.positions[i]-geometry.positions[j]) == 0
           for i in range(3) for j in range(i)):
        raise ValueError("Reference water atoms must be distinct")
    return geometry


def reference_identity(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_reference(record, *, model_sha256, qm, provider, inference_profile=RAW_PROFILE):
    """Require the same settings, source and dependencies without running QM."""
    compressed = inference_profile == COMPRESSED_PROFILE
    if inference_profile not in (RAW_PROFILE, COMPRESSED_PROFILE):
        raise ValueError("Unsupported water reference inference profile")
    fields = FIELDS | ({"low_backend"} if compressed else set())
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Water reference has missing or unsupported fields")
    if type(record["schema_version"]) is not int or record["schema_version"] != (2 if compressed else 1):
        raise ValueError("Water reference schema_version differs from its inference profile")
    if record["kind"] != KIND or record["inference_profile"] != inference_profile:
        raise ValueError("Water reference kind or inference profile differs")
    if compressed and record["low_backend"] not in ("deepmd", "deepmd/kk"):
        raise ValueError("Compressed water reference requires an explicit native low backend")
    if record["model_sha256"] != model_sha256:
        raise ValueError("Water reference model checksum differs")
    if record["qm"] != qm or record["provider"] != provider:
        raise ValueError("Water reference QM/provider settings differ")
    if record["software"] != reference_software():
        raise ValueError("Water reference source or numerical dependencies differ; prepare a new reference")
    water_geometry(record["geometry_angstrom"])
    _text(record["source"], "Water reference source")
    high = _real(record["high_energy_ev"], "Reference QM energy")
    low = _real(record["low_energy_ev"], "Reference DPA energy")
    delta = _real(high-low, "Reference QM-minus-DPA difference")
    # Round-trip a defensive copy and reject NaN in any provenance field.
    normalized = json.loads(json.dumps(record, allow_nan=False))
    return normalized, delta, "water-reference:" + reference_identity(normalized)


def prepare_reference(request_path, output):
    """Calculate the declared reference once; failures never publish reference.json."""
    request_path = Path(request_path).resolve()
    raw = read_json(request_path.read_text())
    fields = {"model", "model_sha256", "qm", "provider", "geometry_angstrom",
              "inference_profile", "source"}
    compressed = isinstance(raw, dict) and raw.get("inference_profile") == COMPRESSED_PROFILE
    fields |= {"low_backend"} if compressed else set()
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Water-reference request has missing or unsupported fields")
    if raw["inference_profile"] not in (RAW_PROFILE, COMPRESSED_PROFILE):
        raise ValueError("Water-reference preparation requires an explicit supported inference profile")
    if compressed and raw["low_backend"] not in ("deepmd", "deepmd/kk"):
        raise ValueError("Compressed water reference requires an explicit native low backend")
    provider = validate_provider(raw["provider"])
    if provider != {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}:
        raise ValueError("Water-reference preparation requires CPU PySCF ground state")
    qm = validate_qm(raw["qm"], provider=provider)
    if [qm["charge"], qm["multiplicity"]] != [0, 1] or qm["grid_response"] is not True:
        raise ValueError("Reference water requires a neutral singlet and complete grid response")
    geometry = water_geometry(raw["geometry_angstrom"])
    source = _text(raw["source"], "Water reference source")
    if not isinstance(raw["model"], str) or not raw["model"]:
        raise ValueError("The reference model path must be supplied")
    model = (request_path.parent/raw["model"]).resolve()
    if (not isinstance(raw["model_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", raw["model_sha256"]) is None
            or file_digest(model) != raw["model_sha256"]):
        raise ValueError("Reference model SHA256 differs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"request": raw, "request_sha256": file_digest(request_path),
                "model_path": str(model), "software": reference_software(),
                "scope": "Fixed isolated-water energy calibration; not chemical accuracy"}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    try:
        from .backends import DeepMDBackend
        from .inference import prepare_raw_inference, strict_runtime_state
        from .providers import make_qm_provider

        if compressed:
            from .execution import validate_dpa4c_archive
            from .native_deepmd import NativeDeepMDDomainBackend

            validate_dpa4c_archive(model)
            backend = NativeDeepMDDomainBackend(
                model, model_sha256=raw["model_sha256"], type_symbols=("O", "H"),
                pair_style=raw["low_backend"])
            try:
                low = _evaluate(backend, geometry)
            finally:
                backend.close_checked()
            runtime = {"profile": COMPRESSED_PROFILE, "native": backend.statistics}
        else:
            torch = prepare_raw_inference(model)
            low = _evaluate(DeepMDBackend(model, charge_spin=[0, 1], no_jit=True), geometry)
            runtime = strict_runtime_state(torch)
        high = _evaluate(make_qm_provider(qm, provider=provider), geometry)
        if file_digest(model) != raw["model_sha256"] or file_digest(request_path) != manifest["request_sha256"]:
            raise ValueError("Water reference input changed during calculation")
        record = {
            "schema_version": 2 if compressed else 1, "kind": KIND, "model_sha256": raw["model_sha256"],
            "qm": qm, "provider": provider, "inference_profile": raw["inference_profile"],
            "geometry_angstrom": geometry.positions.tolist(), "high_energy_ev": high.energy,
            "low_energy_ev": low.energy, "source": source, "software": manifest["software"],
        }
        if compressed:
            record["low_backend"] = raw["low_backend"]
        record, delta, identity = validate_reference(
            record, model_sha256=raw["model_sha256"], qm=qm, provider=provider,
            inference_profile=raw["inference_profile"])
        (output/"runtime.json").write_text(json.dumps(runtime, indent=2)+"\n")
        temporary = output/"reference.json.part"
        temporary.write_text(json.dumps(record, indent=2)+"\n")
        temporary.replace(output/"reference.json")
        return {"reference": str(output/"reference.json"), "identity": identity,
                "delta_water_ev": delta, "status": "prepared",
                "scientific_acceptance": "not assessed"}
    except Exception as exc:
        (output/"failure.json").write_text(json.dumps(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2)+"\n")
        raise
