# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Strict state preparation and portable references for existing TDA methods."""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path


def _read_json(raw):
    from .charges import _invalid_constant, _unique_object

    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def resolve_provider_reference(spec, config_path):
    """Expand an explicitly hashed file to the existing complete inline contract.

    Files are resolved relative to the configuration, never the process working
    directory. The semantic state hash, settings and runtime pins remain checked
    by the existing method-specific reference loader. No backend is constructed.
    """
    if not isinstance(spec, dict) or not isinstance(spec.get("state"), dict):
        return deepcopy(spec)
    if "reference_file" not in spec["state"]:
        return deepcopy(spec)
    state = spec["state"]
    if set(state) != {"method", "reference_file"}:
        raise ValueError("A file state requires method and reference_file only")
    source = state["reference_file"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueError("reference_file requires path and SHA256")
    if not isinstance(source["path"], str) or not source["path"].strip():
        raise ValueError("reference_file.path must be a nonempty filename")
    expected = source["sha256"]
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("reference_file.sha256 must be a lowercase SHA256")
    path = (Path(config_path).resolve().parent / source["path"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("State reference file SHA256 mismatch")
    if state["method"] == "uks-hirshfeld-cdft":
        from .cdft import validate_reference

        record = _read_json(raw)
        reference = validate_reference(record, electrostatic=record.get("charges") is not None)
        result = deepcopy(spec)
        result["state"] = {"method": state["method"], "reference": reference}
        return result
    from .providers import load_tda_reference

    reference = load_tda_reference(state["method"], _read_json(raw),
                                   engine=spec.get("engine", "pyscf"))
    result = deepcopy(spec)
    result["state"] = {"method": state["method"], "reference": reference.to_record()}
    return result


def validate_preparation(request):
    """Resolve an input using existing geometry, method and field validators."""
    from .excited import _charges_from_record, _geometry_from_record, _validate_geometry
    from .providers import _tda_types

    required = {"schema_version", "method", "qm", "qm_geometry", "state"}
    if (not isinstance(request, dict) or not required.issubset(request)
            or set(request) - required - {"environment_geometry", "charges", "engine", "device"}):
        raise ValueError("State preparation requires schema_version, method, qm, qm_geometry and state")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("Unsupported state-preparation schema_version")
    method = request["method"]
    engine, device = request.get("engine", "pyscf"), request.get("device", "cpu")
    if (engine, device) not in (("pyscf", "cpu"), ("gpu4pyscf", "cuda")):
        raise ValueError("State preparation requires pyscf/cpu or gpu4pyscf/cuda")
    settings_type = _tda_types(method, engine=engine)[0]
    qm = request["qm"]
    if not isinstance(qm, dict) or "tda_solver" not in qm:
        raise ValueError("State preparation requires explicit qm.tda_solver")
    if method == "pbe0-tda" and not {"grid_level", "grid_prune", "grid_response"}.issubset(qm):
        raise ValueError("PBE0 state preparation requires explicit grid controls")
    settings = settings_type(**qm)
    if settings.tda_solver != "dense":
        raise ValueError("Native state preparation requires an explicit dense solver")
    state = request["state"]
    if not isinstance(state, dict) or set(state) != {"initial_root", "label"}:
        raise ValueError("State preparation requires initial_root and label")
    root, label = state["initial_root"], state["label"]
    if type(root) is not int or not 1 <= root <= settings.nstates:
        raise ValueError("initial_root must identify one of the requested candidate states")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("State label must be nonempty")
    geometry = _geometry_from_record(request["qm_geometry"])
    environment = request.get("environment_geometry")
    environment = None if environment is None else _geometry_from_record(environment)
    charges = _charges_from_record(request.get("charges"))
    _validate_geometry(geometry, environment, charges)
    if charges is not None and charges.radii_angstrom is not None:
        raise ValueError("Native selected-state preparation supports point charges only")
    return method, settings, geometry, environment, charges, root, label


def _build_reference(method, settings, geometry, environment, charges, root, label):
    if settings.device == "cuda":
        from .gpu_excited import build_gpu_tda_reference

        return build_gpu_tda_reference(
            geometry, settings, initial_root=root, label=label,
            environment_geometry=environment, charges=charges,
        )
    if method == "rhf-tda":
        from .excited import build_tda_reference as build
    else:
        from .pbe0_tda import build_pbe0_tda_reference as build
    return build(geometry, settings, initial_root=root, label=label,
                 environment_geometry=environment, charges=charges)


def _reference_files(method, settings, reference, identity, *, electrostatic):
    """Serialize the existing single-state contract for either preparation route."""
    from .providers import provider_contract, validate_provider

    provider = validate_provider(
        {**identity, "state": {"method": method, "reference": reference.to_record()}},
        electrostatic=electrostatic,
    )
    reference_text = json.dumps(provider["state"]["reference"], indent=2, allow_nan=False) + "\n"
    reference_digest = hashlib.sha256(reference_text.encode()).hexdigest()
    file_provider = {**identity, "state": {
        "method": method, "reference_file": {"path": "reference.json", "sha256": reference_digest}}}
    report = {"schema_version": 1, "status": "reference-prepared", "method": method,
              "reference_sha256": reference.to_record()["sha256"],
              "reference_file_sha256": reference_digest,
              "provider_contract": provider_contract(provider, electrostatic=electrostatic),
              "qualification": "State initialization only; force, native and scientific qualification are separate"}
    files = {"reference.json": reference_text}
    for name, value in (("provider-inline.json", provider), ("provider-file.json", file_provider),
                        ("qm.json", asdict(settings)), ("result.json", report)):
        files[name] = json.dumps(value, indent=2, allow_nan=False) + "\n"
    return files, report


def _write_reference_files(directory, files):
    for name, contents in files.items():
        (directory / name).write_text(contents)


def prepare_state(request_path, output):
    """Prepare an immutable reference; no DPA model or native MD is evaluated."""
    from .config import software_identity
    path = Path(request_path).resolve()
    raw = path.read_bytes()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "started", "request_sha256":
                hashlib.sha256(raw).hexdigest(), "request_path": str(path),
                "software": software_identity()}

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    (output / "request.json").write_bytes(raw)
    write("manifest.json", manifest)
    try:
        resolved = validate_preparation(_read_json(raw))
        method, settings, _, environment, _, _, _ = resolved
        identity = {"engine": "gpu4pyscf" if settings.device == "cuda" else "pyscf",
                    "device": settings.device}
        manifest["software"] = software_identity(identity)
        write("manifest.json", manifest)
        reference = _build_reference(*resolved)
        files, report = _reference_files(
            method, settings, reference, identity, electrostatic=environment is not None
        )
        if path.read_bytes() != raw or software_identity(identity) != manifest["software"]:
            raise ValueError("State-preparation input or implementation changed during evaluation")
        _write_reference_files(output, files)
        manifest["status"] = "completed"
        write("manifest.json", manifest)
        return report
    except Exception as exc:
        manifest["status"] = "failed"
        write("manifest.json", manifest)
        write("failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def validate_state_set_preparation(request):
    """Resolve common inputs and distinct roots without constructing an engine."""
    from .multistate import _requested_roots, _state_names
    from .providers import _tda_types

    if not isinstance(request, dict) or "state" in request or "states" not in request:
        raise ValueError("Multi-state preparation requires states instead of state")
    engine, device = request.get("engine", "pyscf"), request.get("device", "cpu")
    if (engine, device, request.get("method")) not in (
        ("pyscf", "cpu", "rhf-tda"), ("pyscf", "cpu", "pbe0-tda"),
        ("gpu4pyscf", "cuda", "rhf-tda"),
    ):
        raise ValueError("Multi-state preparation requires CPU RHF/PBE0 TDA or GPU RHF TDA")
    states = _state_names(request["states"])
    single = dict(request)
    del single["states"]
    name, root = next(iter(states.items()))
    single["state"] = {"initial_root": root, "label": name}
    method, settings, geometry, environment, charges, _, _ = validate_preparation(single)
    states = _requested_roots(settings, states, settings_type=_tda_types(method, engine=engine)[0])
    return method, settings, geometry, environment, charges, states


def prepare_states(request_path, output):
    """Prepare a portable state collection with one checked SCF/TDA solution.

    Numbered directories preserve request order without using labels as paths.
    A completed parent manifest identifies the collection. Failed staging is
    retained for diagnosis and never silently promoted to a completed collection.
    """
    from .config import software_identity
    from .excited import _charges_record, _geometry_record
    from .multistate import (
        build_gpu_tda_references,
        build_pbe0_tda_references,
        build_tda_references,
    )
    from .providers import _tda_types

    path = Path(request_path).resolve()
    raw = path.read_bytes()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {"engine": "pyscf", "device": "cpu"}
    manifest = {"schema_version": 1, "status": "started", "request_sha256":
                hashlib.sha256(raw).hexdigest(), "request_path": str(path),
                "software": software_identity(identity)}

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def unchanged():
        if path.read_bytes() != raw or software_identity(identity) != manifest["software"]:
            raise ValueError("State-preparation input or implementation changed during evaluation")

    (output / "request.json").write_bytes(raw)
    write("manifest.json", manifest)
    try:
        method, settings, geometry, environment, charges, states = (
            validate_state_set_preparation(_read_json(raw))
        )
        if settings.device == "cuda":
            identity = {"engine": "gpu4pyscf", "device": "cuda"}
            manifest["software"] = software_identity(identity)
            write("manifest.json", manifest)
            builder = build_gpu_tda_references
        else:
            builder = build_tda_references if method == "rhf-tda" else build_pbe0_tda_references
        reference_type = _tda_types(method, engine=identity["engine"])[1]
        references = builder(
            geometry, settings, states=states, environment_geometry=environment, charges=charges
        )
        if set(references) != set(states):
            raise ValueError("Prepared references differ from the requested state names")
        payloads = {}
        report = {"schema_version": 1, "status": "references-prepared",
                  "method": method, **identity, "preparation": "shared-scf-tda",
                  "states": {},
                  "qualification": "State initialization only; force, native and scientific qualification are separate"}
        for ordinal, (name, root) in enumerate(states.items(), 1):
            reference = references[name]
            if (type(reference) is not reference_type or reference.settings != settings
                    or reference.seed_root != root or reference.label != name
                    or _geometry_record(reference.qm_geometry) != _geometry_record(geometry)
                    or _charges_record(reference.charges) != _charges_record(charges)
                    or (reference.environment_geometry is None) != (environment is None)
                    or (environment is not None and _geometry_record(reference.environment_geometry)
                        != _geometry_record(environment))):
                raise ValueError("Prepared reference differs from the requested state or inputs")
            files, leaf = _reference_files(
                method, settings, reference, identity, electrostatic=environment is not None
            )
            directory = f"state-{ordinal:04d}"
            payloads[directory] = files
            provider = json.loads(files["provider-file.json"])
            provider["state"]["reference_file"]["path"] = f"states/{directory}/reference.json"
            report["states"][name] = {"initial_root": root, "directory": f"states/{directory}",
                                      "reference_sha256": leaf["reference_sha256"],
                                      "provider": provider}
        unchanged()
        staging = output / ".states-pending"
        staging.mkdir()
        for directory, files in payloads.items():
            child = staging / directory
            child.mkdir()
            _write_reference_files(child, files)
        unchanged()
        staging.rename(output / "states")
        write("result.json", report)
        manifest["status"] = "completed"
        write("manifest.json", manifest)
        return report
    except Exception as exc:
        manifest["status"] = "failed"
        write("manifest.json", manifest)
        write("failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
