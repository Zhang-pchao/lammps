# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Portable run configuration, file identity, and physical-setting validation."""

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np

from .providers import provider_contract, validate_provider, validate_provider_regions
from .region import FixedRegion
from .state_inputs import resolve_provider_reference


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def software_identity(provider=None):
    """Identify the implementation and numerical dependencies for restart checks."""
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    versions = {}
    packages = ["numpy", "deepmd-kit", "pyscf"]
    if provider is not None and provider["engine"] == "gpu4pyscf":
        # Include wheel and source-install names without importing GPU libraries.
        packages += ["scipy", "torch", "mpi4py", "gpu4pyscf", "cupy",
                     "gpu4pyscf-cuda12x", "gpu4pyscf-cuda13x",
                     "gpu4pyscf-libxc-cuda12x", "gpu4pyscf-libxc-cuda13x",
                     "cupy-cuda12x", "cupy-cuda13x", "cutensor-cu12", "cutensor-cu13"]
    if provider is not None and provider["engine"] == "abacus":
        packages += ["ase", "abacuslite"]
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "coupling_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "packages": versions,
    }
    if provider is not None and provider["engine"] == "abacus":
        sources = {}
        for name in ("ase", "abacuslite"):
            try:
                distribution = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                sources[name] = None
                continue
            source = hashlib.sha256()
            for entry in sorted(distribution.files or (), key=str):
                if str(entry).endswith(".py"):
                    source.update(str(entry).encode() + b"\0")
                    source.update(Path(distribution.locate_file(entry)).read_bytes())
            sources[name] = source.hexdigest()
        result["backend_package_sources"] = sources
    return result


def validate_region(region):
    """Validate explicitly declared whole-molecule membership without loading a backend."""
    if not isinstance(region, dict):
        raise ValueError("region must be a mapping")
    if set(region) != {"atom_ids", "symbols", "molecules"}:
        raise ValueError("region requires atom_ids, symbols, and complete molecules")
    FixedRegion(tuple(region["atom_ids"]), tuple(region["symbols"]))
    molecules = region["molecules"]
    if (
        not isinstance(molecules, list)
        or not molecules
        or any(not isinstance(molecule, list) or not molecule for molecule in molecules)
    ):
        raise ValueError("region.molecules must contain nonempty lists of atom IDs")
    members = [atom for molecule in molecules for atom in molecule]
    if any(type(atom) is not int or atom <= 0 for atom in members):
        raise ValueError("molecule atom IDs must be positive integers")
    if sorted(members) != sorted(region["atom_ids"]):
        raise ValueError("region.molecules must partition the declared region IDs exactly")


def validate_qm(qm, *, electrostatic=False, provider=None, base_dir=None):
    """Resolve method-specific finite settings without importing PySCF."""
    normalized = validate_provider(provider, electrostatic=electrostatic)
    if normalized["engine"] == "abacus":
        from .abacus import validate_mechanical_abacus_settings

        return validate_mechanical_abacus_settings(qm, Path.cwd() if base_dir is None else base_dir)
    if normalized["state"]["method"] != "ground":
        from dataclasses import asdict

        from .providers import load_tda_reference, tda_settings

        method = normalized["state"]["method"]
        reference = load_tda_reference(method, normalized["state"]["reference"],
                                       engine=normalized["engine"])
        return asdict(tda_settings(qm, method, reference))
    qm_defaults = {
        "xc": "pbe",
        "basis": "def2-svp",
        "charge": 0,
        "multiplicity": 1,
        "conv_tol": 1e-9,
        "conv_tol_grad": None,
        "max_cycle": 100,
        "grid_level": 3,
        "grid_response": True,
        "threads": None,
    }
    if electrostatic:
        qm_defaults.update(nlc_grid_level=None, include_qm_dipole=False)
    allowed = set(qm_defaults) | {"nlc_grid_level", "density_fit", "scf_level_shift_hartree", "scf_solver"}
    if not isinstance(qm, dict) or set(qm).difference(allowed):
        raise ValueError("qm contains unsupported PySCF configuration keys")
    qm = {**qm_defaults, **qm}
    from .backends import validate_density_fit, validate_scf_level_shift, validate_scf_solver

    solver = validate_scf_solver(qm.get("scf_solver"), qm["conv_tol_grad"])
    if solver is None:
        qm.pop("scf_solver", None)
    elif electrostatic or normalized["engine"] != "pyscf":
        raise ValueError("The diis-newton solver currently requires isolated CPU ground PySCF")
    shift = validate_scf_level_shift(qm.get("scf_level_shift_hartree"))
    if shift is None:
        qm.pop("scf_level_shift_hartree", None)
    else:
        qm["scf_level_shift_hartree"] = shift
    density_fit = validate_density_fit(qm.get("density_fit"))
    if density_fit is None:
        # Keep conventional-integral inputs and normalized restart records unchanged.
        qm.pop("density_fit", None)
    else:
        qm["density_fit"] = density_fit
    if not electrostatic and qm.get("nlc_grid_level") is None:
        # Preserve the ordinary ground-state defaults and normalized records.
        qm.pop("nlc_grid_level", None)
    for key in ("xc", "basis"):
        if not isinstance(qm[key], str) or not qm[key].strip():
            raise ValueError(f"qm.{key} must be a nonempty string")
    if type(qm["charge"]) is not int:
        raise ValueError("qm.charge must be an integer")
    for key in ("multiplicity", "max_cycle"):
        _positive_integer(qm[key], f"qm.{key}")
    _positive_integer(qm["grid_level"], "qm.grid_level", allow_zero=True)
    if type(qm["grid_response"]) is not bool:
        raise ValueError("qm.grid_response must be true or false")
    if qm["threads"] is not None:
        _positive_integer(qm["threads"], "qm.threads")
    if isinstance(qm["conv_tol"], bool) or not np.isfinite(qm["conv_tol"]) or qm["conv_tol"] <= 0:
        raise ValueError("qm.conv_tol must be finite and positive")
    grad_tol = qm["conv_tol_grad"]
    if grad_tol is not None and (
        isinstance(grad_tol, bool)
        or not isinstance(grad_tol, (int, float))
        or not np.isfinite(grad_tol)
        or grad_tol <= 0
    ):
        raise ValueError("qm.conv_tol_grad must be finite and positive or null")
    if qm.get("nlc_grid_level") is not None:
        _positive_integer(qm["nlc_grid_level"], "qm.nlc_grid_level", allow_zero=True)
    if electrostatic:
        if type(qm["include_qm_dipole"]) is not bool:
            raise ValueError("qm.include_qm_dipole must be true or false")
    if normalized["engine"] == "gpu4pyscf":
        from .gpu_backends import validate_gpu_settings

        qm = validate_gpu_settings(qm)
    return qm


def load_config(path):
    """Resolve declared files and validate the deliberately small run schema."""
    path = Path(path).resolve()
    from .charges import _invalid_constant, _unique_object

    config = json.loads(
        path.read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )
    return _validate_config(config, path)


def _validate_config(config, path):
    """Validate a parsed request using its original file location for relative paths."""
    from copy import deepcopy

    config = deepcopy(config)
    path = Path(path).resolve()
    if not isinstance(config, dict):
        raise ValueError("Mechanical configuration must be a JSON object")
    if config.get("hamiltonian", "fixed-region-mechanical") != "fixed-region-mechanical":
        raise ValueError("Mechanical configuration requires fixed-region-mechanical")
    config["hamiltonian"] = "fixed-region-mechanical"
    allowed = {
        "hamiltonian",
        "schema_version",
        "provider",
        "data_file",
        "model",
        "model_sha256",
        "type_symbols",
        "periodic",
        "region",
        "full_charge_spin",
        "region_charge_spin",
        "qm",
        "timestep_fs",
        "steps",
        "sample_interval",
        "temperature_K",
        "velocity_seed",
    }
    unknown = set(config).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key in ("data_file", "model"):
        config[key] = str((path.parent / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: {config[key]}")
    actual = file_digest(config["model"])
    if actual != config["model_sha256"]:
        raise ValueError("Model SHA256 does not match the declared model")
    config["data_sha256"] = file_digest(config["data_file"])
    if not isinstance(config.get("periodic"), bool):
        raise ValueError("periodic must be true or false (partial PBC is unsupported)")
    symbols = config["type_symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(
            not isinstance(value, str) or re.fullmatch(r"[A-Z][a-z]?", value) is None
            for value in symbols
        )
    ):
        raise ValueError("type_symbols must map each LAMMPS atom type to an element")
    validate_region(config["region"])
    for key in ("full_charge_spin", "region_charge_spin"):
        value = config.get(key)
        if value is not None and (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) is not int for item in value)
            or value[1] < 1
        ):
            raise ValueError(f"{key} requires integer charge and positive multiplicity")
    config["provider"] = validate_provider(resolve_provider_reference(config.get("provider"), path))
    qm = config["qm"] = validate_qm(config["qm"], provider=config["provider"], base_dir=path.parent)
    boundary = "periodic-same-cell" if qm.get("boundary") == "periodic-same-cell" else "isolated"
    config["provider_contract"] = provider_contract(config["provider"], region_boundary=boundary, qm=qm)
    validate_provider_regions(config["provider"], config["region"], periodic=config["periodic"],
                              region_boundary=boundary)
    if config["provider"]["engine"] == "abacus" and any(
            symbol not in qm["pseudopotentials"] for symbol in config["region"]["symbols"]):
        raise ValueError("An ABACUS region element has no declared pseudopotential")
    state = config.get("region_charge_spin")
    if state is not None and state != [qm["charge"], qm["multiplicity"]]:
        raise ValueError("Region DP and QM electronic states must match in this driver")
    config["steps"] = _positive_integer(config["steps"], "steps", allow_zero=True)
    config["sample_interval"] = _positive_integer(
        config.get("sample_interval", 1), "sample_interval"
    )
    if isinstance(config["timestep_fs"], bool):
        raise ValueError("timestep_fs must be a number")
    step = float(config["timestep_fs"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError("timestep_fs must be finite and positive")
    config["timestep_fs"] = step
    if "temperature_K" in config:
        if isinstance(config["temperature_K"], bool):
            raise ValueError("temperature_K must be a number")
        temperature = float(config["temperature_K"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature_K must be finite and positive")
        config["temperature_K"] = temperature
        _positive_integer(config["velocity_seed"], "velocity_seed")
    config["software"] = software_identity(config["provider"])
    physics = {
        key: config.get(key)
        for key in (
            "hamiltonian",
            "model_sha256",
            "provider",
            "provider_contract",
            "software",
            "data_sha256",
            "type_symbols",
            "periodic",
            "region",
            "qm",
            "full_charge_spin",
            "region_charge_spin",
            "timestep_fs",
        )
    }
    config["physics_sha256"] = hashlib.sha256(
        json.dumps(physics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return config
