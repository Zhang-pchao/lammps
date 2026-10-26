# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Finite common-core RHF/TDA molecular-family admission and physics identity."""

import copy
from dataclasses import asdict
from pathlib import Path

from .adaptive_config import MOLECULAR_HAMILTONIAN
from .adaptive_config import _validate_config as validate_molecular_config
from .config import file_digest
from .excited import TDASettings, _digest
from .selected_adaptive import HAMILTONIAN, STATE_KIND
from .selection_config import read_json
from .state_family import StateFamilyReference


def load_selected_adaptive_config(path):
    path = Path(path).resolve()
    return _validate_config(read_json(path.read_text()), path)


def _validate_config(config, path):
    if not isinstance(config, dict):
        raise TypeError("Selected molecular configuration must be an object")
    raw = copy.deepcopy(config)
    if raw.get("hamiltonian") != HAMILTONIAN or raw.get("periodic") is not False:
        raise ValueError("Selected molecular dynamics requires its finite mechanical Hamiltonian")
    provider = {"engine": "pyscf", "device": "cpu", "state": {"method": STATE_KIND}}
    if raw.get("provider") != provider:
        raise ValueError("Selected molecular dynamics requires the explicit CPU common-core RHF/TDA provider")
    specification = raw.pop("family_reference", None)
    if (not isinstance(specification, dict) or set(specification) != {"path", "sha256"}
            or not isinstance(specification["path"], str) or not specification["path"]):
        raise ValueError("family_reference requires a pinned file path and SHA256")
    reference_path = (Path(path).parent / specification["path"]).resolve()
    if file_digest(reference_path) != specification["sha256"]:
        raise ValueError("State-family reference file identity changed")
    family = StateFamilyReference.from_record(read_json(reference_path.read_text()))
    settings = TDASettings(**raw["qm"])
    if settings != family.core.settings or settings.tda_solver != "dense":
        raise ValueError("Selected family controls must equal the prepared dense-TDA reference")
    parts = raw.get("partitions")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Explicit prepared partitions are required")
    for part in parts:
        if (not isinstance(part, dict) or part.get("electronic_state") != STATE_KIND
                or part.get("method_identity") != "cpu-rhf-tda:"+family.sha256
                or part.get("energy_reference") != family.sha256
                or part.get("energy_offset_ev", 0) != 0 or part.get("offset_source") is not None):
            raise ValueError("Every partition must retain its common state identity without energy shifts")
        part["electronic_state"] = "ground"
    # Reuse all fixed-k coverage, grouping, native paths, timestep, model and
    # neutral-singlet checks without constructing any scientific provider.
    raw.update(hamiltonian=MOLECULAR_HAMILTONIAN,
               provider={"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}},
               qm={"xc": "hf", "basis": settings.basis, "charge": 0, "multiplicity": 1})
    normalized = validate_molecular_config(raw, path)
    group_atoms = {g["group_id"]: tuple(g["atom_ids"]) for g in normalized["groups"]}
    symbols = dict(zip(normalized["full_region"]["atom_ids"], normalized["full_region"]["symbols"]))
    core = tuple(sorted(a for g in normalized["core_group_ids"] for a in group_atoms[g]))
    if family.core.qm_geometry.atom_ids != core:
        raise ValueError("The prepared common reference must contain exactly the fixed core")
    if {p["partition_id"] for p in normalized["partitions"]} != {name for name, _ in family.partitions}:
        raise ValueError("Prepared state references must cover the complete declared family")
    for part in normalized["partitions"]:
        expected = tuple(sorted(a for g in part["qm_group_ids"] for a in group_atoms[g]))
        reference = family.reference(part["partition_id"])
        if (reference.qm_geometry.atom_ids != expected
                or reference.qm_geometry.symbols != tuple(symbols[a] for a in expected)):
            raise ValueError("Prepared state atoms/elements differ from their complete partition")
        part["electronic_state"] = STATE_KIND
    if file_digest(reference_path) != specification["sha256"]:
        raise ValueError("State-family reference changed during admission")
    normalized.update(
        hamiltonian=HAMILTONIAN, provider=provider, qm=asdict(settings),
        family_reference=family.to_record(), family_reference_file_sha256=specification["sha256"],
        provider_contract={"engine": "pyscf", "device": "cpu", "method": STATE_KIND,
                           "boundary": "finite", "embedding": "mechanical",
                           "state_correspondence": "common-core-and-bilateral-transition-density",
                           "force_coordinates": ["qm", "environment", "switching"],
                           "electronic_propagation": False})
    normalized["adaptive_contract"]["state_correspondence"] = "all-active-pairs-and-common-core"
    physics = {k: v for k, v in normalized.items() if k not in {
        "physics_sha256", "model", "data_file", "steps", "sample_interval", "temperature_K", "velocity_seed"}}
    normalized["physics_sha256"] = _digest(physics)
    return normalized
