# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit finite PBE0 interaction-correction input; validation imports no engine."""
from copy import deepcopy

from .config import validate_region

CORRECTED_GAP_HAMILTONIAN = "finite-electrostatic-gap-anchor-cross-delta"
INTERACTION_MODEL = "finite-cross-interaction-delta-v1"


def validate_interaction_delta(value, qm_region, environment_region, method):
    required = {"model", "ground_method", "core_region", "fragment_region", "charge_spin"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Interaction correction requires model, ground_method, complete regions and charge_spin")
    if value["model"] != INTERACTION_MODEL or value["ground_method"] != "matched-pbe0":
        raise ValueError("Choose the explicit cross interaction model with matched-pbe0 ground terms")
    if method not in ("pbe0-tda", "pbe0-ground"):
        raise ValueError("This native interaction correction requires the matched PBE0 route")
    conditions = value["charge_spin"]
    if not isinstance(conditions, dict) or set(conditions) != {"pair", "core", "fragment"}:
        raise ValueError("Pair, core and fragment charge_spin conditions must be explicit")
    for condition in conditions.values():
        if (not isinstance(condition, list) or condition != [0, 1]
                or any(type(v) is not int for v in condition)):
            raise ValueError("This native correction supports explicit neutral singlets [0, 1] only")
    for name, parent in (("core_region", qm_region), ("fragment_region", environment_region)):
        region = value[name]
        validate_region(region)
        selected = set(region["atom_ids"])
        if not selected <= set(parent["atom_ids"]):
            raise ValueError(name + " must be contained in its declared parent region")
        parent_symbols = dict(zip(parent["atom_ids"], parent["symbols"]))
        if any(parent_symbols[i] != s for i, s in zip(region["atom_ids"], region["symbols"])):
            raise ValueError(name + " changes the parent's element identity")
        included = [group for group in parent["molecules"] if selected.intersection(group)]
        if (any(not set(group) <= selected for group in included)
                or {frozenset(v) for v in region["molecules"]} != {frozenset(v) for v in included}):
            raise ValueError(name + " must select complete molecules from its parent")
    return deepcopy(value)
