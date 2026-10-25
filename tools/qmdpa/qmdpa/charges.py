# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Resolve declared fixed charges without inferring chemistry or electronic states.

Only the Python standard library is used. Resolution does not load a provider
or read coordinates. The result supplies the explicit FixedEmbeddingCharges
constructor fields and a JSON-safe provenance record.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

TOTAL_CHARGE_ATOL_E = 1e-8
_PRESETS = {
    "TIP3P-original-1983": (
        -0.834,
        0.417,
        "https://docs.lammps.org/Howto_tip3p.html",
        ("Original TIP3P, Jorgensen et al., J. Chem. Phys. 79, 926 (1983); doi:10.1063/1.445869"),
    ),
    "SPC": (
        -0.820,
        0.410,
        "https://docs.lammps.org/Howto_spc.html",
        "SPC charge parameters tabulated in the official LAMMPS documentation",
    ),
    "SPC/E": (
        -0.8476,
        0.4238,
        "https://docs.lammps.org/Howto_spc.html",
        "SPC/E charge parameters tabulated in the official LAMMPS documentation",
    ),
}
_FIELDS = {
    "per-atom": {"charges"},
    "atom-types": {"atom_types", "type_charges"},
    "water-preset": {"preset", "groups"},
    "file": {"path", "sha256"},
}


def _object(value, required, optional=(), *, name):
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or (set(value) - required - set(optional))
    ):
        raise ValueError(f"{name} has missing or unsupported fields")
    return value


def _source(value, name="source"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must explicitly describe the charge source")
    return value


def _charge(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _atom_id(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Atom IDs must be positive integers, not booleans")
    return value


def _entries(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    return value


def _coverage(actual, expected, name):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(
            f"{name} must cover the declared atoms exactly; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _per_atom(entries):
    values = {}
    for entry in _entries(entries, "charges"):
        _object(entry, {"atom_id", "charge_e"}, name="Each per-atom charge")
        atom = _atom_id(entry["atom_id"])
        if atom in values:
            raise ValueError(f"Duplicate charge atom ID {atom}")
        values[atom] = _charge(entry["charge_e"], "charge_e")
    return values


def _atom_types(spec):
    assigned, values = {}, {}
    for entry in _entries(spec["atom_types"], "atom_types"):
        _object(entry, {"atom_id", "type"}, name="Each atom type assignment")
        atom = _atom_id(entry["atom_id"])
        label = _source(entry["type"], "Atom type label")
        if atom in assigned:
            raise ValueError(f"Duplicate atom type assignment for atom ID {atom}")
        assigned[atom] = label
    for entry in _entries(spec["type_charges"], "type_charges"):
        _object(entry, {"type", "charge_e"}, name="Each type charge")
        label = _source(entry["type"], "Atom type label")
        if label in values:
            raise ValueError(f"Duplicate charge for atom type {label}")
        values[label] = _charge(entry["charge_e"], "Type charge_e")
    if set(assigned.values()) != set(values):
        raise ValueError("Every used atom type needs exactly one charge; unused types are rejected")
    return (
        {atom: values[label] for atom, label in assigned.items()},
        {
            "atom_types": [{"atom_id": atom, "type": assigned[atom]} for atom in sorted(assigned)],
            "type_charges": [
                {"type": label, "charge_e": values[label]} for label in sorted(values)
            ],
        },
    )


def _water_preset(spec, identity):
    name = spec["preset"]
    if not isinstance(name, str):
        raise TypeError("preset must name an explicitly supported water charge template")
    if name.upper().startswith("TIP4P") or name.upper() == "OPC":
        raise ValueError(
            "TIP4P/OPC presets are unsupported: virtual charge sites and their force "
            "mapping are not implemented. Choose a supported three-site charge template "
            "or supply an independently specified fixed-site model."
        )
    if name not in _PRESETS:
        raise ValueError(f"Unknown water charge preset; choose one of {sorted(_PRESETS)}")
    oxygen_charge, hydrogen_charge, url, citation = _PRESETS[name]
    values, groups = {}, []
    for group in _entries(spec["groups"], "groups"):
        if not isinstance(group, (list, tuple)) or len(group) != 3:
            raise ValueError("Each water group must explicitly list three IDs in O,H,H order")
        oxygen, first, second = (_atom_id(atom) for atom in group)
        if len({oxygen, first, second}) != 3 or any(atom in values for atom in group):
            raise ValueError("Water group atom IDs must be distinct and appear in only one group")
        if tuple(identity.get(atom) for atom in group) != ("O", "H", "H"):
            raise ValueError("Each water group must reference present atoms in O,H,H order")
        values.update({oxygen: oxygen_charge, first: hydrogen_charge, second: hydrogen_charge})
        groups.append([oxygen, *sorted((first, second))])
    return values, {
        "preset": name,
        "groups": sorted(groups),
        "reference_url": url,
        "reference": citation,
        "scope": "Charge template only; no geometry, LJ, constraints or full-forcefield assignment",
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant {value} is unsupported")


def _file(spec, base_directory):
    filename, expected = spec["path"], spec["sha256"]
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("External charges require an explicit file path")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
        raise ValueError("External charges require a complete SHA256")
    raw = (Path(base_directory) / filename).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected.lower():
        raise ValueError("External charge file SHA256 differs from its declared identity")
    document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    _object(
        document, {"schema_version", "units", "source", "charges"}, name="External charge document"
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("External charge document schema_version must be 1")
    if document["units"] != "e":
        raise ValueError("External charge document units must be e")
    source = _source(document["source"], "External file source")
    # Content identity deliberately excludes the file's name and absolute location.
    return _per_atom(document["charges"]), {
        "sha256": digest,
        "file_schema_version": 1,
        "file_source": source,
    }


def resolve_fixed_charges(spec, atom_ids, symbols, *, base_directory="."):
    """Resolve one complete source, then explicit overrides, in requested ID order.

    The spec is a strict JSON-compatible mapping; IDs are positive Python ints.
    Returned embedding is suitable for FixedEmbeddingCharges(**embedding).
    Returned provenance includes base values, overrides and content identities.
    No file path participates in that record. Nonzero and fractional declared
    totals are allowed here; the caller separately validates electronic states.
    """
    if not isinstance(spec, dict):
        raise TypeError("A fixed-charge request must be an explicit object")
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise ValueError(f"Choose an explicit charge kind from {sorted(_FIELDS)}")
    _object(
        spec,
        {"kind", "units", "source", "expected_total_charge_e"} | _FIELDS[kind],
        {"overrides"},
        name="Fixed-charge request",
    )
    if spec["units"] != "e":
        raise ValueError("Fixed-charge units must be e")
    source = _source(spec["source"])
    expected = _charge(spec["expected_total_charge_e"], "expected_total_charge_e")
    ids = tuple(_atom_id(atom) for atom in atom_ids)
    elements = tuple(symbols)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Declared atom IDs must be nonempty and distinct")
    if len(elements) != len(ids) or any(
        not isinstance(symbol, str) or re.fullmatch(r"[A-Z][a-z]?", symbol) is None
        for symbol in elements
    ):
        raise ValueError("Each declared atom ID requires one element symbol")
    if kind == "per-atom":
        values, details = _per_atom(spec["charges"]), {}
    elif kind == "atom-types":
        values, details = _atom_types(spec)
    elif kind == "water-preset":
        values, details = _water_preset(spec, dict(zip(ids, elements)))
    else:
        values, details = _file(spec, base_directory)
    _coverage(values, ids, "Base charges")
    base = [values[atom] for atom in ids]
    overrides = spec.get("overrides", [])
    if not isinstance(overrides, list):
        raise TypeError("overrides must be a list of explicitly sourced atom-ID replacements")
    changed, applied = set(), []
    for item in overrides:
        _object(item, {"atom_id", "charge_e", "source"}, name="Each charge override")
        atom = _atom_id(item["atom_id"])
        if atom not in values:
            raise ValueError(f"Override atom ID {atom} is outside the declared atoms")
        if atom in changed:
            raise ValueError(f"Duplicate charge override for atom ID {atom}")
        changed.add(atom)
        override_source = _source(item["source"], "Override source")
        replacement = _charge(item["charge_e"], "Override charge_e")
        applied.append(
            {
                "atom_id": atom,
                "previous_charge_e": values[atom],
                "charge_e": replacement,
                "source": override_source,
            }
        )
        values[atom] = replacement
    charges = [values[atom] for atom in ids]
    try:
        total = math.fsum(charges)
    except OverflowError as exc:
        raise ValueError("Total fixed charge must be finite") from exc
    if not math.isfinite(total) or not math.isclose(
        total, expected, rel_tol=0, abs_tol=TOTAL_CHARGE_ATOL_E
    ):
        raise ValueError(
            f"Resolved total charge {total} e differs from expected_total_charge_e={expected}"
        )
    customized = bool(applied)
    description = f"{source}; fixed-charge resolver v1 ({kind})"
    if kind == "water-preset":
        description += f"; {details['preset']}; {details['reference_url']}"
        description += "; custom charge template after overrides" if customized else ""
    elif kind == "file":
        description += f"; {details['file_source']}; sha256={details['sha256']}"
    return {
        "embedding": {"atom_ids": list(ids), "charges_e": charges, "source": description},
        "provenance": {
            "resolver_schema_version": 1,
            "kind": kind,
            "units": "e",
            "source": source,
            "atom_ids": list(ids),
            "symbols": list(elements),
            "base_charges_e": base,
            "expected_total_charge_e": expected,
            "total_charge_e": total,
            "total_charge_atol_e": TOTAL_CHARGE_ATOL_E,
            "customized": customized,
            "details": details,
            "overrides": sorted(applied, key=lambda item: item["atom_id"]),
        },
    }
