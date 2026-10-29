#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Named scalar values and complete same-frame Jacobians for bias feedback.

This interface admits no electronic provider or sampled surface by itself.
Coordinates and Cartesian forces use Angstrom and eV/Angstrom respectively.
"""

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from .core import Geometry, _atom_ids, _readonly_array

HARTREE_EV = 27.211386245988
UNITS = {"eV": {"eV": 1.0, "hartree": HARTREE_EV},
         "angstrom": {"angstrom": 1.0, "bohr": 0.529177210903},
         "dimensionless": {"dimensionless": 1.0}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def geometry_identity(geometry):
    if not isinstance(geometry, Geometry) or geometry.periodic:
        raise ValueError("Named CVs require an explicit finite Geometry")
    rows = sorted(zip(geometry.atom_ids, geometry.symbols, geometry.positions.tolist(), strict=True))
    return digest(rows)


def _dependencies(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("CV dependencies require named source/state/partition checksums")
    if any(not isinstance(k, str) or not k or not isinstance(v, str)
           or re.fullmatch(r"[0-9a-f]{64}", v) is None for k, v in raw.items()):
        raise ValueError("CV dependency identities must be SHA256 strings")
    return dict(sorted(raw.items()))


@dataclass(frozen=True)
class ScalarCV:
    name: str
    unit: str
    dependencies_json: str

    @property
    def dependencies(self):
        return json.loads(self.dependencies_json)


class NamedCVSchema:
    """Version one supports multiple named scalars; tensors fail explicitly."""

    def __init__(self, raw):
        if (not isinstance(raw, dict)
                or set(raw) != {"schema_version", "coordinate_unit", "variables"}
                or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
                or raw["coordinate_unit"] != "angstrom"):
            raise ValueError("Named CV schema requires version 1 and Angstrom coordinates")
        variables = raw["variables"]
        if not isinstance(variables, list) or not variables:
            raise ValueError("Named CV variables must be a nonempty list")
        specs = []
        for item in variables:
            if not isinstance(item, dict) or set(item) != {"name", "shape", "unit", "dependencies"}:
                raise ValueError("Each CV requires name, shape, unit and dependencies")
            name = item["name"]
            if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
                    or name.startswith("qmdpa_")):
                raise ValueError("CV names must be identifiers outside the qmdpa_ namespace")
            if item["shape"] != [] or item["unit"] not in UNITS:
                raise ValueError("Named CV version 1 requires scalar shape [] and an explicit supported unit")
            specs.append(ScalarCV(name, item["unit"], json.dumps(_dependencies(item["dependencies"]), sort_keys=True)))
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("Named CV names must be unique")
        self.variables = tuple(specs)
        self.sha256 = digest(self.record())

    def record(self):
        return {"schema_version": 1, "coordinate_unit": "angstrom", "variables": [
            {"name": s.name, "shape": [], "unit": s.unit, "dependencies": s.dependencies}
            for s in self.variables]}


class CVFrame:
    """Copy and validate values/Jacobians, remapping every atom by global ID."""

    def __init__(self, schema, geometry, results):
        self.schema = schema
        self.geometry_sha256 = geometry_identity(geometry)
        self.atom_ids = geometry.atom_ids
        names = tuple(s.name for s in schema.variables)
        if not isinstance(results, dict) or set(results) != set(names):
            raise ValueError("CV results must match every declared name exactly")
        values, jacobians = [], []
        for spec in schema.variables:
            row = results[spec.name]
            fields = {"value", "unit", "atom_ids", "jacobian", "geometry_sha256", "dependencies"}
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("CV results require value, unit, full Jacobian and same-frame identities")
            if (row["geometry_sha256"] != self.geometry_sha256
                    or row["dependencies"] != spec.dependencies):
                raise ValueError("CV geometry or dependency identity differs")
            ids = _atom_ids(row["atom_ids"])
            if set(ids) != set(self.atom_ids):
                raise ValueError("CV Jacobian must explicitly cover every global atom ID")
            factor = UNITS[spec.unit].get(row["unit"])
            if factor is None:
                raise ValueError("CV input unit is incompatible with its declared unit")
            if type(row["value"]) not in (int, float) or not np.isfinite(row["value"]):
                raise ValueError("CV value must be a finite scalar")
            jac = _readonly_array(row["jacobian"], "CV Jacobian")
            if jac.shape != (len(ids), 3):
                raise ValueError("CV Jacobian must have shape (all atoms, 3)")
            lookup = dict(zip(ids, jac, strict=True))
            values.append(row["value"] * factor)
            jacobians.append([lookup[i] * factor for i in self.atom_ids])
        self.values = _readonly_array(values, "CV values")
        self.jacobian = _readonly_array(jacobians, "CV Jacobians")
        self.sha256 = digest(self.record())

    def record(self):
        return {"schema_sha256": self.schema.sha256, "geometry_sha256": self.geometry_sha256,
                "atom_ids": list(self.atom_ids), "values": self.values.tolist(),
                "jacobian_per_angstrom": self.jacobian.tolist()}

    def pullback(self, derivatives):
        """Return -J^T dB/dx; the caller adds this force to its total once."""
        if not isinstance(derivatives, dict) or set(derivatives) != {s.name for s in self.schema.variables}:
            raise ValueError("Bias derivatives must match all named CVs")
        values = _readonly_array([derivatives[s.name] for s in self.schema.variables], "Bias derivatives")
        if values.shape != self.values.shape:
            raise ValueError("Each bias derivative must be scalar")
        return -np.einsum("a,aij->ij", values, self.jacobian)


def collective_evaluate(bias, frame, owned_ids, step, comm, *, accept=False):
    """Root owns PLUMED; all ranks receive only their owned Cartesian forces.

    Every rank supplies the same full CVFrame and its disjoint native ownership.
    Validation errors are exchanged before evaluation, so no rank waits on a
    failed root. Native ghosts must not be included in owned_ids.
    """
    try:
        owned = tuple(owned_ids)
        if owned:
            owned = _atom_ids(owned)
        if not set(owned) <= set(frame.atom_ids):
            raise ValueError("Owned IDs are outside the CV geometry")
        packet = (None, frame.sha256, frame.atom_ids, owned, step, accept)
    except Exception as exc:  # noqa: BLE001 - Collective error propagation.
        packet = (str(exc), None, (), (), None, None)
    packets = comm.allgather(packet)
    error = next((p[0] for p in packets if p[0] is not None), None)
    if error is None:
        owners = [a for p in packets for a in p[3]]
        if (any((p[1], p[2], p[4], p[5]) != (packets[0][1], packets[0][2], step, accept) for p in packets)
                or sorted(owners) != sorted(frame.atom_ids)):
            error = "MPI CV frames/steps differ or atom ownership is missing/duplicate"
    if error is not None:
        raise ValueError(error)
    result = None
    if comm.rank == 0:
        try:
            result = (None, (bias.accept if accept else bias.evaluate)(frame, step))
        except Exception as exc:  # noqa: BLE001 - Collective error propagation.
            result = (f"{type(exc).__name__}: {exc}", None)
    error, result = comm.bcast(result, root=0)
    if error is not None:
        raise ValueError(error)
    index = {a: i for i, a in enumerate(frame.atom_ids)}
    return result, np.asarray([result["forces_eV_per_angstrom"][index[a]] for a in owned]).reshape(-1, 3)
