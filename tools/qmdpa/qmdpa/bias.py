#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Optional static distance restraints using the official native spring force."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

FORM = "harmonic-distance"
_MODES = {
    "fixed-region-mechanical",
    "finite-electrostatic-additive",
    "finite-electrostatic-gap-anchor",
}
# The native spring clips shorter distances; reject that nondifferentiable domain.
MIN_DISTANCE_ANGSTROM = 1e-10


def _scalar(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite" + (" positive" if positive else "") + " number")
    return float(value)


def normalize_bias(raw, config):
    """Validate explicit units and supported assembly without evaluating forces."""
    if config["hamiltonian"] not in _MODES or config["periodic"]:
        raise ValueError("Static bias requires finite fixed-region ME, additive EE or gap-anchor")
    if not isinstance(raw, dict) or set(raw) != {"form", "restraints"} or raw["form"] != FORM:
        raise ValueError("bias requires form harmonic-distance and restraints")
    rows = raw["restraints"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("bias.restraints must be a nonempty list")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "atom_ids", "k_ev_per_angstrom2", "distance_angstrom"
        }:
            raise ValueError("Each restraint requires atom_ids, k_ev_per_angstrom2 and distance_angstrom")
        ids = row["atom_ids"]
        if (not isinstance(ids, list) or len(ids) != 2
                or any(type(i) is not int or i <= 0 for i in ids) or ids[0] == ids[1]):
            raise ValueError("A restraint requires two distinct positive integer atom IDs")
        ids = sorted(ids)
        if tuple(ids) in seen:
            raise ValueError("Duplicate distance restraints are not supported")
        seen.add(tuple(ids))
        result.append({
            "atom_ids": ids,
            "k_ev_per_angstrom2": _scalar(row["k_ev_per_angstrom2"], "Restraint k", positive=True),
            "distance_angstrom": _scalar(row["distance_angstrom"], "Restraint distance", positive=True),
        })
    # One native single-atom group per ID plus the built-in all group.
    if len({i for row in result for i in row["atom_ids"]}) > 31:
        raise ValueError("Static bias supports at most 31 distinct restrained atoms")
    return {"form": FORM, "restraints": sorted(result, key=lambda row: row["atom_ids"])}


def bind_bias(config, raw):
    """Extend the existing Hamiltonian identity; never hash only the bias."""
    bias = normalize_bias(raw, config)
    base = config["physics_sha256"]
    config["bias"] = bias
    config["unbiased_physics_sha256"] = base
    config["physics_sha256"] = hashlib.sha256(json.dumps(
        {"unbiased_physics_sha256": base, "bias": bias}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return config


def describe_bias(bias):
    return {
        "configuration": bias,
        "potential": "sum(0.5*k*(distance-r0)^2)",
        "native_style": "spring couple; one atom per group; energy yes",
        "force_coordinates": "both restrained atoms",
        "scope": "static finite-distance bias; no free-energy or rate qualification",
    }


def _coordinates(rows, expected):
    coordinates = {}
    for atom_id, position in rows:
        if atom_id in coordinates:
            raise ValueError("Bias atom IDs have multiple owners")
        xyz = np.asarray(position, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Bias coordinates must be finite three-vectors")
        coordinates[atom_id] = xyz
    if set(coordinates) != expected:
        raise ValueError("Bias atom IDs are missing from the current complete system")
    return coordinates


def _distances(bias, coordinates):
    values = []
    for row in bias["restraints"]:
        first, second = row["atom_ids"]
        distance = float(np.linalg.norm(coordinates[first] - coordinates[second]))
        if not math.isfinite(distance) or distance <= MIN_DISTANCE_ANGSTROM:
            raise ValueError("Bias distance must exceed 1e-10 Angstrom at every evaluated step")
        values.append(distance)
    return values


class NativeBiasAtoms:
    """Collect finite biased atoms within the existing callback collective."""

    def local_packet(self, nlocal, tags, positions):
        """Collect restrained IDs in the existing callback collective."""
        if not nlocal:
            return []
        tags, xyz = np.asarray(tags), np.asarray(positions)
        if tags.shape != (nlocal,) or tags.dtype.kind not in "iu" or xyz.shape != (nlocal, 3):
            raise ValueError("Bias requires valid owned atom IDs and coordinates")
        images = self.lmp.numpy.extract_atom("image")
        rows = []
        for index, atom_id in enumerate(tags):
            if atom_id in self.ids:
                if any(self.lmp.decode_image_flags(int(images[index]))):
                    raise ValueError("Nonperiodic bias atoms must have zero image flags")
                rows.append((int(atom_id), xyz[index].tolist()))
        return rows


class NativeDistanceBias(NativeBiasAtoms):
    """Configure native forces and observe native values; do not compute a force."""

    def __init__(self, lmp, config):
        self.lmp = lmp
        self.config = config
        self.bias = normalize_bias(config["bias"], config)
        self.ids = {i for row in self.bias["restraints"] for i in row["atom_ids"]}
        if lmp.extract_global("units") != "metal" or any(lmp.extract_box()[5]):
            raise ValueError("Static bias requires metal units and nonperiodic native boundaries")
        if not lmp.has_style("fix", "spring"):
            raise ValueError("The loaded LAMMPS library has no spring fix")
        for atom_id in sorted(self.ids):
            group = f"qmdpa_bias_atom_{atom_id}"
            # Binary restart retains groups, but not spring fixes. Rebuild membership.
            if lmp.has_id("group", group):
                lmp.command(f"group {group} clear")
            lmp.command(f"group {group} id {atom_id}")
        self.fix_ids = []
        for index, row in enumerate(self.bias["restraints"]):
            first, second = row["atom_ids"]
            fix = f"qmdpa_bias_{index}"
            if lmp.has_id("fix", fix):
                raise ValueError("A native bias fix is already installed")
            lmp.command(
                f"fix {fix} qmdpa_bias_atom_{first} spring couple qmdpa_bias_atom_{second} "
                f"{row['k_ev_per_angstrom2']:.17g} 0 0 0 {row['distance_angstrom']:.17g}"
            )
            lmp.command(f"fix_modify {fix} energy yes")
            self.fix_ids.append(fix)

    def validate_packets(self, packets):
        """Reject undefined distances before a quantum call or accepted state."""
        coordinates = _coordinates([row for packet in packets for row in packet], self.ids)
        _distances(self.bias, coordinates)

    def capture(self, frame, comm):
        """Read native scalar/vector values on all ranks, then record on the owner."""
        from .runtime import _root_call

        try:
            values = [[float(self.lmp.extract_fix(fix, 0, 0)),
                       [float(self.lmp.extract_fix(fix, 0, 1, axis)) for axis in range(3)]]
                      for fix in self.fix_ids]
            packet = (None, values)
        except Exception as exc:  # noqa: BLE001 - Preserve collective or top-level failure reporting.
            packet = (f"{type(exc).__name__}: {exc}", None)
        packets = [packet] if comm is None else comm.gather(packet, root=0)

        def attach():
            errors = [error for error, _ in packets if error is not None]
            if errors:
                raise ValueError("; ".join(errors))
            values = packets[0][1]
            if any(other != values for _, other in packets[1:]):
                raise ValueError("Native bias records differ between MPI ranks")
            coordinates = _coordinates(
                [(atom["id"], atom["x"]) for atom in frame["atoms"] if atom["id"] in self.ids],
                self.ids,
            )
            distances = _distances(self.bias, coordinates)
            energy = sum(value[0] for value in values)
            frame["bias"] = {
                "form": FORM,
                "energy_eV": energy,
                "unbiased_potential_eV": frame["potential_eV"] - energy,
                "restraints": [
                    {"atom_ids": row["atom_ids"], "distance_angstrom": distance,
                     "energy_eV": value[0], "force_on_first_eV_per_angstrom": value[1]}
                    for row, distance, value in zip(self.bias["restraints"], distances, values)
                ],
            }
            validate_bias_frame(frame, self.config)

        _root_call(comm, attach)


def validate_bias_frame(frame, config):
    """Require bias identity and complete energy accounting in accepted records."""
    expected, record = config.get("bias"), frame.get("bias")
    if expected is None:
        if record is not None:
            raise ValueError("An unbiased run cannot accept a bias frame")
        return
    if not isinstance(record, dict) or set(record) != {
        "form", "energy_eV", "unbiased_potential_eV", "restraints"
    } or record["form"] != FORM:
        raise ValueError("An accepted biased frame requires its complete bias record")
    rows = record["restraints"]
    if not isinstance(rows, list) or len(rows) != len(expected["restraints"]):
        raise ValueError("Bias frame restraint count differs")
    ids = {i for row in expected["restraints"] for i in row["atom_ids"]}
    atoms = [atom for atom in frame["atoms"] if atom["id"] in ids]
    if any(any(atom["image"]) for atom in atoms):
        raise ValueError("Nonperiodic bias atoms must have zero image flags")
    coordinates = _coordinates([(atom["id"], atom["x"]) for atom in atoms], ids)
    distances = _distances(expected, coordinates)
    energies = []
    for row, specification, distance in zip(rows, expected["restraints"], distances):
        if not isinstance(row, dict) or set(row) != {
            "atom_ids", "distance_angstrom", "energy_eV", "force_on_first_eV_per_angstrom"
        } or row["atom_ids"] != specification["atom_ids"]:
            raise ValueError("Bias frame atom identity differs")
        observed = _scalar(row["distance_angstrom"], "Bias frame distance", positive=True)
        if not math.isclose(observed, distance, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Bias frame distance differs from coordinates")
        energy = _scalar(row["energy_eV"], "Bias energy")
        if energy < 0:
            raise ValueError("Harmonic bias energy cannot be negative")
        force = row["force_on_first_eV_per_angstrom"]
        if not isinstance(force, list) or len(force) != 3:
            raise ValueError("Bias frame force must be a finite three-vector")
        for component in force:
            _scalar(component, "Bias force")
        energies.append(energy)
    total = _scalar(record["energy_eV"], "Total bias energy")
    unbiased = _scalar(record["unbiased_potential_eV"], "Unbiased potential")
    if not math.isclose(total, sum(energies), abs_tol=1e-10, rel_tol=0):
        raise ValueError("Bias energy sum differs from individual restraints")
    if not math.isclose(frame["potential_eV"], unbiased + total, abs_tol=1e-7, rel_tol=0):
        raise ValueError("Biased and unbiased potential energies are inconsistent")


def compare_bias_restart(frame, reference, config):
    for item in (frame, reference):
        validate_bias_frame(item, config)
    if config.get("bias") is None:
        return
    for name in ("energy_eV", "unbiased_potential_eV"):
        if abs(frame["bias"][name] - reference["bias"][name]) > 1e-6:
            raise ValueError("Restart bias energy differs by more than 1e-6 eV")
    for current, previous in zip(frame["bias"]["restraints"], reference["bias"]["restraints"]):
        if abs(current["energy_eV"] - previous["energy_eV"]) > 1e-6:
            raise ValueError("Restart restraint energy differs by more than 1e-6 eV")
        if not np.allclose(current["force_on_first_eV_per_angstrom"],
                           previous["force_on_first_eV_per_angstrom"], atol=1e-5, rtol=0):
            raise ValueError("Restart bias force differs by more than 1e-5 eV/Angstrom")
