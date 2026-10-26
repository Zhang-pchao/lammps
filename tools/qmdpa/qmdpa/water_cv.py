#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Finite water soft-Voronoi configuration and independent scalar observations.

PLUMED owns all production derivatives. These scalar observations validate
geometry and restart records; smooth occupancies are not integer QM charges.
"""

import math

import numpy as np

from .bias import MIN_DISTANCE_ANGSTROM
from .dynamics import positive_scalar

FORM = "water-self-ionization-voronoi"
LABELS = ("qmdpa_logdistance", "qmdpa_ionization")
OBSERVABLES = ("ionization", "iondistance_angstrom", "logdistance")


def validate_cv(raw, config):
    if set(raw) != {"form", "oxygen_ids", "hydrogen_ids",
                    "coordination_kappa_per_angstrom", "distance_kappa_per_angstrom"}:
        raise ValueError("Water Voronoi CV requires explicit oxygen/hydrogen IDs and both kappa values")
    oxygen, hydrogen = raw["oxygen_ids"], raw["hydrogen_ids"]
    for ids in (oxygen, hydrogen):
        if (not isinstance(ids, list) or any(type(i) is not int or i < 1 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Water CV selections require distinct positive integer atom IDs")
    if len(oxygen) < 2 or len(hydrogen) != 2 * len(oxygen) or set(oxygen) & set(hydrogen):
        raise ValueError("Water CV requires disjoint neutral O/H selections with at least two oxygens")
    symbols = dict(zip(config["region"]["atom_ids"], config["region"]["symbols"], strict=True))
    if any(symbols.get(i) != "O" for i in oxygen) or any(symbols.get(i) != "H" for i in hydrogen):
        raise ValueError("Water CV selections must match O/H atoms inside the fixed QM region")
    return {
        "form": FORM, "oxygen_ids": sorted(oxygen), "hydrogen_ids": sorted(hydrogen),
        "atom_ids": sorted(oxygen + hydrogen),
        **{name: positive_scalar(raw[name], "sampling.cv." + name) for name in
           ("coordination_kappa_per_angstrom", "distance_kappa_per_angstrom")},
    }


def log_distance(value):
    """C1 classical transform; the numeric argument is distance in Angstrom."""
    if not math.isfinite(value) or value <= -0.1:
        raise ValueError("Water log-distance is outside its finite logarithm domain")
    return 1.1 * math.log((value + 0.1) / 1.1) if value <= 1.0 else value - 1.0


def values(cv, coordinates):
    oxygen = np.asarray([coordinates[i] for i in cv["oxygen_ids"]])
    hydrogen = np.asarray([coordinates[i] for i in cv["hydrogen_ids"]])
    distances = np.linalg.norm(oxygen[:, None, :] - hydrogen[None, :, :], axis=2)
    oo = np.linalg.norm(oxygen[:, None, :] - oxygen[None, :, :], axis=2)
    pair_indices = np.triu_indices(len(oxygen), 1)
    if (not np.isfinite(distances).all() or not np.isfinite(oo).all()
            or np.any(distances <= MIN_DISTANCE_ANGSTROM)
            or np.any(oo[pair_indices] <= MIN_DISTANCE_ANGSTROM)):
        raise ValueError("Water Voronoi distances must exceed 1e-10 Angstrom")

    def defects(kappa):
        # Shift by the smallest distance to avoid overflow and underflow.
        weights = np.exp(-kappa * (distances - distances.min(axis=0)))
        return (weights / weights.sum(axis=0)).sum(axis=1) - 2.0

    qc = defects(cv["coordination_kappa_per_angstrom"])
    qd = defects(cv["distance_kappa_per_angstrom"])
    ionization = float(np.sum(qc**2))
    distance = float(-np.sum((oo * qd[:, None] * qd[None, :])[pair_indices]))
    return {"ionization": ionization, "iondistance_angstrom": distance,
            "logdistance": log_distance(distance)}


def render(cv):
    oxygen = ",".join(map(str, cv["oxygen_ids"]))
    hydrogen = ",".join(map(str, cv["hydrogen_ids"]))
    return [
        f"qmdpa_water_o: GROUP ATOMS={oxygen}",
        f"qmdpa_water_h: GROUP ATOMS={hydrogen}",
        ("qmdpa_ionization: VORONOI_COORDINATION CENTERS=qmdpa_water_o "
         "ASSIGNED=qmdpa_water_h REFERENCE=2 SELECT=qmdpa_water_o POWER=2 NOPBC "
         f"KAPPA={cv['coordination_kappa_per_angstrom']:.17g}"),
        ("qmdpa_iondistance: VORONOI_DISTANCE CENTERS=qmdpa_water_o "
         "ASSIGNED=qmdpa_water_h REFERENCE=2 GROUP1=qmdpa_water_o NOPBC "
         f"KAPPA={cv['distance_kappa_per_angstrom']:.17g}"),
        ("qmdpa_logdistance: CUSTOM ARG=qmdpa_iondistance "
         "FUNC=select(step(1.0-x),1.10*log((x+0.10)/1.10),x-1.0) PERIODIC=NO"),
    ]
