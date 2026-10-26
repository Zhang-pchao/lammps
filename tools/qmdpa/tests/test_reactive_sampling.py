#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Reactive OPES uses a fixed water pool across integer proton reassignment."""

from copy import deepcopy

import pytest
from test_reactive_config import (
    document,
    frame,
    load,
    numerical_versions,  # noqa: F401 - Reuse the package-identity fixture.
)
from test_sampling import request, thermostat

from qmdpa import opes, water_cv
from qmdpa.dispatch import describe_run
from qmdpa.sampling import render_plumed


def sampling(tmp_path):
    raw = request(tmp_path)
    raw["cv"] = {"form": water_cv.FORM, "oxygen_ids": [1, 4, 7],
                 "hydrogen_ids": [2, 3, 5, 6, 8, 9],
                 "coordination_kappa_per_angstrom": 7., "distance_kappa_per_angstrom": 8.}
    raw["bias"] = {"form": opes.FORM, "sigma": [.15, .04], "sigma_min": [.06, .01],
                   "barrier_ev": .1, "bias_factor": 5., "pace_steps": 2}
    return raw


def configured(tmp_path):
    raw = document()
    raw.update(steps=8, sample_interval=2, thermostat=thermostat(), sampling=sampling(tmp_path))
    return raw


def test_unified_reactive_nvt_and_complete_pool_opes(tmp_path):
    base = load(tmp_path)
    raw = configured(tmp_path)
    cfg = load(tmp_path, raw)
    assert cfg["physics_sha256"] != base["physics_sha256"]
    assert cfg["sampling"]["cv"]["atom_ids"] == list(range(1, 10))
    view = describe_run(cfg)
    assert view["assembly"]["native_pair"] == "zero"
    assert view["assembly"]["additional_native_bias"] == "plumed"
    assert "OPES_METAD" in render_plumed(cfg["sampling"], tmp_path)
    active = [p for p in frame(cfg)["reactive_water"]["partitions"] if p["weight"] > 0]
    assert len(active) == 1 and 4 not in active[0]["qm_atom_ids"]
    assert 4 in cfg["sampling"]["cv"]["atom_ids"]
    raw["sampling"]["cv"]["hydrogen_ids"].reverse()
    assert load(tmp_path, raw)["physics_sha256"] == cfg["physics_sha256"]


@pytest.mark.parametrize("case", ["subset", "environment", "swapped", "distance", "harmonic"])
def test_reactive_sampling_domain_and_combination_rejections(tmp_path, case):
    raw = configured(tmp_path)
    cv = raw["sampling"]["cv"]
    if case == "subset":
        cv.update(oxygen_ids=[1, 4], hydrogen_ids=[2, 3, 5, 6])
    elif case == "environment":
        cv["oxygen_ids"].append(10)
        cv["hydrogen_ids"] += [11, 12]
    elif case == "swapped":
        cv["oxygen_ids"][0], cv["hydrogen_ids"][0] = cv["hydrogen_ids"][0], cv["oxygen_ids"][0]
    elif case == "distance":
        raw["sampling"]["cv"] = {"form": "distance", "atom_ids": [1, 2]}
    else:
        raw["sampling"]["bias"] = {"form": "harmonic", "center_angstrom": 1., "k_ev_per_angstrom2": .1}
    with pytest.raises(ValueError, match="pool|O/H|Reactive-water sampling"):
        load(tmp_path, raw)


def test_nvt_without_bias_and_bias_identity_are_separate(tmp_path):
    raw = configured(tmp_path)
    cfg = load(tmp_path, raw)
    no_bias = deepcopy(raw)
    no_bias.pop("sampling")
    nvt = load(tmp_path, no_bias)
    assert "thermostat" in nvt and "sampling" not in nvt
    assert nvt["physics_sha256"] != cfg["physics_sha256"]
    raw["sampling"]["bias"]["barrier_ev"] *= 2
    assert load(tmp_path, raw)["physics_sha256"] != cfg["physics_sha256"]
