#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Independent chain-rule and rejection checks for named scalar CVs."""

from copy import deepcopy

import numpy as np
import pytest

from qmdpa.core import Geometry
from qmdpa.named_cv import CVFrame, HARTREE_EV, NamedCVSchema, geometry_identity


def declaration():
    return {"schema_version": 1, "coordinate_unit": "angstrom", "variables": [
        {"name": "gap", "shape": [], "unit": "eV", "dependencies": {"states": "a"*64}},
        {"name": "position", "shape": [], "unit": "angstrom", "dependencies": {"model": "b"*64}}]}


def model(positions, ids=(17, 4, 91), unit="eV"):
    schema = NamedCVSchema(declaration())
    geo = Geometry(ids, ("H",)*3, positions)
    x = geo.positions
    # An analytic state difference and an external-field coordinate, with a
    # derivative on the third (environment-like) atom. No QM accuracy claim.
    value = float(x[0] @ x[0] - 0.7*x[1] @ x[1] + 0.3*x[2, 0])
    jac = np.array([2*x[0], -1.4*x[1], [0.3, 0., 0.]])
    factor = HARTREE_EV if unit == "hartree" else 1.
    results = {
        "gap": {"value": value/factor, "unit": unit, "jacobian": (jac[::-1]/factor).tolist(),
                "atom_ids": list(ids[::-1]), "geometry_sha256": geometry_identity(geo),
                "dependencies": {"states": "a"*64}},
        "position": {"value": float(x[2, 1]-x[0, 1]), "unit": "angstrom",
                     "jacobian": [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
                     "atom_ids": list(ids), "geometry_sha256": geometry_identity(geo),
                     "dependencies": {"model": "b"*64}}}
    return schema, geo, results


@pytest.mark.parametrize("shift", [-2., 0., 2.])
@pytest.mark.parametrize("unit", ["eV", "hartree"])
def test_full_force_chain_rule_and_units(shift, unit):
    xyz = np.array([[0.2, 0.1, -0.3], [1.+shift, -0.1, 0.4], [2., 0.8, -0.2]])
    frame = CVFrame(*model(xyz, unit=unit))
    center, kappa = np.array([0.4, 0.1]), np.array([1.2, 0.7])
    derivative = kappa*(frame.values-center)
    forces = frame.pullback(dict(zip(("gap", "position"), derivative, strict=True)))
    for h in (1e-4, 5e-5):
        numerical = np.zeros_like(xyz)
        for i, axis in np.ndindex(xyz.shape):
            energies = []
            for sign in (-1, 1):
                displaced = xyz.copy()
                displaced[i, axis] += sign*h
                cv = CVFrame(*model(displaced, unit=unit)).values
                energies.append(float(0.5*np.sum(kappa*(cv-center)**2)))
            numerical[i, axis] = -(energies[1]-energies[0])/(2*h)
        assert np.max(abs(forces-numerical)) < 2e-7
    assert np.linalg.norm(forces[2]) > 0


@pytest.mark.parametrize("mutation", ["version", "unit", "shape", "duplicate", "unknown", "dependency"])
def test_schema_rejection(mutation):
    raw = declaration()
    if mutation == "version": raw["schema_version"] = True
    if mutation == "unit": raw["variables"][0]["unit"] = "kcal"
    if mutation == "shape": raw["variables"][0]["shape"] = [1]
    if mutation == "duplicate": raw["variables"][1]["name"] = "gap"
    if mutation == "unknown": raw["extra"] = 1
    if mutation == "dependency": raw["variables"][0]["dependencies"] = {"old": "no"}
    with pytest.raises(ValueError): NamedCVSchema(raw)


@pytest.mark.parametrize("mutation", ["stale", "state", "missing", "duplicate", "nan", "shape", "unit"])
def test_result_rejection(mutation):
    schema, geo, results = model(np.eye(3))
    row = results["gap"]
    if mutation == "stale": row["geometry_sha256"] = "f"*64
    if mutation == "state": row["dependencies"]["states"] = "c"*64
    if mutation == "missing": row["atom_ids"].pop(); row["jacobian"].pop()
    if mutation == "duplicate": row["atom_ids"][0] = row["atom_ids"][1]
    if mutation == "nan": row["jacobian"][0][0] = float("nan")
    if mutation == "shape": row["jacobian"] = [1., 2., 3.]
    if mutation == "unit": row["unit"] = "angstrom"
    with pytest.raises(ValueError): CVFrame(schema, geo, results)


def test_defensive_copy_and_geometry_reordering():
    schema, geo, results = model(np.eye(3))
    frame = CVFrame(schema, geo, results)
    results["gap"]["jacobian"][0][0] = 99
    assert frame.jacobian[0, 2, 0] == 0.3
    with pytest.raises(ValueError): frame.values.setflags(write=True)
    reverse = Geometry(geo.atom_ids[::-1], geo.symbols[::-1], geo.positions[::-1])
    assert geometry_identity(reverse) == geometry_identity(geo)
    with pytest.raises(ValueError): frame.pullback({"gap": 1})


def test_legacy_named_water_widths_are_identical(tmp_path):
    from test_opes import configured
    from qmdpa.sampling import bind_sampling, render_plumed
    config, raw = configured(tmp_path)
    expected = deepcopy(config)
    bind_sampling(expected, raw, tmp_path)
    for name in ("sigma", "sigma_min"):
        raw["bias"][name] = dict(zip(("logdistance", "ionization"), raw["bias"][name], strict=True))
    bind_sampling(config, raw, tmp_path)
    assert config == expected
    assert render_plumed(config["sampling"], tmp_path) == render_plumed(expected["sampling"], tmp_path)
    raw["bias"]["sigma"]["unknown"] = 1.
    with pytest.raises(ValueError): bind_sampling(config, raw, tmp_path)
