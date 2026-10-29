#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Independent reactive additive derivatives, charge coverage and route identity."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from test_molecular_additive import AnalyticEmbedded
from test_reactive_config import (
    document as mechanical_document, frame, load,
    numerical_versions,  # noqa: F401 - Reuse package metadata without backend imports.
)
from test_reactive_sampling import sampling
from test_reactive_water import PairModel, family, geometry
from test_sampling import thermostat

from qmdpa.additive import CrossLennardJones
from qmdpa.dispatch import describe_run
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.reactive_additive import ReactiveWaterAdditiveMixer
from qmdpa.reactive_config import validate_reactive_frame
from qmdpa.reactive_mechanical import ReactiveWaterGuard
from qmdpa.reactive_water import ADDITIVE_HAMILTONIAN


class WaterEmbedded(AnalyticEmbedded):
    def __init__(self, charges, zero):
        super().__init__(charges)
        self.zero = zero

    def evaluate(self, qm, env):
        value = super().evaluate(qm, env)
        return replace(value, energy_ev=value.energy_ev + qm.symbols.count("O")*self.zero)


class EnvironmentOnly(PairModel):
    def __init__(self):
        super().__init__(.01)
        self.domains = []

    def evaluate(self, geom):
        self.domains.append(geom.atom_ids)
        assert len(geom.atom_ids) < 12
        return super().evaluate(geom)


def table():
    return FixedEmbeddingCharges(tuple(range(1, 13)), [-.8, .4, .4]*4, "analytic-water")


def mixer(zero=0., **changes):
    args = dict(family=family(), guard=ReactiveWaterGuard(.4, 1.6, 1.6), charges=table(),
                reference_delta_ev=.02+zero, reference_source="analytic-water-reference",
                method_identity="analytic-ground", energy_reference="water-difference")
    args.update(changes)
    return ReactiveWaterAdditiveMixer(
        EnvironmentOnly(), lambda field: WaterEmbedded(field, zero),
        CrossLennardJones({(a,b): (.0005, 1.) for a in ("O", "H") for b in ("O", "H")},
                         source="analytic-cross"), **args)


@pytest.mark.parametrize("transfers", [(.2, 0.), (.5, .5), (1., .8), (1., 1.)])
def test_every_coordinate_derivative_and_reference_offset(transfers):
    obj, geom = mixer(), geometry(transfers)
    terms = obj.evaluate_terms(geom)
    for h in (1e-5, 5e-6):
        for row in range(12):
            for axis in range(3):
                energies = []
                for sign in (-1, 1):
                    xyz = geom.positions.copy()
                    xyz[row, axis] += sign*h
                    energies.append(obj.evaluate(replace(geom, positions=xyz)).energy)
                assert abs(-(energies[1]-energies[0])/(2*h)-terms.total.forces[row,axis]) < 1e-5
    shifted = mixer(3.4).evaluate(geom)
    assert shifted.energy-terms.total.energy == pytest.approx(3.4, abs=1e-10)
    np.testing.assert_allclose(shifted.forces, terms.total.forces, atol=1e-10, rtol=0)
    np.testing.assert_allclose(terms.total.forces.sum(axis=0), 0, atol=1e-10, rtol=0)
    reversed_value = obj.evaluate(geom.select(tuple(reversed(geom.atom_ids))))
    np.testing.assert_allclose(reversed_value.forces[::-1], terms.total.forces, atol=1e-10, rtol=0)
    assert reversed_value.energy == pytest.approx(terms.total.energy, abs=1e-10)


def test_total_assembly_environment_feedback_and_shared_domain_reuse():
    obj, geom = mixer(), geometry((.2,0.))
    terms = obj.evaluate_terms(geom)
    weighted = sum(w*v.energy for w,v in zip(terms.weights.weights,terms.partition_totals)
                   if v is not None)
    assert terms.total.energy == pytest.approx(weighted, abs=1e-12)
    assert np.max(abs(terms.switching_forces)) > .01
    assert np.max(abs(terms.total.forces[9:])) > .01
    obj = mixer()
    terms = obj.evaluate_terms(geometry((.5,.5)))
    unique = {id(p.calculator):p.calculator for p in obj.partitions}
    assert sum(c.embedded_backend.calls for c in unique.values()) == len(
        {ids for ids,v in zip(terms.qm_atom_ids,terms.partition_totals) if v is not None})
    assert all(len(c.environment_backend.domains) >= 1 for c in unique.values())


def test_invalid_charge_and_provider_state_fail_before_evaluation():
    bad = np.array(table().charges_e)
    bad[1] += .01
    bad[2] -= .01
    with pytest.raises(ValueError, match="zero fixed charge"):
        mixer(charges=FixedEmbeddingCharges(table().atom_ids,bad,"invalid reassignment"))
    with pytest.raises(ValueError, match="cover"):
        mixer(charges=FixedEmbeddingCharges((1,),[0.],"incomplete"))
    with pytest.raises(ValueError, match="permanent environment"):
        mixer(family=family(waters=3),
              charges=FixedEmbeddingCharges(tuple(range(1,10)),[-.8,.4,.4]*3,"three-water"))
    obj = mixer()
    obj.partitions[-1].calculator.embedded_backend.charge = 1
    with pytest.raises(ValueError, match="charge/multiplicity"):
        obj.evaluate(geometry())
    assert all(not p.calculator.environment_backend.domains for p in obj.partitions)


def document():
    raw = mechanical_document()
    raw["hamiltonian"] = ADDITIVE_HAMILTONIAN
    raw.pop("full_charge_spin")
    raw.pop("region_charge_spin")
    raw["environment_charge_spin"] = [0,1]
    raw["embedding"] = {"atom_ids":list(range(1,13)), "charges_e":[-.8,.4,.4]*4,
                        "source":"analytic-water"}
    raw["cross"] = {"form":"lj12-6","source":"analytic-cross","cutoff":None,"mixing":"none",
                    "pairs":[{"qm_symbol":a,"environment_symbol":b,"epsilon_ev":.0005,
                              "sigma_angstrom":1.} for a in ("O","H") for b in ("O","H")]}
    return raw


def test_route_identity_fixed_pool_and_reactive_frame(tmp_path):
    raw = document()
    base = load(tmp_path,raw)
    view = describe_run(base)
    assert view["coupling"]["ee_model"] == "reactive-water-additive"
    assert view["assembly"]["native_pair"] == "zero"
    assert view["energy_contract"]["partition_terms"][0] == "low_environment"
    assert "full_charge_spin" not in base
    selected = deepcopy(raw)
    selected.pop("hamiltonian")
    selected["coupling"] = view["coupling"]
    assert load(tmp_path,selected)["physics_sha256"] == base["physics_sha256"]
    raw.update(steps=8,sample_interval=2,thermostat=thermostat(),sampling=sampling(tmp_path))
    cfg = load(tmp_path,raw)
    assert cfg["sampling"]["cv"]["atom_ids"] == list(range(1,10))
    record = frame(cfg)
    assert record["reactive_water"]["hamiltonian"] == ADDITIVE_HAMILTONIAN
    validate_reactive_frame(record,cfg)
    record["reactive_water"]["hamiltonian"] = "finite-reactive-water-mechanical"
    with pytest.raises(ValueError,match="grouping"):
        validate_reactive_frame(record,cfg)
    bad = deepcopy(raw)
    bad["sampling"]["cv"]["oxygen_ids"] = [1,4]
    bad["sampling"]["cv"]["hydrogen_ids"] = [2,3,5,6]
    with pytest.raises(ValueError,match="complete declared"):
        load(tmp_path,bad)
    reordered = deepcopy(raw)
    reordered["embedding"]["atom_ids"].reverse()
    reordered["embedding"]["charges_e"].reverse()
    assert load(tmp_path,reordered)["physics_sha256"] == cfg["physics_sha256"]
    changed = deepcopy(raw)
    changed["cross"]["pairs"][0]["epsilon_ev"] *= 2
    assert load(tmp_path,changed)["physics_sha256"] != cfg["physics_sha256"]


@pytest.mark.parametrize("case", ["state","coverage","net-charge","cross","baseline","compressed","isolated-solver"])
def test_unsupported_models_fail_at_admission(tmp_path,case):
    raw = document()
    if case == "state":
        raw["environment_charge_spin"] = [1,1]
    elif case == "coverage":
        raw["embedding"]["atom_ids"][0] = 99
    elif case == "net-charge":
        raw["embedding"]["charges_e"][9] = -.7
    elif case == "cross":
        raw["cross"]["pairs"].pop()
    elif case == "baseline":
        raw["full_charge_spin"] = [0,1]
    elif case == "isolated-solver":
        raw["qm"]["scf_solver"] = "diis-newton"
        raw["qm"]["conv_tol_grad"] = 1e-9
        raw["buffer_reference"]["qm"].update(scf_solver="diis-newton", conv_tol_grad=1e-9)
    else:
        raw["model"] = "model.pt2"
    with pytest.raises(ValueError, match="diis-newton" if case == "isolated-solver" else None):
        load(tmp_path,raw)
