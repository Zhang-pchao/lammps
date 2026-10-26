#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Integer reactive ownership, complete smooth weights and conservative forces."""

from dataclasses import replace

import numpy as np
import pytest

from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.reactive_mechanical import ReactiveWaterGuard, ReactiveWaterMechanicalMixer
from qmdpa.reactive_water import ProtonChannel, ReactiveWaterFamily
from qmdpa.selection_config import FixedGroup


def geometry(transfers=(0., 0.), waters=4):
    xyz = []
    for i in range(waters):
        oxygen = 2.8*i
        xyz.extend([[oxygen, 0., 0.], [oxygen+.96, 0., 0.], [oxygen-.24, .93, .1]])
    xyz = np.asarray(xyz)
    xyz[1, 0] += .88*transfers[0]
    xyz[4, 0] += .88*transfers[1]
    return Geometry(tuple(range(1, 3*waters+1)), ("O", "H", "H")*waters, xyz)


def family(*, waters=4, radii=(.3, .8), **changes):
    groups = tuple(FixedGroup(f"w{i+1}", (3*i+1, 3*i+2, 3*i+3), ("O", "H", "H"))
                   for i in range(waters))
    args = {"groups": groups, "pool_group_ids": ("w1", "w2", "w3"), "seed_group_id": "w1",
            "channels": (ProtonChannel(2, "w1", "w2"), ProtonChannel(5, "w2", "w3")),
            "inner_radius_angstrom": radii[0], "outer_radius_angstrom": radii[1],
            "transfer_inner_angstrom": .2, "transfer_outer_angstrom": .7}
    return ReactiveWaterFamily(**dict(args, **changes))


class PairModel:
    charge = 0
    multiplicity = 1
    charge_spin = (0, 1)

    def __init__(self, k, water_zero=0.):
        self.k, self.water_zero = k, water_zero

    def evaluate(self, geom):
        energy, force = 0., np.zeros_like(geom.positions)
        for i in range(len(geom.atom_ids)):
            for j in range(i):
                delta = geom.positions[i]-geom.positions[j]
                energy += .5*self.k*np.dot(delta, delta)
                force[i] -= self.k*delta
                force[j] += self.k*delta
        energy += geom.symbols.count("O")*self.water_zero
        return Evaluation(energy, force)


def mixer(fam=None, zero=0.):
    low, high = PairModel(.01), PairModel(.03, zero)
    return ReactiveWaterMechanicalMixer(
        low, SubtractiveCalculator(low, high), family=family() if fam is None else fam,
        guard=ReactiveWaterGuard(.4, 1.6, 1.6), reference_delta_ev=.02+zero,
        reference_source="analytic-reference", method_identity="pair-test",
        energy_reference="declared-water-offset")


@pytest.mark.parametrize("transfers,topology,charges,selected", [
    ((0., 0.), (0, 0), (0, 0, 0, 0), (1, 2, 3)),
    ((1., 0.), (1, 0), (-1, 1, 0, 0), (1, 2, 3, 4, 5, 6)),
    ((1., 1.), (1, 1), (-1, 0, 1, 0), (1, 3, 5, 7, 8, 9)),
])
def test_actual_post_transfer_group_and_integer_electrons(transfers, topology, charges, selected):
    fam = family()
    result = fam.evaluate(geometry(transfers))
    active = [p for p, w in zip(fam.partitions, result.weights, strict=True) if w > 0]
    assert len(active) == 1
    partition = active[0]
    assert partition.topology == topology
    assert tuple(q for _, q in partition.group_charges) == charges
    assert partition.qm_atom_ids == selected
    assert partition.electron_count == 10*len(partition.qm_group_ids)
    assert sum(dict(partition.group_charges)[g] for g in partition.qm_group_ids) == 0
    assert all(q == 0 for g, q in partition.group_charges if g not in partition.qm_group_ids)


@pytest.mark.parametrize("transfers", [(.2, .0), (.5, .5), (1., .8), (1., 1.)])
def test_complete_weights_and_all_cartesian_derivatives(transfers):
    fam, geom = family(), geometry(transfers)
    weights = fam.evaluate(geom)
    assert sum(weights.weights) == pytest.approx(1., abs=1e-13)
    assert np.max(np.abs(weights.gradients_per_angstrom.sum(axis=0))) < 1e-12
    for h in (1e-5, 5e-6):
        for atom in range(len(geom.atom_ids)):
            for axis in range(3):
                shifted = []
                for sign in (-1, 1):
                    xyz = geom.positions.copy()
                    xyz[atom, axis] += sign*h
                    shifted.append(fam.evaluate(replace(geom, positions=xyz)).weights)
                fd = (shifted[1]-shifted[0])/(2*h)
                assert np.max(np.abs(fd-weights.gradients_per_angstrom[:, atom, axis])) <= 2e-7
    for partition, w in zip(fam.partitions, weights.weights, strict=True):
        if w == 0:
            continue
        for channel, probability in zip(fam.channels, weights.ownership_probabilities, strict=True):
            if 0 < probability < 1:
                assert {channel.donor_group_id, channel.acceptor_group_id} <= set(partition.qm_group_ids)


@pytest.mark.parametrize("transfers", [(.2, 0.), (.5, .5), (1., .8), (1., 1.)])
def test_conservative_total_force_and_energy_zero(transfers):
    calculator, geom = mixer(), geometry(transfers)
    result = calculator.evaluate(geom)
    for h in (1e-5, 5e-6):
        for atom in range(len(geom.atom_ids)):
            for axis in range(3):
                energies = []
                for sign in (-1, 1):
                    xyz = geom.positions.copy()
                    xyz[atom, axis] += sign*h
                    energies.append(calculator.evaluate(replace(geom, positions=xyz)).energy)
                assert abs(-(energies[1]-energies[0])/(2*h)-result.forces[atom, axis]) < 2e-7
    shifted = mixer(zero=3.4).evaluate(geom)
    assert shifted.energy-result.energy == pytest.approx(3.4, abs=1e-11)
    np.testing.assert_allclose(shifted.forces, result.forces, rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0., rtol=0, atol=1e-11)
    reversed_geom = geom.select(tuple(reversed(geom.atom_ids)))
    reversed_result = calculator.evaluate(reversed_geom)
    assert reversed_result.energy == pytest.approx(result.energy, abs=1e-11)
    np.testing.assert_allclose(reversed_result.forces[::-1], result.forces, rtol=0, atol=1e-11)


def test_all_qm_cancellation_and_identical_fragment_reuse():
    fam, geom = family(waters=3, radii=(10., 11.)), geometry((.4, .5), waters=3)
    calculator = mixer(fam)
    terms = calculator.evaluate_terms(geom)
    expected = PairModel(.03).evaluate(geom)
    assert terms.total.energy == pytest.approx(expected.energy-2*.02, abs=1e-12)
    np.testing.assert_allclose(terms.total.forces, expected.forces, rtol=0, atol=1e-12)
    assert len(terms.evaluated_fragments) == 1


@pytest.mark.parametrize("changes", [
    {"channels": (ProtonChannel(2, "w1", "w2"), ProtonChannel(2, "w1", "w3"))},
    {"channels": (ProtonChannel(3, "w2", "w3"),)},
    {"channels": (ProtonChannel(2, "w1", "w2"), ProtonChannel(3, "w1", "w3"))},
    {"pool_group_ids": ("w1", "w2", "w2")},
    {"max_partitions": 1}, {"transfer_inner_angstrom": .8},
])
def test_reject_incomplete_or_invalid_ownership_family(changes):
    with pytest.raises((ValueError, TypeError)):
        family(**changes)


def test_guard_rejects_undeclared_transfer_and_lost_water():
    geom = geometry()
    xyz = geom.positions.copy()
    xyz[2] = xyz[3]+[.9, 0., 0.]
    with pytest.raises(EvaluationError, match="ownership domain|undeclared"):
        mixer().evaluate(replace(geom, positions=xyz))
    with pytest.raises(ValueError, match="finite"):
        family().evaluate(replace(geom, cell=np.eye(3)*20))
