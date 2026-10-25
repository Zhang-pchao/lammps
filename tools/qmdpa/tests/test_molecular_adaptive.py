# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Multiple complete-molecule exchange and complete conservative derivatives."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from test_adaptive import PairPolynomial, displaced

from qmdpa.adaptive import MechanicalPartition
from qmdpa.core import EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.molecular_adaptive import MolecularMechanicalMixer, QuinticMolecularWeights
from qmdpa.selection_config import FixedGroup


def groups():
    return tuple(FixedGroup(name, (2 * i + 1, 2 * i + 2), ("H", "H"))
                 for i, name in enumerate(("core", "a", "b", "c")))


def geometry():
    return Geometry(tuple(range(1, 9)), ("H",) * 8,
                    [[0, 0, 0], [0, 0, .74], [2.2, .1, 0], [2.2, .1, .74],
                     [.1, 2.6, .1], [.1, 2.6, .84], [-2.8, .2, .1], [-2.8, .2, .84]])


def switch(k=1):
    return QuinticMolecularWeights(1, (("c", 7), ("b", 5), ("a", 3)), k, 2., 3.)


def partitions(k=1):
    return tuple(MechanicalPartition(
        "".join(subset), ("core",) + subset,
        SubtractiveCalculator(PairPolynomial(.04), PairPolynomial(.07, -5)),
        0, 1, "analytic-ground-v1", "same-composition-shared-reference")
        for subset in combinations(("a", "b", "c"), k))


def mixer(k=1, **changes):
    args = dict(groups=groups(), core_group_ids=("core",), partitions=partitions(k), switch=switch(k))
    args.update(changes)
    return MolecularMechanicalMixer(PairPolynomial(.02, -10), **args)


@pytest.mark.parametrize("k", [1, 2])
def test_every_weight_derivative_including_normalization_and_center(k):
    policy, atoms = switch(k), geometry()
    value = policy.evaluate(atoms)
    assert value.group_subsets == tuple(combinations(("a", "b", "c"), k))
    np.testing.assert_allclose(value.weights.sum(), 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(value.gradients_per_angstrom.sum(axis=0), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(value.gradients_per_angstrom.sum(axis=1), 0, atol=1e-12, rtol=0)
    assert np.linalg.norm(value.gradients_per_angstrom[:, 0]) > .1
    h = 1e-6
    for i in range(8):
        for axis in range(3):
            fd = (policy.evaluate(displaced(atoms, i, axis, h)).weights
                  - policy.evaluate(displaced(atoms, i, axis, -h)).weights) / (2 * h)
            np.testing.assert_allclose(fd, value.gradients_per_angstrom[:, i, axis], atol=2e-8, rtol=0)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("h", [1e-4, 5e-5])
def test_all_atom_force_is_complete_energy_derivative(k, h):
    obj, atoms = mixer(k), geometry()
    t = obj.evaluate_terms(atoms)
    assert len(t.active_partition_ids) == 3
    assert np.max(np.abs(t.switching_forces)) > .01
    fd = np.zeros_like(atoms.positions)
    for i in range(8):
        for axis in range(3):
            energies = [obj.evaluate(displaced(atoms, i, axis, sign * h)).energy
                        for sign in (-2, -1, 1, 2)]
            fd[i, axis] = -(energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]) / (12 * h)
    np.testing.assert_allclose(t.total.forces, fd, atol=2e-7, rtol=0)
    np.testing.assert_allclose(t.total.forces.sum(axis=0), 0, atol=1e-12, rtol=0)
    wrong = t.total.forces - t.switching_forces
    assert np.max(np.abs(wrong - fd)) > .01


@pytest.mark.parametrize("k", [1, 2])
def test_common_offset_cancels_from_switch_force_and_relative_offset_does_not(k):
    atoms, obj = geometry(), mixer(k)
    base = obj.evaluate_terms(atoms)
    common = tuple(replace(p, energy_offset_ev=1000., offset_source="declared common shift")
                   for p in partitions(k))
    shifted = mixer(k, partitions=common).evaluate_terms(atoms)
    assert shifted.total.energy - base.total.energy == pytest.approx(1000., abs=1e-11)
    np.testing.assert_allclose(shifted.total.forces, base.total.forces, atol=5e-12, rtol=0)
    one = list(partitions(k))
    one[0] = replace(one[0], energy_offset_ev=2., offset_source="declared single-partition shift")
    changed = mixer(k, partitions=one).evaluate_terms(atoms)
    assert changed.total.energy - base.total.energy == pytest.approx(2 * base.weights.weights[0])
    np.testing.assert_allclose(changed.total.forces - base.total.forces,
                              -2 * base.weights.gradients_per_angstrom[0], atol=1e-12, rtol=0)


@pytest.mark.parametrize("k", [1, 2])
def test_translation_rotation_reordering_are_invariant(k):
    atoms, obj = geometry(), mixer(k)
    original = obj.evaluate_terms(atoms)
    order = (8, 3, 5, 1, 4, 2, 7, 6)
    shuffled = obj.evaluate_terms(atoms.select(order))
    np.testing.assert_allclose(shuffled.total.forces,
        original.total.forces[[atoms.atom_ids.index(a) for a in order]], atol=1e-12, rtol=0)
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    moved = replace(atoms, positions=atoms.positions @ rotation + [2., -4., .6])
    transformed = obj.evaluate_terms(moved)
    assert transformed.total.energy == pytest.approx(original.total.energy, abs=1e-12)
    np.testing.assert_allclose(transformed.total.forces, original.total.forces @ rotation,
                              atol=1e-12, rtol=0)


def test_pure_endpoint_skips_only_exactly_inactive_partitions_and_reactivates():
    obj, atoms = mixer(), geometry()
    xyz = atoms.positions.copy()
    xyz[2:4] = [[1.8, 0, 0], [1.8, 0, .74]]
    xyz[4:6] += [0, 2., 0]
    xyz[6:8] += [-2., 0, 0]
    pure = replace(atoms, positions=xyz)
    result = obj.evaluate_terms(pure)
    assert result.active_partition_ids == ("a",)
    assert result.corrections[1:] == (None, None)
    np.testing.assert_array_equal(result.weights.weights, [1, 0, 0])
    np.testing.assert_array_equal(result.switching_forces, np.zeros((8, 3)))
    assert obj.full_low.calls == 1
    assert [p.calculator.high.calls for p in obj.partitions] == [1, 0, 0]
    obj.evaluate(atoms)
    assert [p.calculator.high.calls for p in obj.partitions] == [2, 1, 1]
    obj.evaluate(pure)
    assert [p.calculator.high.calls for p in obj.partitions] == [3, 1, 1]


@pytest.mark.parametrize("boundary", [2., 3.])
def test_candidate_activation_boundaries_have_continuous_energy_force(boundary):
    obj, atoms = mixer(), geometry()
    values = []
    for shift in (-1e-5, 0, 1e-5):
        xyz = atoms.positions.copy()
        xyz[2:4] = [[boundary + shift, 0, 0], [boundary + shift, 0, .74]]
        values.append(obj.evaluate_terms(replace(atoms, positions=xyz)))
    h = 1e-5
    fd = -(values[2].total.energy - values[0].total.energy) / (2 * h)
    # Both atoms of the moving group are translated, including all weight forces.
    assert fd == pytest.approx(values[1].total.forces[2:4, 0].sum(), abs=2e-7)
    assert np.max(np.abs(values[2].total.forces - values[0].total.forces)) < 1e-3
    if boundary == 3:
        assert values[1].corrections[0] is None and values[2].corrections[0] is None
        assert values[0].corrections[0] is not None


def test_simultaneous_crossings_are_stateless_and_no_small_positive_weight_is_pruned():
    obj, atoms = mixer(), geometry()
    results = []
    for shift in (1e-5, 1e-7, 0., -1e-7, -1e-5):
        xyz = atoms.positions.copy()
        xyz[2:4] = [[3 + shift, 0, 0], [3 + shift, 0, .74]]
        xyz[6:8] = [[-3 - shift, 0, 0], [-3 - shift, 0, .74]]
        frame = replace(atoms, positions=xyz)
        values = obj.evaluate_terms(frame)
        back = mixer().evaluate_terms(frame)
        np.testing.assert_array_equal(values.total.forces, back.total.forces)
        assert values.total.energy == back.total.energy
        results.append(values)
    assert results[0].active_partition_ids == ("b",)
    assert results[-1].active_partition_ids == ("a", "b", "c")
    assert 0 < results[3].weights.weights[0] < 1e-15
    assert results[3].corrections[0] is not None


@pytest.mark.parametrize("k", [1, 2])
def test_coverage_failure_precedes_all_scientific_work(k):
    obj, atoms = mixer(k), geometry()
    xyz = atoms.positions.copy()
    xyz[2:] *= 3
    with pytest.raises(EvaluationError, match="Too few"):
        obj.evaluate(replace(atoms, positions=xyz))
    assert obj.full_low.calls == 0
    assert all(p.calculator.high.calls == 0 for p in obj.partitions)


@pytest.mark.parametrize("change", ["missing", "duplicate", "core", "composition", "method", "reference", "state"])
def test_family_and_chemical_identity_cannot_silently_change(change):
    declared = groups()
    parts = list(partitions())
    core = ("core",)
    if change == "missing":
        parts.pop()
    elif change == "duplicate":
        parts[1] = replace(parts[0], partition_id="another")
    elif change == "core":
        core = ("a",)
    elif change == "composition":
        declared = declared[:-1] + (replace(declared[-1], symbols=("He", "H")),)
    elif change == "method":
        parts[1] = replace(parts[1], method_identity="another method")
    elif change == "reference":
        parts[1] = replace(parts[1], energy_reference="another zero")
    else:
        parts[1].calculator.high.charge = 1
    with pytest.raises((ValueError, TypeError)):
        obj = mixer(groups=declared, partitions=parts, core_group_ids=core)
        obj.evaluate(geometry())


def test_active_failure_raises_without_stale_energy_and_next_call_recomputes():
    obj = mixer()
    obj.evaluate(geometry())
    obj.partitions[1].calculator.high.failed = True
    with pytest.raises(EvaluationError, match="Deliberate"):
        obj.evaluate(geometry())
    obj.partitions[1].calculator.high.failed = False
    result = obj.evaluate(geometry())
    assert result.energy == mixer().evaluate(geometry()).energy
    assert obj.full_low.calls == 3


def test_declared_state_is_checked_even_for_an_inactive_partition():
    obj, atoms = mixer(), geometry()
    xyz = atoms.positions.copy()
    xyz[4:] *= 3
    obj.partitions[-1].calculator.high.charge = 1
    with pytest.raises(ValueError, match="charge/multiplicity"):
        obj.evaluate(replace(atoms, positions=xyz))
    assert obj.full_low.calls == 0


@pytest.mark.parametrize("settings", [
    {"occupancy": 0}, {"occupancy": 3}, {"occupancy": True}, {"max_partitions": 2},
    {"inner_radius_angstrom": 3.}, {"outer_radius_angstrom": float("inf")},
    {"candidate_anchors": (("a", 3), ("a", 5))},
    {"candidate_anchors": (("a", 1), ("b", 5))},
])
def test_bad_switch_settings_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        replace(switch(), **settings)


def test_immutable_outputs_and_finite_boundary_guards():
    value = mixer().evaluate_terms(geometry())
    for array in (value.total.forces, value.switching_forces, value.weights.weights,
                  value.weights.gradients_per_angstrom, value.weights.candidate_weights):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(ValueError, match="nonperiodic"):
        switch().evaluate(replace(geometry(), cell=np.eye(3) * 10))
    xyz = geometry().positions.copy()
    xyz[2] = xyz[0]
    with pytest.raises(ValueError, match="nonzero"):
        switch().evaluate(replace(geometry(), positions=xyz))
