# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Conservative algebra and complete fixed-group exchange contracts."""

from dataclasses import replace

import numpy as np
import pytest

from qmdpa.adaptive import ConservativeMechanicalMixer, MechanicalPartition, QuinticDistanceExchange
from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.selection_config import FixedGroup


class PairPolynomial:
    """Analytic translation/rotation invariant control, not a water model."""

    def __init__(self, scale, offset=0.0):
        self.scale, self.offset = scale, offset
        self.charge, self.multiplicity = 0, 1
        self.charge_spin = np.array([0, 1])
        self.calls = 0
        self.failed = False

    def evaluate(self, geometry):
        self.calls += 1
        if self.failed:
            raise EvaluationError("Deliberate provider rejection")
        energy, forces = self.offset, np.zeros_like(geometry.positions)
        for i, first in enumerate(geometry.atom_ids):
            for j in range(i + 1, len(geometry.atom_ids)):
                coefficient = self.scale * (1 + (first + geometry.atom_ids[j]) % 3)
                delta = geometry.positions[i] - geometry.positions[j]
                energy += 0.5 * coefficient * np.dot(delta, delta)
                forces[i] -= coefficient * delta
                forces[j] += coefficient * delta
        return Evaluation(energy, forces)


def geometry():
    return Geometry(
        tuple(range(1, 10)),
        ("O", "H", "H") * 3,
        [
            [0, 0, 0],
            [0.8, 0.5, 0],
            [-0.8, 0.5, 0],
            [3, 0, 0],
            [3.8, 0.5, 0],
            [2.2, 0.5, 0],
            [0, 3.2, 0.1],
            [0.8, 3.7, 0.1],
            [-0.8, 3.7, 0.1],
        ],
    )


def groups():
    return tuple(
        FixedGroup(name, ids, ("O", "H", "H"))
        for name, ids in (("core", (1, 2, 3)), ("a", (4, 5, 6)), ("b", (7, 8, 9)))
    )


def partition(name, candidate):
    return MechanicalPartition(
        name,
        ("core", candidate),
        SubtractiveCalculator(PairPolynomial(0.04), PairPolynomial(0.07, -5.0)),
        qm_charge=0,
        qm_multiplicity=1,
        method_identity="analytic-pair-control-v1",
        energy_reference="shared-analytic-reference-v1",
    )


def mixer(*, partitions=None, declared_groups=None):
    return ConservativeMechanicalMixer(
        PairPolynomial(0.02, -10.0),
        groups=groups() if declared_groups is None else declared_groups,
        partitions=(partition("A", "a"), partition("B", "b")) if partitions is None else partitions,
        switch=QuinticDistanceExchange(1, 4, 7, 0.4),
    )


def displaced(original, row, axis, displacement):
    positions = original.positions.copy()
    positions[row, axis] += displacement
    return Geometry(original.atom_ids, original.symbols, positions)


def translate_candidate(original, shift):
    positions = original.positions.copy()
    positions[3:6, 0] += shift
    return Geometry(original.atom_ids, original.symbols, positions)


def test_complete_energy_derivative_includes_nonzero_switching_forces():
    calculator, atoms = mixer(), geometry()
    terms = calculator.evaluate_terms(atoms)
    h = 1e-5
    finite_difference = np.zeros_like(atoms.positions)
    for row in range(len(atoms.atom_ids)):
        for axis in range(3):
            plus = calculator.evaluate(displaced(atoms, row, axis, h)).energy
            minus = calculator.evaluate(displaced(atoms, row, axis, -h)).energy
            finite_difference[row, axis] = -(plus - minus) / (2 * h)
    np.testing.assert_allclose(terms.total.forces, finite_difference, atol=1e-7, rtol=0)
    assert terms.switching_force_max_ev_per_angstrom > 1e-3
    incomplete = terms.baseline.forces + terms.weighted_correction_forces
    assert np.max(np.abs(incomplete - finite_difference)) > 1e-3
    assert terms.total.energy == pytest.approx(
        np.dot(terms.weights, terms.partition_energies_ev), abs=1e-12
    )
    np.testing.assert_allclose(terms.total.forces.sum(axis=0), 0, atol=1e-12, rtol=0)


def test_weights_include_center_and_both_candidates_with_correct_cartesian_gradients():
    switch, atoms = QuinticDistanceExchange(1, 4, 7, 0.4), geometry()
    result = switch.evaluate(atoms)
    np.testing.assert_allclose(result.weights.sum(), 1, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result.gradients_per_angstrom.sum(axis=0), 0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(result.gradients_per_angstrom.sum(axis=1), 0, atol=1e-15, rtol=0)
    assert all(np.linalg.norm(result.gradients_per_angstrom[0, row]) > 0 for row in (0, 3, 6))
    h = 1e-6
    for row in range(len(atoms.atom_ids)):
        for axis in range(3):
            numerical = (
                switch.evaluate(displaced(atoms, row, axis, h)).weights
                - switch.evaluate(displaced(atoms, row, axis, -h)).weights
            ) / (2 * h)
            np.testing.assert_allclose(
                result.gradients_per_angstrom[:, row, axis], numerical, atol=2e-9, rtol=0
            )


@pytest.mark.parametrize("shift,active", [(-1.0, 0), (1.0, 1)])
def test_pure_endpoints_match_fixed_partition_without_skipping_calls(shift, active):
    calculator = mixer()
    terms = calculator.evaluate_terms(translate_candidate(geometry(), shift))
    assert terms.weights.tolist() == ([1, 0] if active == 0 else [0, 1])
    assert np.count_nonzero(terms.switching_forces) == 0
    assert np.count_nonzero(terms.weight_gradients_per_angstrom) == 0
    assert terms.total.energy == pytest.approx(terms.partition_energies_ev[active], abs=1e-12)
    np.testing.assert_allclose(
        terms.total.forces,
        terms.baseline.forces + terms.corrections[active].forces,
        atol=1e-13,
        rtol=0,
    )
    assert calculator.full_low.calls == 1
    for declared in calculator.partitions:
        assert declared.calculator.high.calls == declared.calculator.low.calls == 1


def test_common_energy_zero_does_not_generate_a_switching_force():
    calculator, atoms = mixer(), geometry()
    original = calculator.evaluate_terms(atoms)
    calculator.full_low.offset += 1000
    baseline_shift = calculator.evaluate_terms(atoms)
    assert baseline_shift.total.energy - original.total.energy == pytest.approx(1000, abs=1e-12)
    np.testing.assert_array_equal(baseline_shift.total.forces, original.total.forces)
    for declared in calculator.partitions:
        declared.calculator.high.offset += 100
    shared_correction_shift = calculator.evaluate_terms(atoms)
    assert shared_correction_shift.total.energy - baseline_shift.total.energy == pytest.approx(
        100, abs=1e-12
    )
    np.testing.assert_allclose(
        shared_correction_shift.total.forces, original.total.forces, atol=1e-12, rtol=0
    )


def test_partition_offset_is_explicit_and_its_switching_force_is_retained():
    first, second = partition("A", "a"), partition("B", "b")
    original = mixer(partitions=(first, second)).evaluate_terms(geometry())
    shifted_first = replace(first, energy_offset_ev=0.125, offset_source="Declared control offset")
    shifted = mixer(partitions=(shifted_first, second)).evaluate_terms(geometry())
    assert shifted.total.energy - original.total.energy == pytest.approx(
        0.125 * original.weights[0], abs=1e-12
    )
    np.testing.assert_allclose(
        shifted.total.forces - original.total.forces,
        -0.125 * original.weight_gradients_per_angstrom[0],
        atol=1e-12,
        rtol=0,
    )
    assert shifted.energy_offsets_ev == (0.125, 0)
    assert shifted.offset_sources == ("Declared control offset", None)
    with pytest.raises(ValueError, match="offset source"):
        replace(first, energy_offset_ev=0.125)


def test_reordered_atoms_translation_and_rotation_preserve_mapping():
    calculator, atoms = mixer(), geometry()
    original = calculator.evaluate_terms(atoms)
    reordered = Geometry(atoms.atom_ids[::-1], atoms.symbols[::-1], atoms.positions[::-1])
    other = calculator.evaluate_terms(reordered)
    assert other.total.energy == original.total.energy
    np.testing.assert_array_equal(other.total.forces, original.total.forces[::-1])
    np.testing.assert_array_equal(
        other.weight_gradients_per_angstrom, original.weight_gradients_per_angstrom[:, ::-1]
    )
    translated = Geometry(atoms.atom_ids, atoms.symbols, atoms.positions + [1.2, -0.4, 0.5])
    moved = calculator.evaluate_terms(translated)
    assert moved.total.energy == pytest.approx(original.total.energy, abs=1e-12)
    np.testing.assert_allclose(moved.total.forces, original.total.forces, atol=1e-12, rtol=0)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = calculator.evaluate_terms(
        Geometry(atoms.atom_ids, atoms.symbols, atoms.positions @ rotation.T)
    )
    assert rotated.total.energy == pytest.approx(original.total.energy, abs=1e-12)
    np.testing.assert_allclose(
        rotated.total.forces, original.total.forces @ rotation.T, atol=1e-12, rtol=0
    )


def test_finite_large_switching_force_norm_avoids_square_overflow():
    first, second = partition("A", "a"), partition("B", "b")
    first.calculator.high.offset = 1e200
    terms = mixer(partitions=(first, second)).evaluate_terms(geometry())
    expected = 1e200 * np.max(np.linalg.norm(terms.switching_forces / 1e200, axis=1))
    assert np.isfinite(terms.switching_force_max_ev_per_angstrom)
    assert terms.switching_force_max_ev_per_angstrom == pytest.approx(expected, rel=1e-15)


def test_results_are_immutable_and_failures_are_not_replaced():
    calculator, atoms = mixer(), geometry()
    accepted = calculator.evaluate_terms(atoms)
    for values in (
        accepted.weights,
        accepted.weight_gradients_per_angstrom,
        accepted.switching_forces,
        accepted.weighted_correction_forces,
        accepted.total.forces,
    ):
        with pytest.raises(ValueError):
            values.flat[0] = 0
    calculator.partitions[1].calculator.high.failed = True
    with pytest.raises(EvaluationError, match="Deliberate"):
        calculator.evaluate(atoms)


@pytest.mark.parametrize(
    "changes",
    [
        {"qm_charge": 1},
        {"qm_multiplicity": 3},
        {"electronic_state": "excited"},
        {"energy_offset_ev": True},
        {"energy_offset_ev": float("nan")},
        {"method_identity": ""},
        {"energy_reference": ""},
    ],
)
def test_invalid_or_unbound_partition_state_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(partition("A", "a"), **changes)


def test_low_region_condition_and_changed_high_state_are_checked():
    declared = partition("A", "a")
    declared.calculator.low.charge_spin = [1, 1]
    with pytest.raises(ValueError, match="Low-region"):
        replace(declared)
    calculator = mixer()
    calculator.partitions[0].calculator.high.charge = 1
    with pytest.raises(ValueError, match="High-provider"):
        calculator.evaluate(geometry())


@pytest.mark.parametrize(
    "change",
    [
        "composition",
        "overlap",
        "different-method",
        "different-reference",
        "wrong-candidate",
        "duplicate-partition",
    ],
)
def test_unsupported_partition_families_are_rejected(change):
    declarations = groups()
    first, second = partition("A", "a"), partition("B", "b")
    if change == "composition":
        declarations = declarations[:2] + (FixedGroup("b", (7, 8, 9), ("O", "H", "O")),)
    elif change == "overlap":
        declarations = declarations[:2] + (FixedGroup("b", (4, 8, 9), ("O", "H", "H")),)
    elif change == "different-method":
        second = replace(second, method_identity="another-method")
    elif change == "different-reference":
        second = replace(second, energy_reference="another-zero")
    elif change == "wrong-candidate":
        second = replace(second, qm_group_ids=("core", "a"))
    else:
        second = replace(second, partition_id="A")
    with pytest.raises(ValueError):
        mixer(partitions=(first, second), declared_groups=declarations)


def test_periodic_missing_and_changed_identity_are_rejected_before_calculation():
    calculator, atoms = mixer(), geometry()
    invalid = [
        Geometry(atoms.atom_ids, atoms.symbols, atoms.positions, np.eye(3) * 20),
        atoms.select(atoms.atom_ids[:-1]),
        Geometry(atoms.atom_ids, ("N", *atoms.symbols[1:]), atoms.positions),
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            calculator.evaluate(value)
    assert calculator.full_low.calls == 0


@pytest.mark.parametrize(
    "anchors,width",
    [
        ((1, 1, 7), 0.4),
        ((True, 4, 7), 0.4),
        ((1, 4, 7), 0),
        ((1, 4, 7), True),
        ((1, 4, 7), float("inf")),
    ],
)
def test_invalid_anchor_or_width_declarations(anchors, width):
    with pytest.raises((ValueError, TypeError)):
        QuinticDistanceExchange(*anchors, width)


@pytest.mark.parametrize("coincident", [(0, 3), (3, 6)])
def test_coincident_anchors_are_rejected(coincident):
    atoms = geometry()
    positions = atoms.positions.copy()
    positions[coincident[0]] = positions[coincident[1]]
    with pytest.raises(ValueError, match="coincide"):
        mixer().evaluate(Geometry(atoms.atom_ids, atoms.symbols, positions))


def test_pure_boundary_weights_and_first_derivatives_are_continuous():
    atoms = geometry()
    switch = QuinticDistanceExchange(1, 4, 7, 0.4)
    second_radius = np.linalg.norm(atoms.positions[6] - atoms.positions[0])
    for endpoint in (-0.4, 0.4):
        edge = switch.evaluate(translate_candidate(atoms, second_radius + endpoint - 3))
        inside_s = endpoint + (1e-5 if endpoint < 0 else -1e-5)
        inside = switch.evaluate(translate_candidate(atoms, second_radius + inside_s - 3))
        np.testing.assert_allclose(inside.weights, edge.weights, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            inside.gradients_per_angstrom, edge.gradients_per_angstrom, atol=1e-7, rtol=0
        )


def test_reversible_coordinate_scan_has_no_membership_history():
    calculator, atoms = mixer(), geometry()
    shifts = (-1.0, -0.4, 0.0, 0.4, 1.0)
    forward = [calculator.evaluate_terms(translate_candidate(atoms, shift)) for shift in shifts]
    backward = [
        calculator.evaluate_terms(translate_candidate(atoms, shift)) for shift in reversed(shifts)
    ]
    for first, second in zip(forward, reversed(backward)):
        assert first.total.energy == second.total.energy
        np.testing.assert_array_equal(first.total.forces, second.total.forces)
        np.testing.assert_array_equal(first.weights, second.weights)


def test_equal_zero_corrections_reduce_to_the_single_shared_baseline():
    first, second = partition("A", "a"), partition("B", "b")
    for declared in (first, second):
        declared.calculator.high.scale = declared.calculator.low.scale
        declared.calculator.high.offset = declared.calculator.low.offset
    result = mixer(partitions=(first, second)).evaluate_terms(geometry())
    assert result.total.energy == result.baseline.energy
    np.testing.assert_array_equal(result.total.forces, result.baseline.forces)
    assert result.switching_force_max_ev_per_angstrom == 0


@pytest.mark.parametrize("attribute,value", [("charge", 1), ("multiplicity", 3)])
def test_same_composition_with_different_electronic_states_cannot_be_blended(attribute, value):
    first, second = partition("A", "a"), partition("B", "b")
    setattr(second.calculator.high, attribute, value)
    second.calculator.low.charge_spin = [1, 1] if attribute == "charge" else [0, 3]
    replacement = {"qm_charge": value} if attribute == "charge" else {"qm_multiplicity": value}
    second = replace(second, **replacement)
    with pytest.raises(ValueError, match="must share"):
        mixer(partitions=(first, second))


def test_distinct_anchor_ids_in_the_same_declared_group_are_not_an_exchange():
    with pytest.raises(ValueError, match="three distinct complete groups"):
        ConservativeMechanicalMixer(
            PairPolynomial(0.02),
            groups=groups(),
            partitions=(partition("A", "a"), partition("B", "b")),
            switch=QuinticDistanceExchange(1, 2, 7, 0.4),
        )
