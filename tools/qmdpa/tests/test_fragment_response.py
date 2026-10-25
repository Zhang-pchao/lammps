# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Independent derivatives, pair coverage, mapping and failure propagation."""

from dataclasses import replace

import numpy as np
import pytest
from test_gap_anchor import Embedded
from test_interaction_delta import CORE, FRAGMENT, PairModel, atoms

from qmdpa.core import EvaluationError
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.fragment_response import ClusterGapDelta, OneBodyAlignedPairSet, PairInteractionSet
from qmdpa.interaction_delta import CorrectedCalculator, CrossInteractionDelta


def delta(core, fragment):
    h = PairModel(0.21)
    low_model = PairModel(0.13)
    return CrossInteractionDelta(h, h, h, low_model, low_model, low_model, core_atom_ids=core, fragment_atom_ids=fragment)


def gap(ground_only=False):
    field = FixedEmbeddingCharges(FRAGMENT, [-0.8, 0.4], "analytic fixed field")
    ground = PairModel(0.21, 4.0)
    selected = ground if ground_only else PairModel(0.36, 7.0)
    eg = Embedded(0.21, 0.07, field)
    es = eg if ground_only else Embedded(0.36, 0.17, field)
    return ClusterGapDelta(ground, selected, eg, es, core_atom_ids=CORE, fragment_atom_ids=FRAGMENT)


@pytest.mark.parametrize("step", [1e-3, 5e-4])
@pytest.mark.parametrize("which", ["pairs", "gap"])
def test_complete_cartesian_derivatives_with_spectator_and_large_ids(step, which):
    g = atoms()
    calc = (
        gap()
        if which == "gap"
        else PairInteractionSet([delta(CORE, FRAGMENT), delta((18,), FRAGMENT)])
    )
    result = calc.evaluate(g)
    for i in range(len(g.atom_ids)):
        for axis in range(3):
            values = []
            for sign in (1, -1):
                x = g.positions.copy()
                x[i, axis] += sign * step
                values.append(calc.evaluate(replace(g, positions=x)).energy)
            assert -(values[0] - values[1]) / (2 * step) == pytest.approx(
                result.forces[i, axis], abs=1e-8
            )
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=1e-14)
    if which == "gap":
        np.testing.assert_array_equal(result.forces[2], 0.0)
    assert not result.forces.flags.writeable


def test_pair_set_matches_independent_full_pair_law_difference():
    g = atoms()
    calc = PairInteractionSet([delta(CORE, FRAGMENT), delta((18,), FRAGMENT)])
    e = 0.0
    f = np.zeros_like(g.positions)
    for atom in CORE + (18,):
        i = g.atom_ids.index(atom)
        for other in FRAGMENT:
            j = g.atom_ids.index(other)
            r = g.positions[i] - g.positions[j]
            e += 0.5 * 0.08 * np.dot(r, r)
            f[i] -= 0.08 * r
            f[j] += 0.08 * r
    actual = calc.evaluate(g)
    assert actual.energy == pytest.approx(e, abs=1e-13)
    np.testing.assert_allclose(actual.forces, f, atol=1e-13)


@pytest.mark.parametrize(
    "second",
    [
        lambda: delta(CORE, FRAGMENT),
        lambda: delta(FRAGMENT, CORE),
        lambda: delta(CORE + (18,), FRAGMENT),
    ],
)
def test_pair_overlap_is_rejected_even_when_fragment_sets_differ(second):
    with pytest.raises(ValueError, match="double-count"):
        PairInteractionSet([delta(CORE, FRAGMENT), second()])


def test_gap_ground_limit_and_rigid_covariance():
    g = atoms()
    np.testing.assert_array_equal(gap(True).evaluate(g).forces, 0.0)
    assert gap(True).evaluate(g).energy == 0.0
    calc = gap()
    a = calc.evaluate(g)
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    b = calc.evaluate(
        replace(g, positions=g.positions @ rotation + [1, 2, 3]).select(tuple(reversed(g.atom_ids)))
    )
    assert a.energy == pytest.approx(b.energy, abs=1e-13)
    np.testing.assert_allclose(b.forces, (a.forces @ rotation)[::-1], atol=1e-13)


def test_gap_field_change_and_backend_failure_propagate():
    g = atoms()
    calc = gap()
    calc.evaluate(g)
    calc.embedded_selected.charges = FixedEmbeddingCharges(FRAGMENT, [-0.7, 0.4], "changed")
    with pytest.raises(ValueError, match="field"):
        calc.evaluate(g)
    calc = gap()

    class Failed:
        def evaluate(self, g):
            raise EvaluationError("explicit provider failure")

    calc.pair_selected = Failed()
    with pytest.raises(EvaluationError, match="explicit provider"):
        calc.evaluate(g)


def test_gap_missing_region_or_periodic_input_cannot_drop_forces():
    g = atoms()
    calc = gap()
    with pytest.raises((ValueError, KeyError)):
        calc.evaluate(g.select(CORE))
    with pytest.raises(ValueError, match="finite"):
        calc.evaluate(replace(g, cell=np.eye(3) * 20))




def aligned(high=None, low=None, alignments=((0, "fragment"),)):
    h = PairModel(0.21, 4.0) if high is None else high
    low_model = PairModel(0.13, -9.0) if low is None else low
    pair = CrossInteractionDelta(h, h, h, low_model, low_model, low_model, core_atom_ids=CORE, fragment_atom_ids=FRAGMENT)
    return OneBodyAlignedPairSet((pair,), alignments=alignments)


def test_one_body_reuses_explicit_monomer_without_additional_calls():
    g = atoms()
    h = PairModel(0.21, 4.0)
    low_model = PairModel(0.13, -9.0)
    result = aligned(h, low_model).evaluate_terms(g)
    assert h.calls == [CORE + FRAGMENT, CORE, FRAGMENT] == low_model.calls
    term = result.alignments[0]
    assert term.high is result.pair_set.pairs[0].high.fragment
    assert term.low is result.pair_set.pairs[0].low.fragment
    assert term.atom_ids == FRAGMENT and term.side == "fragment" and term.pair_index == 0
    assert result.model != result.pair_set.model
    assert not result.correction.forces.flags.writeable
    # Independent quadratic law: every cross pair and one internal D-D pair.
    expected = 13.0 * len(FRAGMENT)
    forces = np.zeros_like(g.positions)
    pairs = [(a, b) for a in CORE for b in FRAGMENT] + [(FRAGMENT[0], FRAGMENT[1])]
    for a, b in pairs:
        i, j = g.atom_ids.index(a), g.atom_ids.index(b)
        dr = g.positions[i] - g.positions[j]
        expected += 0.5 * 0.08 * np.dot(dr, dr)
        forces[i] -= 0.08 * dr
        forces[j] += 0.08 * dr
    assert result.correction.energy == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(result.correction.forces, forces, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(
        term.correction.forces[[g.atom_ids.index(a) for a in CORE + (18,)]], 0.0
    )
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-14, rtol=0)


@pytest.mark.parametrize("step", [1e-3, 5e-4])
@pytest.mark.parametrize("side", ["core", "fragment"])
def test_one_body_all_coordinate_two_step_fd(step, side):
    g = atoms()
    calc = aligned(alignments=((0, side),))
    result = calc.evaluate(g)
    for i in range(len(g.atom_ids)):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                x = g.positions.copy()
                x[i, axis] += sign * step
                energies.append(calc.evaluate(replace(g, positions=x)).energy)
            assert -(energies[0] - energies[1]) / (2 * step) == pytest.approx(
                result.forces[i, axis], abs=1e-8
            )


def test_one_body_permutation_translation_rotation_and_identity():
    g = atoms()
    calc = aligned()
    a = calc.evaluate(g)
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    transformed = replace(g, positions=g.positions @ rotation + [1.0, 2.0, 3.0]).select(
        tuple(reversed(g.atom_ids))
    )
    b = calc.evaluate(transformed)
    assert a.energy == pytest.approx(b.energy, abs=1e-12)
    np.testing.assert_allclose(b.forces, (a.forces @ rotation)[::-1], atol=1e-13, rtol=0)
    symbols = list(g.symbols)
    symbols[0] = "N"
    with pytest.raises(ValueError, match="symbols"):
        calc.evaluate(replace(g, symbols=symbols))


def test_one_body_equal_methods_cancel_and_shared_shift_preserves_gap():
    g = atoms()
    h = PairModel(0.21, 4.0)
    same = aligned(h, h).evaluate(g)
    assert same.energy == 0.0
    np.testing.assert_array_equal(same.forces, 0.0)
    ground, selected = PairModel(0.21, 2.0), PairModel(0.27, 3.0)
    gc, sc = CorrectedCalculator(ground, aligned()), CorrectedCalculator(selected, aligned())
    a, b = gc.evaluate(g), sc.evaluate(g)
    aa, bb = ground.evaluate(g), selected.evaluate(g)
    assert b.energy - a.energy == pytest.approx(bb.energy - aa.energy, abs=1e-12)
    np.testing.assert_allclose(b.forces - a.forces, bb.forces - aa.forces, atol=1e-13, rtol=0)


@pytest.mark.parametrize(
    "choices",
    [
        (),
        ((-1, "fragment"),),
        ((1, "fragment"),),
        ((True, "fragment"),),
        ((0, "pair"),),
        ((0,),),
        (0,),
        ((0, "fragment"), (0, "fragment")),
    ],
)
def test_one_body_invalid_or_duplicate_selection_rejected(choices):
    with pytest.raises(ValueError):
        aligned(alignments=choices)


def test_one_body_shared_fragment_selected_once_and_partial_overlap_rejected():
    pair1 = delta(CORE, FRAGMENT)
    pair2 = delta((18,), FRAGMENT)
    with pytest.raises(ValueError, match="overlap"):
        OneBodyAlignedPairSet((pair1, pair2), alignments=((0, "fragment"), (1, "fragment")))
    partial = delta((18,), (FRAGMENT[0],))
    with pytest.raises(ValueError, match="overlap"):
        OneBodyAlignedPairSet((pair1, partial), alignments=((0, "fragment"), (1, "fragment")))
    result = OneBodyAlignedPairSet((pair1, pair2), alignments=((1, "fragment"),)).evaluate_terms(
        atoms()
    )
    assert len(result.alignments) == 1
    assert result.alignments[0].high is result.pair_set.pairs[1].high.fragment


def test_one_body_two_disjoint_monomers_and_failures():
    g = atoms()
    calc = aligned(alignments=((0, "core"), (0, "fragment")))
    a = calc.evaluate_terms(g)
    assert len(a.alignments) == 2
    with pytest.raises(ValueError, match="nonperiodic"):
        calc.evaluate(replace(g, cell=np.eye(3) * 20))
    with pytest.raises((ValueError, KeyError)):
        calc.evaluate(g.select(CORE))

    class Failed:
        def evaluate(self, geometry):
            raise EvaluationError("one-body provider failed")

    with pytest.raises(EvaluationError, match="one-body provider"):
        aligned(high=Failed()).evaluate(g)
