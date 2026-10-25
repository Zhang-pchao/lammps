# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Complete derivatives and identity/failure controls for exact term reduction."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from test_fragment_response_native import select
from test_gap_anchor import Embedded

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import config, load
from test_interaction_delta import CORE, FRAGMENT, PairModel, atoms

from qmdpa.core import EvaluationError
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.fragment_response import (
    ClusterGapDelta,
    OneBodyAlignedPairSet,
    PairInteractionSet,
    ReducedFragmentResponseCalculator,
)
from qmdpa.gap_anchor import GapAnchorCalculator
from qmdpa.interaction_delta import CrossInteractionDelta

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def composed(aligned=True, distinct=False):
    high, low = PairModel(0.21, 4.0), PairModel(0.13, -9.0)
    field = FixedEmbeddingCharges(FRAGMENT, [-0.8, 0.4], "analytic field")
    anchor = GapAnchorCalculator(
        low,
        low,
        high,
        Embedded(0.21, 0.07, field),
        Embedded(0.36, 0.17, field),
        qm_atom_ids=CORE + (18,),
        environment_atom_ids=FRAGMENT,
    )
    pairs = [
        CrossInteractionDelta(
            high, high, high, low, low, low, core_atom_ids=ids, fragment_atom_ids=FRAGMENT
        )
        for ids in (CORE, (18,))
    ]
    pair_set = (
        OneBodyAlignedPairSet(pairs, alignments=((0, "fragment"),))
        if aligned
        else PairInteractionSet(pairs)
    )
    ground = PairModel(0.21, 4.0) if distinct else high
    gap = ClusterGapDelta(
        ground,
        PairModel(0.36, 7.0),
        Embedded(0.21, 0.07, field),
        Embedded(0.36, 0.17, field),
        core_atom_ids=CORE,
        fragment_atom_ids=FRAGMENT,
    )
    return ReducedFragmentResponseCalculator(anchor, pair_set, gap), high, low


@pytest.mark.parametrize("aligned", [True, False])
def test_expanded_ledger_parity_and_cancelled_pair_not_evaluated(aligned):
    calc, high, low = composed(aligned)
    g = atoms()
    expanded = calc.evaluate_terms(g).total
    high.calls.clear()
    low.calls.clear()
    reduced = calc.evaluate(g)
    assert reduced.energy == pytest.approx(expanded.energy, abs=1e-12)
    np.testing.assert_allclose(reduced.forces, expanded.forces, atol=1e-13, rtol=0)
    assert CORE + FRAGMENT not in high.calls
    assert high.calls.count(FRAGMENT) == 1  # Shared environment monomer survives once.
    assert calc.last_execution["cancelled_terms"] >= 1
    assert calc.last_execution["required_selected_terms"] == 3
    assert not reduced.forces.flags.writeable


@pytest.mark.parametrize("step", [1e-3, 5e-4])
@pytest.mark.parametrize("aligned", [True, False])
def test_all_coordinate_finite_differences(step, aligned):
    calc, _, _ = composed(aligned)
    g = atoms()
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
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=1e-13, rtol=0)


def test_independent_provider_identity_does_not_cancel_equal_models():
    calc, high, _ = composed(distinct=True)
    result = calc.evaluate(atoms())
    assert CORE + FRAGMENT in high.calls
    assert calc.correction.pair_ground.calls == [CORE + FRAGMENT]
    expected, _, _ = composed()
    assert result.energy == pytest.approx(expected.evaluate(atoms()).energy, abs=1e-12)


def test_atom_order_rigid_covariance_and_no_cross_geometry_reuse():
    calc, high, _ = composed()
    g = atoms()
    a = calc.evaluate(g)
    count = len(high.calls)
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    moved = replace(g, positions=g.positions @ rotation + [2.0, -1.0, 3.0])
    order = tuple(reversed(g.atom_ids))
    b = calc.evaluate(moved.select(order))
    assert len(high.calls) == 2 * count
    assert a.energy == pytest.approx(b.energy, abs=1e-12)
    np.testing.assert_allclose(b.forces, (a.forces @ rotation)[::-1], atol=1e-13, rtol=0)
    x = g.positions.copy()
    x[0, 0] += 0.1
    c = calc.evaluate(replace(g, positions=x))
    assert c.energy != a.energy


@pytest.mark.parametrize("damage", ["field", "elements", "periodic", "missing"])
def test_original_input_guards_remain_fail_closed(damage):
    calc, high, _ = composed()
    g = atoms()
    calc.evaluate(g)
    high.calls.clear()
    if damage == "field":
        calc.correction.embedded_selected.charges = FixedEmbeddingCharges(
            FRAGMENT, [-0.7, 0.4], "changed"
        )
    elif damage == "elements":
        symbols = list(g.symbols)
        symbols[0] = "N"
        g = replace(g, symbols=symbols)
    elif damage == "periodic":
        g = replace(g, cell=np.eye(3) * 20)
    else:
        g = g.select(CORE)
    with pytest.raises((ValueError, KeyError)):
        calc.evaluate(g)
    assert calc.last_execution is None and not high.calls


@pytest.mark.parametrize("role", ["anchor", "pair", "field"])
def test_every_selected_provider_is_required_and_failure_propagates(role):
    calc, _, _ = composed()
    anchor, gap = calc.baseline.baseline, calc.correction
    backend = {
        "anchor": anchor.embedded_selected_backend,
        "pair": gap.pair_selected,
        "field": gap.embedded_selected,
    }[role]

    def fail(*args):
        raise EvaluationError("required selected state failed")

    backend.evaluate = fail
    with pytest.raises(EvaluationError, match="required selected"):
        calc.evaluate(atoms())
    assert calc.last_execution is None


def test_selected_provider_is_called_even_if_its_net_coefficient_is_zero():
    calc, _, _ = composed()
    gap = calc.correction
    gap.embedded_ground = gap.embedded_selected
    calc.evaluate(atoms())
    assert len(gap.embedded_selected.calls) == 1
    gap.embedded_selected.evaluate = lambda *args: (_ for _ in ()).throw(
        EvaluationError("zero-weight state failed")
    )
    with pytest.raises(EvaluationError, match="zero-weight state"):
        calc.evaluate(atoms())


def test_unsupported_composition_rejects_before_solving():
    calc, _, _ = composed()
    with pytest.raises(TypeError):
        ReducedFragmentResponseCalculator(calc.baseline, calc.baseline.correction, calc.correction)


def test_optional_configuration_is_explicit_and_described(tmp_path, config):
    select(config)
    config["reuse_ground_terms"] = True
    before = load(tmp_path, config)
    config["reduce_fragment_terms"] = True
    after = load(tmp_path, config)
    assert after["assembly"]["fragment_evaluation"] == "identical-signed-term-reduction"
    assert after["physics_sha256"] != before["physics_sha256"]
    assert after["fragment_response"] == before["fragment_response"]


@pytest.mark.parametrize("value,reuse", [(1, True), ("true", True), (True, False)])
def test_ambiguous_reduction_option_rejects_before_backend(tmp_path, config, value, reuse):
    select(config)
    config["reuse_ground_terms"] = reuse
    config["reduce_fragment_terms"] = value
    with (
        patch("qmdpa.backends.DeepMDBackend") as backend,
        pytest.raises(ValueError, match="reduce_fragment_terms"),
    ):
        load(tmp_path, config)
    backend.assert_not_called()


def test_option_cannot_silently_select_another_hamiltonian(tmp_path, config):
    config["reduce_fragment_terms"] = False
    with pytest.raises(ValueError, match="reduce_fragment_terms"):
        load(tmp_path, config)
