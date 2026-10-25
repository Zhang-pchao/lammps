# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Analytic controls for the distinct finite gap-anchor Hamiltonian."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_reference import (
    ENV_IDS,
    QM_IDS,
    LowModel,
    cross_force_in_full_order,
    finite_difference,
    geometry,
    pair_quadratic,
)

from qmdpa.core import Evaluation, EvaluationError, SubtractiveCalculator
from qmdpa.embedding import EmbeddedEvaluation, FixedEmbeddingCharges
from qmdpa.gap_anchor import GapAnchorCalculator, make_rhf_gap_anchor


class Embedded:
    """Translation-invariant internal and charge-dependent cross quadratic."""

    def __init__(self, stiffness, interaction, charges):
        self.stiffness, self.interaction, self.charges = stiffness, interaction, charges
        self.calls = []

    def evaluate(self, qm, environment):
        self.calls.append((qm, environment))
        energy, qforce = pair_quadratic(qm.positions, self.stiffness)
        mforce = np.zeros_like(environment.positions)
        for i, first in enumerate(qm.positions):
            for j, second in enumerate(environment.positions):
                delta = first - second
                coefficient = self.interaction * self.charges.charges_e[j]
                energy += coefficient * np.dot(delta, delta) / 2
                force = -coefficient * delta
                qforce[i] += force
                mforce[j] -= force
        return EmbeddedEvaluation(energy, qm.atom_ids, environment.atom_ids, qforce, mforce)


def model(*, scale=1.0, ground_state=False, **overrides):
    charges = FixedEmbeddingCharges(ENV_IDS, np.array([-0.8, 0.4]) * scale, source="analytic")
    ground = Embedded(0.21, 0.07, charges)
    arguments = dict(
        full_low_backend=LowModel(0.13),
        region_low_backend=LowModel(0.11),
        isolated_ground_backend=LowModel(0.21),
        embedded_ground_backend=ground,
        embedded_selected_backend=ground if ground_state else Embedded(0.36, 0.17, charges),
        qm_atom_ids=QM_IDS,
        environment_atom_ids=ENV_IDS,
    )
    arguments.update(overrides)
    return GapAnchorCalculator(**arguments)


def test_five_terms_and_both_force_orders():
    atoms, calc = geometry(), model()
    t = calc.evaluate_terms(atoms)
    assert t.hamiltonian == "finite-electrostatic-gap-anchor"
    assert t.atom_ids == atoms.atom_ids
    assert t.qm_atom_ids == QM_IDS and t.environment_atom_ids == ENV_IDS
    expected = (
        t.full_low.energy
        - t.region_low.energy
        + t.isolated_ground.energy
        + t.embedded_selected.energy_ev
        - t.embedded_ground.energy_ev
    )
    assert t.total.energy == pytest.approx(expected, abs=1e-14)
    f = t.full_low.forces.copy()
    qrows = [atoms.atom_ids.index(x) for x in QM_IDS]
    f[qrows] += t.isolated_ground.forces - t.region_low.forces
    f += cross_force_in_full_order(t.embedded_selected, atoms)
    f -= cross_force_in_full_order(t.embedded_ground, atoms)
    np.testing.assert_allclose(t.total.forces, f, atol=1e-14, rtol=0)
    assert np.max(np.abs(t.gap.environment_forces_ev_per_angstrom)) > 0.01
    assert calc.full_low_backend.inputs[0].atom_ids == QM_IDS + ENV_IDS
    for provider in (calc.embedded_ground_backend, calc.embedded_selected_backend):
        assert len(provider.calls) == 1
        assert provider.calls[0][0].atom_ids == QM_IDS
        assert provider.calls[0][1].atom_ids == ENV_IDS


@pytest.mark.parametrize("step", [1e-4, 5e-5])
def test_all_coordinate_force_and_net_force(step):
    atoms, calc = geometry(), model()
    result = calc.evaluate(atoms)
    fd = finite_difference(atoms, lambda g: calc.evaluate(g).energy, step)
    np.testing.assert_allclose(result.forces, fd, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=1e-14, rtol=0)


def test_translation_rotation_reordering_and_repeat():
    atoms, calc = geometry(), model()
    first = calc.evaluate_terms(atoms)
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    moved = replace(atoms, positions=atoms.positions @ rotation + [3, -2, 1])
    other = calc.evaluate_terms(moved.select(tuple(reversed(atoms.atom_ids))))
    assert other.total.energy == pytest.approx(first.total.energy, abs=1e-14)
    np.testing.assert_allclose(
        other.total.forces, (first.total.forces @ rotation)[::-1], atol=1e-14
    )
    np.testing.assert_array_equal(calc.evaluate(atoms).forces, first.total.forces)
    assert calc.full_low_backend.inputs[-1].atom_ids == QM_IDS + ENV_IDS


def test_ground_state_gap_cancels_exactly_without_second_call():
    atoms, calc = geometry(), model(ground_state=True)
    t = calc.evaluate_terms(atoms)
    assert t.embedded_selected is t.embedded_ground
    assert t.gap.energy_ev == 0
    np.testing.assert_array_equal(t.gap.qm_forces_ev_per_angstrom, 0)
    np.testing.assert_array_equal(t.gap.environment_forces_ev_per_angstrom, 0)
    assert t.total.energy == t.ground_me.energy
    np.testing.assert_array_equal(t.total.forces, t.ground_me.forces)
    assert len(calc.embedded_ground_backend.calls) == 1


def test_zero_field_is_selected_state_me():
    atoms = geometry()
    actual = model(scale=0).evaluate(atoms)
    expected = SubtractiveCalculator(LowModel(0.11), LowModel(0.36), full_low=LowModel(0.13))
    result = expected.evaluate(atoms, atoms.select(QM_IDS))
    assert actual.energy == pytest.approx(result.energy, abs=1e-14)
    np.testing.assert_allclose(actual.forces, result.forces, atol=1e-14)


def test_snapshots_are_immutable():
    calc = model()
    t = calc.evaluate_terms(geometry())
    for item in (t.full_low, t.region_low, t.isolated_ground, t.ground_me, t.total):
        with pytest.raises(ValueError):
            item.forces.setflags(write=True)
    for item in (t.embedded_ground, t.embedded_selected, t.gap):
        for value in (item.qm_forces_ev_per_angstrom, item.environment_forces_ev_per_angstrom):
            with pytest.raises(ValueError):
                value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        t.total = None
    with pytest.raises(AttributeError):
        calc.charges = None


@pytest.mark.parametrize(
    "qm,env",
    [([], ENV_IDS), (QM_IDS, []), (QM_IDS, QM_IDS), ((30, 30), ENV_IDS), ((True, 3), ENV_IDS)],
)
def test_bad_regions(qm, env):
    with pytest.raises(ValueError):
        model(qm_atom_ids=qm, environment_atom_ids=env)


@pytest.mark.parametrize("change", ["source", "values", "radii", "ids", "missing"])
def test_different_fields_and_late_mutation_are_rejected(change):
    calc = model()
    c = calc.charges
    changed = {
        "source": lambda: replace(c, source="different"),
        "values": lambda: replace(c, charges_e=[-0.7, 0.4]),
        "radii": lambda: replace(c, radii_angstrom=[1, 1]),
        "ids": lambda: replace(c, atom_ids=tuple(reversed(ENV_IDS))),
        "missing": lambda: None,
    }[change]()
    with pytest.raises(ValueError, match="same fixed"):
        model(embedded_selected_backend=Embedded(0.36, 0.17, changed))
    calc.evaluate(geometry())
    calc.embedded_selected_backend.charges = changed
    with pytest.raises(ValueError, match="field changed"):
        calc.evaluate(geometry())


def test_charge_order_geometry_and_symbols_are_fixed():
    with pytest.raises(ValueError, match="ID order"):
        model(environment_atom_ids=tuple(reversed(ENV_IDS)))
    atoms, calc = geometry(), model()
    with pytest.raises(ValueError, match="nonperiodic"):
        calc.evaluate(replace(atoms, cell=np.eye(3) * 20))
    with pytest.raises(ValueError, match="cover all"):
        calc.evaluate(atoms.select(atoms.atom_ids[:-1]))
    calc.evaluate(atoms)
    with pytest.raises(ValueError, match="symbols"):
        calc.evaluate(replace(atoms, symbols=("C",) + atoms.symbols[1:]))


@pytest.mark.parametrize(
    "field", ["full_low_backend", "region_low_backend", "isolated_ground_backend"]
)
@pytest.mark.parametrize("bad", [None, Evaluation(0, np.zeros((1, 3)))])
def test_bad_ordinary_returns(field, bad):
    with pytest.raises(EvaluationError):
        model(**{field: SimpleNamespace(evaluate=lambda g: bad)}).evaluate(geometry())


@pytest.mark.parametrize("field", ["embedded_ground_backend", "embedded_selected_backend"])
@pytest.mark.parametrize(
    "bad",
    [
        None,
        Evaluation(0, np.zeros((2, 3))),
        EmbeddedEvaluation(0, ENV_IDS, QM_IDS, np.zeros((2, 3)), np.zeros((2, 3))),
    ],
)
def test_bad_embedded_returns(field, bad):
    calc = model()
    with patch.object(getattr(calc, field), "evaluate", return_value=bad):
        with pytest.raises(EvaluationError):
            calc.evaluate(geometry())


@pytest.mark.parametrize(
    "field",
    [
        "full_low_backend",
        "region_low_backend",
        "isolated_ground_backend",
        "embedded_ground_backend",
        "embedded_selected_backend",
    ],
)
def test_provider_failure_after_success_is_propagated(field):
    atoms, calc = geometry(), model()
    calc.evaluate(atoms)
    with patch.object(
        getattr(calc, field), "evaluate", side_effect=EvaluationError("controlled failure")
    ):
        with pytest.raises(EvaluationError, match="controlled failure"):
            calc.evaluate(atoms)


def test_factory_requires_explicit_embedded_reference():
    for reference in (None, SimpleNamespace(environment_geometry=None)):
        with pytest.raises(ValueError, match="frozen embedded"):
            make_rhf_gap_anchor(None, None, reference)
