"""Independent reference-field derivatives, image invariance and strict identities."""
import dataclasses
import itertools

import numpy as np
import pytest

from qmdpa.core import Geometry
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.periodic_field import (
    COULOMB_EV_ANGSTROM,
    PeriodicGaussianField,
    PeriodicGaussianReference,
)


def fixture():
    cell = np.diag([6.0, 7.0, 8.0])
    qm = Geometry((8, 2), ("H", "H"), [[2.2, 3.1, 2.7], [3.0, 3.2, 3.1]], cell)
    env = Geometry((7, 5), ("H", "H"), [[1.1, 2.0, 3.0], [4.9, 4.4, 5.2]], cell)
    field = PeriodicGaussianField(env.atom_ids, [0.3, -0.3], [0.8, 1.0],
                                  [6.0, 7.0, 8.0], 3, "analytic fixture")
    ref = PeriodicGaussianReference(FixedEmbeddingCharges(qm.atom_ids, [0.2, -0.2],
                                                         "declared reference"), field)
    return qm, env, field, ref


def test_independent_cosine_energy_and_complete_derivatives():
    qm, env, field, ref = fixture()
    result = ref.evaluate(qm, env)
    expected = 0.0
    for index in itertools.product(range(-3, 4), repeat=3):
        if index == (0, 0, 0):
            continue
        g = 2 * np.pi * np.array(index) / field.cell_angstrom
        g2 = g @ g
        c = 4 * np.pi * COULOMB_EV_ANGSTROM / (np.prod(field.cell_angstrom) * g2)
        for i, q in enumerate(ref.qm_charges.charges_e):
            for j, charge in enumerate(field.charges_e):
                expected += c * q * charge * np.exp(-0.5 * field.sigma_angstrom[j]**2 * g2) * np.cos(
                    g @ (qm.positions[i] - env.positions[j]))
    assert result.energy_ev == pytest.approx(expected, abs=1e-14)
    for step in (1e-4, 5e-5):
        for region, original, forces in (
                ("qm", qm, result.qm_forces_ev_per_angstrom),
                ("environment", env, result.environment_forces_ev_per_angstrom)):
            for i in range(2):
                for axis in range(3):
                    energies = []
                    for sign in (-1, 1):
                        positions = original.positions.copy()
                        positions[i, axis] += sign * step
                        moved = dataclasses.replace(original, positions=positions)
                        value = ref.evaluate(moved, env) if region == "qm" else ref.evaluate(qm, moved)
                        energies.append(value.energy_ev)
                    assert -(energies[1] - energies[0]) / (2 * step) == pytest.approx(
                        forces[i, axis], abs=2e-8)
    np.testing.assert_allclose(result.qm_forces_ev_per_angstrom.sum(0)
                               + result.environment_forces_ev_per_angstrom.sum(0), 0, atol=1e-14)


def test_images_translation_and_immutable_identity():
    qm, env, field, ref = fixture()
    original = ref.evaluate(qm, env)
    for shift in (np.array([6.0, -14.0, 8.0]), np.array([0.17, -0.21, 0.19])):
        moved = ref.evaluate(dataclasses.replace(qm, positions=qm.positions + shift),
                             dataclasses.replace(env, positions=env.positions + shift))
        assert moved.energy_ev == pytest.approx(original.energy_ev, abs=1e-14)
        np.testing.assert_allclose(moved.qm_forces_ev_per_angstrom,
                                   original.qm_forces_ev_per_angstrom, atol=1e-14)
        np.testing.assert_allclose(moved.environment_forces_ev_per_angstrom,
                                   original.environment_forces_ev_per_angstrom, atol=1e-14)
    xyz = env.positions.copy()
    xyz[0, 1] += 7.0
    image = ref.evaluate(qm, dataclasses.replace(env, positions=xyz))
    assert image.energy_ev == pytest.approx(original.energy_ev, abs=1e-14)
    for values in (field.charges_e, field.sigma_angstrom, field.cell_angstrom):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    record = field.record()
    record["charges_e"][0] = 1
    assert field.charges_e[0] == 0.3


@pytest.mark.parametrize("change", [
    {"charges_e": [0.3, 0.2]}, {"sigma_angstrom": [0.8, 0]},
    {"sigma_angstrom": [float("nan"), 1]}, {"cell_angstrom": [6, 0, 8]},
    {"cell_angstrom": [[6, 0, 0], [0, 7, 0], [0, 0, 8]]},
    {"kmax": True}, {"kmax": 0}, {"kmax": 33}, {"source": ""},
    {"atom_ids": [7, 7]}, {"charges_e": [True, False]},
])
def test_field_rejects_unsupported_or_ambiguous_inputs(change):
    _, _, field, _ = fixture()
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(field, **change)


def test_geometry_and_reference_identity_fail_closed():
    qm, env, field, ref = fixture()
    for geometry in (dataclasses.replace(env, cell=None),
                     dataclasses.replace(env, cell=np.diag([6.0, 7.0, 8.1])),
                     Geometry((5, 7), env.symbols, env.positions, env.cell)):
        with pytest.raises(ValueError):
            ref.evaluate(qm, geometry)
    with pytest.raises(ValueError):
        PeriodicGaussianReference(FixedEmbeddingCharges(qm.atom_ids, [0.2, 0.2], "bad"), field)
    zero = PeriodicGaussianReference(ref.qm_charges, dataclasses.replace(field, charges_e=[0, 0]))
    value = zero.evaluate(qm, env)
    assert value.energy_ev == 0
    assert np.count_nonzero(value.qm_forces_ev_per_angstrom) == 0
    assert np.count_nonzero(value.environment_forces_ev_per_angstrom) == 0
