# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite molecular corrections inside a periodic full-system potential."""

from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_gpu_provider import GPU, case
from test_periodic_me import PeriodicPair, geometry

from qmdpa.core import Evaluation, Geometry, SubtractiveCalculator
from qmdpa.dispatch import describe_run
from qmdpa.providers import validate_provider_regions
from qmdpa.region import unwrap

__all__ = ["case"]


class FinitePair:
    def __init__(self, stiffness):
        self.stiffness = stiffness

    def evaluate(self, geom):
        assert not geom.periodic
        forces = np.zeros_like(geom.positions)
        energy = 0.0
        for i in range(len(forces)):
            for j in range(i):
                delta = geom.positions[i] - geom.positions[j]
                energy += 0.5 * self.stiffness * np.dot(delta, delta)
                forces[i] -= self.stiffness * delta
                forces[j] += self.stiffness * delta
        return Evaluation(energy, forces)


@pytest.mark.parametrize("engine,device", [("pyscf", "cpu"), ("gpu4pyscf", "cuda")])
def test_three_boundaries_are_separate_without_constructing_qm(case, engine, device):
    case.config["periodic"] = True
    case.config["provider"] = {"engine": engine, "device": device,
                               "state": {"method": "ground"}}
    with patch("qmdpa.gpu_backends.GPU4PySCFBackend") as gpu:
        info = describe_run(case.load())
    gpu.assert_not_called()
    assert info["boundaries"] == {
        "full_low": "periodic-3d", "region_low": "isolated-molecule",
        "qm": "isolated-molecule", "embedding": "none",
        "coordinate_mapping": "native-unwrapped-images",
        "cell": "fixed-native-cell", "all_qm_cancellation": "not-guaranteed",
        "stress": "unsupported",
    }
    assert info["assembly"]["native_pair"] == "deepmd"
    assert info["assembly"]["external_quantity"] == "qm-minus-region-low-correction"


@pytest.mark.parametrize("change", ["environment", "charges", "periodic-qm", "unspecified-pbc"])
def test_gpu_periodic_admission_does_not_leak_to_other_boundaries(change):
    options = {"periodic": True}
    if change == "environment":
        options["environment_region"] = {"atom_ids": [2], "symbols": ["H"]}
    elif change == "charges":
        options["charges"] = object()
    elif change == "periodic-qm":
        options["region_boundary"] = "periodic-same-cell"
    else:
        options["periodic"] = None
    with pytest.raises(ValueError):
        validate_provider_regions(deepcopy(GPU), {"atom_ids": [1], "symbols": ["H"]},
                                  **options)


def test_periodic_me_preserves_complete_force_and_has_no_environment_correction():
    full = geometry()
    ids = (9, 10000000001)
    low, high, full_low = FinitePair(1.3), FinitePair(2.7), PeriodicPair(0.8)
    calc = SubtractiveCalculator(low, high, full_low=full_low)
    result = calc.evaluate(full, full.select(ids, isolated=True))
    np.testing.assert_allclose(result.forces[2:], full_low.evaluate(full).forces[2:], atol=1e-14)
    for atom in range(len(full.atom_ids)):
        for axis in range(3):
            energies = []
            for sign in (-1, 1):
                xyz = full.positions.copy()
                xyz[atom, axis] += sign * 1e-5
                moved = Geometry(full.atom_ids, full.symbols, xyz, full.cell)
                energies.append(calc.evaluate(moved, moved.select(ids, isolated=True)).energy)
            assert abs(-(energies[1] - energies[0]) / 2e-5 - result.forces[atom, axis]) < 2e-9


def test_equivalent_wrapped_coordinates_and_images_preserve_the_finite_molecule():
    full = geometry()
    images = np.array([[2, -1, 3], [-1, 1, 2], [0, 2, -2], [3, 0, 1]])
    wrapped = full.positions - images @ full.cell
    recovered = unwrap(wrapped, images, full.cell)
    np.testing.assert_allclose(recovered, full.positions, atol=5e-15)
    calc = SubtractiveCalculator(FinitePair(1), FinitePair(3), full_low=PeriodicPair(2))
    selected = full.select((9, 10000000001), isolated=True)
    represented = Geometry(full.atom_ids, full.symbols, wrapped, full.cell)
    actual = calc.evaluate(represented, selected)
    expected = calc.evaluate(full, selected)
    assert actual.energy == pytest.approx(expected.energy, abs=2e-12)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=2e-13)


def test_b2_all_qm_retains_the_full_periodic_minus_finite_boundary_residual():
    full = geometry()
    finite = full.select(full.atom_ids, isolated=True)
    low, high, full_low = FinitePair(1), FinitePair(3), PeriodicPair(2)
    result = SubtractiveCalculator(low, high, full_low=full_low).evaluate(full, finite)
    residual = full_low.evaluate(full).energy - low.evaluate(finite).energy
    assert abs(residual) > 1e-3
    assert result.energy - high.evaluate(finite).energy == pytest.approx(residual, abs=1e-12)


def test_periodic_molecular_provider_still_rejects_a_periodic_qm_geometry():
    full = geometry()
    backend = Mock()
    with pytest.raises(ValueError, match="isolated"):
        SubtractiveCalculator(backend, backend).correction(full)
    backend.evaluate.assert_not_called()
