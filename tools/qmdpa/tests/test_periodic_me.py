# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Same-cell periodic contracts; analytic controls do not qualify real backends."""

from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_abacus_provider import PROVIDER, request
from test_dispatch import save
from test_lammps_bridge import Driver

from qmdpa.abacus import map_abacus_coordinates, validate_abacus_settings
from qmdpa.abacus_worker import _geometry_and_atoms
from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.lammps import ExternalCoupling
from qmdpa.providers import make_qm_provider, provider_contract, validate_provider_regions
from qmdpa.region import FixedRegion


def periodic_request(tmp_path):
    config = request(tmp_path)
    config["periodic"] = True
    config["qm"].update(boundary="periodic-same-cell", coordinate_gauge="periodic-wrap",
                        coordinate_offset_angstrom=[0, 0, 0])
    return config


class PeriodicPair:
    def __init__(self, stiffness):
        self.stiffness = stiffness
        self.calls = []

    def evaluate(self, geometry):
        assert geometry.periodic
        self.calls.append(geometry)
        scale = 2 * np.pi / np.diag(geometry.cell)
        energy = 0.0
        forces = np.zeros_like(geometry.positions)
        for i in range(len(forces)):
            for j in range(i):
                angle = (geometry.positions[i] - geometry.positions[j]) * scale
                energy += self.stiffness * np.sum(1 - np.cos(angle))
                force = -self.stiffness * np.sin(angle) * scale
                forces[i] += force
                forces[j] -= force
        return Evaluation(energy, forces)


def geometry():
    return Geometry((10000000001, 9, 31, 7), ("H",) * 4,
                    [[-0.2, 0.3, 0.4], [0.6, 0.5, 0.2], [3.1, 2.7, 3.3], [3.6, 3.0, 3.8]],
                    np.diag([8.0, 9.0, 10.0]))


def test_periodic_total_all_forces_fd_and_environment_identity():
    full = geometry()
    ids = (9, 10000000001)
    low, high = PeriodicPair(1.3), PeriodicPair(2.7)
    calc = SubtractiveCalculator(low, high, region_boundary="periodic-same-cell")
    result = calc.evaluate(full, full.select(ids))
    np.testing.assert_allclose(result.forces[2:], low.evaluate(full).forces[2:], atol=1e-14)
    for i in range(4):
        for axis in range(3):
            energies = []
            for sign in (-1, 1):
                xyz = full.positions.copy()
                xyz[i, axis] += sign * 1e-5
                moved = Geometry(full.atom_ids, full.symbols, xyz, full.cell)
                energies.append(calc.evaluate(moved, moved.select(ids)).energy)
            assert abs(-(energies[1] - energies[0]) / 2e-5 - result.forces[i, axis]) < 2e-9
    expected = high.evaluate(full)
    actual = calc.evaluate(full, full)
    assert actual.energy == pytest.approx(expected.energy, abs=1e-13)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=1e-13)


def test_individual_images_and_permuted_ids_preserve_periodic_total():
    full = geometry()
    calc = SubtractiveCalculator(PeriodicPair(1), PeriodicPair(3),
                                 region_boundary="periodic-same-cell")
    qm = full.select((9, 10000000001))
    shifted = Geometry(qm.atom_ids, qm.symbols, qm.positions + [[8, -18, 0], [-16, 0, 30]], qm.cell)
    a, b = calc.evaluate(full, qm), calc.evaluate(full, shifted)
    assert a.energy == pytest.approx(b.energy, abs=1e-12)
    np.testing.assert_allclose(a.forces, b.forces, atol=1e-12)


@pytest.mark.parametrize("change", ["finite-full", "finite-qm", "different-cell", "bad-map"])
def test_inconsistent_periodic_geometry_rejected_before_backend(change):
    full = geometry()
    qm = full.select((9, 10000000001))
    if change == "finite-full":
        full = Geometry(full.atom_ids, full.symbols, full.positions)
    elif change == "finite-qm":
        qm = Geometry(qm.atom_ids, qm.symbols, qm.positions)
    elif change == "different-cell":
        qm = Geometry(qm.atom_ids, qm.symbols, qm.positions, qm.cell * 1.01)
    else:
        qm = Geometry(qm.atom_ids, qm.symbols, qm.positions + 0.1, qm.cell)
    backend = Mock()
    with pytest.raises(ValueError):
        SubtractiveCalculator(backend, backend, region_boundary="periodic-same-cell").evaluate(full, qm)
    backend.evaluate.assert_not_called()


def test_boundary_cannot_be_inferred_or_silently_changed():
    full = geometry()
    with pytest.raises(ValueError, match="isolated"):
        SubtractiveCalculator(Mock(), Mock()).correction(full)
    with pytest.raises(ValueError, match="explicit cell"):
        SubtractiveCalculator(Mock(), Mock(), region_boundary="periodic-same-cell").correction(
            full.select(full.atom_ids, isolated=True))
    with pytest.raises(ValueError):
        SubtractiveCalculator(Mock(), Mock(), region_boundary="automatic")


def test_explicit_boundary_description_and_restart_identity(tmp_path):
    config = periodic_request(tmp_path)
    loaded = load_run_config(save(tmp_path, config))
    info = describe_run(loaded)
    assert info["provider_contract"]["qm_boundary"] == "periodic-same-cell"
    assert info["provider_contract"]["coordinate_mapping"] == "periodic-wrap"
    assert info["boundaries"]["region_low"] == "periodic-same-cell"
    assert info["boundaries"]["stress"] == "unsupported"
    assert info["energy_contract"]["total_terms"][-1]["name"] == "periodic_ground"
    original = loaded["physics_sha256"]
    config["qm"]["supercell_angstrom"] = [14, 14, 14]
    assert load_run_config(save(tmp_path, config))["physics_sha256"] != original
    config = request(tmp_path)
    finite = load_run_config(save(tmp_path, config))
    assert "boundary" not in finite["qm"]
    assert finite["provider_contract"] == provider_contract(PROVIDER)
    assert finite["physics_sha256"] != original


@pytest.mark.parametrize("change", ["global-finite", "offset", "centroid", "socket", "unknown",
                                   "implicit", "selected", "gpu", "electrostatic"])
def test_reject_unsupported_periodic_combinations_before_construction(tmp_path, change):
    config = periodic_request(tmp_path)
    if change == "global-finite":
        config["periodic"] = False
    elif change == "offset":
        config["qm"]["coordinate_offset_angstrom"] = [1, 0, 0]
    elif change == "centroid":
        config["qm"]["coordinate_gauge"] = "arithmetic-centroid"
    elif change == "socket":
        config["qm"]["execution_mode"] = "socket"
    elif change == "unknown":
        config["qm"]["boundary"] = "periodic-2d"
    elif change == "implicit":
        del config["qm"]["boundary"]
    elif change == "selected":
        config["provider"]["state"] = {"method": "pbe0-tda"}
    elif change == "gpu":
        config["provider"]["device"] = "cuda"
    else:
        config["coupling"] = {"surface": "ground", "embedding": "electrostatic"}
    with patch("qmdpa.abacus.AbacusBackend") as engine:
        with pytest.raises(ValueError):
            load_run_config(save(tmp_path, config))
    engine.assert_not_called()


def test_provider_admission_is_not_transferred_to_other_engines():
    with pytest.raises(ValueError):
        provider_contract(region_boundary="periodic-same-cell")
    with pytest.raises(ValueError):
        validate_provider_regions(None, {}, periodic=True, region_boundary="periodic-same-cell")


def test_periodic_mapping_and_worker_receive_identical_cell(tmp_path):
    qm = validate_abacus_settings(periodic_request(tmp_path)["qm"], tmp_path)
    geom = Geometry((9, 3), ("H", "H"), [[-0.2, 12.1, 6.0], [12.6, -12.2, 6.0]], np.eye(3) * 12)
    expected = [[11.8, 0.1, 6.0], [0.6, 11.8, 6.0]]
    np.testing.assert_allclose(map_abacus_coordinates(geom, qm), expected, atol=1e-14)
    ase = ModuleType("ase")
    ase.Atoms = Mock()
    with patch.dict("sys.modules", {"ase": ase}):
        restored, _ = _geometry_and_atoms({"geometry": {
            "atom_ids": list(geom.atom_ids), "symbols": list(geom.symbols),
            "positions_angstrom": geom.positions.tolist(), "cell_angstrom": geom.cell.tolist(),
        }}, qm)
    np.testing.assert_array_equal(restored.cell, geom.cell)
    np.testing.assert_allclose(ase.Atoms.call_args.kwargs["positions"], expected, atol=1e-14)
    assert ase.Atoms.call_args.kwargs["cell"] == [12, 12, 12]


@pytest.mark.parametrize("cell", [None, np.diag([12, 12, 13]), [[12, 0, 0], [1, 12, 0], [0, 0, 12]]])
def test_actual_cell_mismatch_permanently_fails_before_worker(tmp_path, cell):
    qm = validate_abacus_settings(periodic_request(tmp_path)["qm"], tmp_path)
    provider = make_qm_provider(qm, provider=PROVIDER, directory=tmp_path / "runtime")
    geom = Geometry((9, 3), ("H", "H"), [[0, 0, 0], [0.8, 0, 0]], cell)
    with patch.object(provider, "_start") as start:
        for _ in range(2):
            with pytest.raises(EvaluationError, match="orthorhombic cell"):
                provider.evaluate(geom)
        start.assert_not_called()
    provider.close_checked()
    assert provider.statistics["failed_evaluations"] == 1
    assert not (provider.directory / "worker.json").exists()


def test_native_callback_passes_fixed_cell_and_maps_correction_ids():
    driver = Driver()
    region = FixedRegion((22, 5), ("O", "H"))
    provider = Mock()
    provider.correction.return_value = Evaluation(2.3, [[1, 2, 3], [4, 5, 6]])
    coupling = ExternalCoupling(driver, region, provider, region_boundary="periodic-same-cell")
    coupling.run(0)
    geom = provider.correction.call_args.args[0]
    assert geom.atom_ids == (22, 5)
    np.testing.assert_array_equal(geom.cell, np.eye(3) * 8)
    np.testing.assert_array_equal(driver.forces, [[4, 5, 6], [0, 0, 0], [1, 2, 3]])
    assert not driver.swallowed
    box = list(driver.box)
    box[1] = [9, 8, 8]
    driver.box = tuple(box)
    with pytest.raises(EvaluationError, match="fixed simulation box"):
        coupling.run(0)
    assert driver.timeout and provider.correction.call_count == 1


@pytest.mark.parametrize("change", ["partial", "tilted"])
def test_native_periodic_cell_admission(change):
    driver = Driver()
    box = list(driver.box)
    if change == "partial":
        box[5] = [1, 1, 0]
    else:
        box[2] = 0.5
    driver.box = tuple(box)
    with pytest.raises(ValueError, match="orthorhombic"):
        ExternalCoupling(driver, FixedRegion((22, 5), ("O", "H")), Mock(),
                         region_boundary="periodic-same-cell")
