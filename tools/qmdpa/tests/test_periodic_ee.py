# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Periodic total, configuration and native transport contracts."""

import copy
import dataclasses
from unittest.mock import Mock

import numpy as np
import pytest
from test_additive_lammps_bridge import AdditiveDriver, GatherComm, TotalCalculator
from test_dispatch import save
from test_periodic_abacus import inputs
from test_periodic_me import PeriodicPair, periodic_request

from qmdpa.core import EvaluationError, Geometry
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.embedding import EmbeddedEvaluation, FixedEmbeddingCharges
from qmdpa.inference import RAW_PROFILE
from qmdpa.lammps import AdditiveExternalCoupling, PeriodicReferenceExternalCoupling
from qmdpa.periodic_ee import PeriodicReferenceSubtractiveCalculator
from qmdpa.periodic_ee_config import HAMILTONIAN
from qmdpa.periodic_field import PeriodicGaussianReference
from qmdpa.region import FixedRegion


class EmbeddedModel:
    def __init__(self, reference):
        self.reference = reference

    def evaluate(self, qm, env):
        own = PeriodicPair(2.3).evaluate(qm)
        cross = self.reference.evaluate(qm, env)
        return EmbeddedEvaluation(own.energy + 1.7 * cross.energy_ev, qm.atom_ids, env.atom_ids,
                                  own.forces + 1.7 * cross.qm_forces_ev_per_angstrom,
                                  1.7 * cross.environment_forces_ev_per_angstrom)


def fixture(tmp_path):
    _, field, qm, env = inputs(tmp_path)
    ref = PeriodicGaussianReference(FixedEmbeddingCharges(qm.atom_ids, [0.2, -0.2],
                                                         source="test reference"), field)
    low = PeriodicPair(1.3)
    calculator = PeriodicReferenceSubtractiveCalculator(
        low, low, EmbeddedModel(ref), ref, qm_atom_ids=qm.atom_ids,
        environment_atom_ids=env.atom_ids)
    full = Geometry(qm.atom_ids + env.atom_ids, qm.symbols + env.symbols,
                    np.concatenate((qm.positions, env.positions)), qm.cell)
    return full, calculator


def test_complete_total_all_coordinate_fd_mapping_and_images(tmp_path):
    full, calculator = fixture(tmp_path)
    result = calculator.evaluate_terms(full)
    assert result.model == HAMILTONIAN
    assert result.total.energy == pytest.approx(result.full_low.energy + result.embedded.energy_ev
                                               - result.region_low.energy - result.reference_cross.energy_ev)
    for step in (1e-5, 5e-6):
        for row in range(len(full.atom_ids)):
            for axis in range(3):
                energies = []
                for sign in (-1, 1):
                    xyz = full.positions.copy()
                    xyz[row, axis] += sign * step
                    energies.append(calculator.evaluate(dataclasses.replace(full, positions=xyz)).energy)
                assert abs(-(energies[1] - energies[0]) / (2 * step)
                           - result.total.forces[row, axis]) < 2e-8
    np.testing.assert_allclose(result.total.forces.sum(axis=0), 0, atol=2e-14)
    order = (12, 2, 11, 1)
    permuted = calculator.evaluate(full.select(order))
    assert permuted.energy == pytest.approx(result.total.energy, abs=1e-13)
    np.testing.assert_allclose(permuted.forces, result.total.forces[[3, 1, 2, 0]], atol=1e-13)
    shifted = full.positions.copy()
    shifted[2] += [6, -12, 18]
    image = calculator.evaluate(dataclasses.replace(full, positions=shifted))
    assert image.energy == pytest.approx(result.total.energy, abs=1e-13)
    np.testing.assert_allclose(image.forces, result.total.forces, atol=1e-13)


@pytest.mark.parametrize("kind", ["finite", "cell", "symbol", "coverage"])
def test_bad_geometry_rejected_before_any_provider(tmp_path, kind):
    full, calculator = fixture(tmp_path)
    if kind == "symbol":
        calculator.evaluate(full)
        full = dataclasses.replace(full, symbols=("He", "H", "H", "H"))
    elif kind == "finite":
        full = dataclasses.replace(full, cell=None)
    elif kind == "cell":
        full = dataclasses.replace(full, cell=full.cell * 1.01)
    else:
        full = full.select((1, 2, 11))
    calculator.full_low_backend = Mock()
    with pytest.raises(ValueError):
        calculator.evaluate(full)
    calculator.full_low_backend.evaluate.assert_not_called()


def request(tmp_path):
    raw = periodic_request(tmp_path)
    raw["hamiltonian"] = HAMILTONIAN
    raw["qm_region"] = raw.pop("region")
    raw["environment_region"] = {"atom_ids": [11, 12], "symbols": ["H", "H"],
                                 "molecules": [[11], [12]]}
    settings, _, _, _ = inputs(tmp_path)
    raw["qm"] = settings
    raw["reference"] = {"form": PeriodicGaussianReference.model,
                        "qm_charges": {"atom_ids": [1, 2], "charges_e": [0.2, -0.2],
                                       "source": "explicit reference"}}
    raw["inference_profile"] = RAW_PROFILE
    return raw


def test_route_description_and_restart_identity(tmp_path):
    raw = request(tmp_path)
    direct = load_run_config(save(tmp_path, raw))
    description = describe_run(direct)
    assert description["assembly"]["native_pair"] == "zero"
    assert description["boundaries"]["qm"] == "periodic-same-cell"
    assert description["provider_contract"]["force_coordinates"] == ["qm", "environment"]
    assert [term["coefficient"] for term in description["energy_contract"]["total_terms"]] == [1, -1, 1, -1]
    selected = copy.deepcopy(raw)
    selected.pop("hamiltonian")
    selected["coupling"] = {"surface": "ground", "embedding": "electrostatic",
                            "ee_model": "periodic-reference-subtractive"}
    assert load_run_config(save(tmp_path, selected))["physics_sha256"] == direct["physics_sha256"]
    for changed in ("width", "charge", "cube", "reference", "source", "cell", "density-cutoff"):
        other = copy.deepcopy(raw)
        field = other["qm"]["periodic_field"]
        if changed == "width":
            field["sigma_angstrom"][0] += 0.01
        elif changed == "charge":
            field["charges_e"] = [0.2, -0.2]
        elif changed == "cube":
            field["kmax"] += 1
        elif changed == "reference":
            other["reference"]["qm_charges"]["charges_e"] = [0.1, -0.1]
        elif changed == "density-cutoff":
            other["qm"]["ecutrho"] = 8 * other["qm"].get("ecutwfc", 60)
        elif changed == "source":
            field["source"] += " updated"
        else:
            field["cell_angstrom"][0] += 0.1
            other["qm"]["supercell_angstrom"][0] += 0.1
        assert load_run_config(save(tmp_path, other))["physics_sha256"] != direct["physics_sha256"]


@pytest.mark.parametrize("kind", ["periodic", "provider", "full-state", "region-state", "order",
                                   "form", "charge", "empty", "extra"])
def test_unqualified_combinations_rejected(tmp_path, kind):
    raw = request(tmp_path)
    if kind == "periodic":
        raw["periodic"] = False
    elif kind == "provider":
        raw["provider"]["device"] = "gpu"
    elif kind in ("full-state", "region-state"):
        raw[kind.split("-")[0] + "_charge_spin"] = [1, 2]
    elif kind == "order":
        raw["qm"]["periodic_field"]["atom_ids"].reverse()
    elif kind == "form":
        raw["reference"]["form"] = "fixed-point-charge-cross-coulomb"
    elif kind == "charge":
        raw["reference"]["qm_charges"]["charges_e"] = [0.1, 0.1]
    elif kind == "empty":
        raw["environment_region"] = {"atom_ids": [], "symbols": [], "molecules": []}
    else:
        raw["qm"]["stress"] = True
    with pytest.raises((ValueError, TypeError)):
        load_run_config(save(tmp_path, raw))


def bridge_fixture():
    lmp = AdditiveDriver()
    lmp.box = (*lmp.box[:5], [1, 1, 1], 0)
    region = FixedRegion((22, 5, 99), ("O", "H", "O"))
    return lmp, region


def test_periodic_zero_pair_transport_reuses_full_atom_guards():
    lmp, region = bridge_fixture()
    calculator = TotalCalculator()
    coupling = PeriodicReferenceExternalCoupling(lmp, region, calculator, type_symbols=("O", "H"))
    lmp.images[:] = [1, 0, -1]
    assert coupling.run(0) == 9.5
    geometry = calculator.geometries[-1]
    np.testing.assert_array_equal(geometry.cell, np.eye(3) * 8)
    np.testing.assert_array_equal(geometry.positions[:, 0], [-8, 8, 0])
    np.testing.assert_array_equal(lmp.forces, [[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    lmp.native_forces[0, 0] = 0.1
    with pytest.raises(EvaluationError, match="exactly zero"):
        coupling.run(1)
    lmp.native_forces[:] = 0
    with pytest.raises(EvaluationError):
        coupling.run(1)
    assert len(calculator.geometries) == 1


def test_periodic_guard_does_not_weaken_finite_route():
    lmp, region = bridge_fixture()
    with pytest.raises(ValueError, match="nonperiodic"):
        AdditiveExternalCoupling(lmp, region, TotalCalculator(), type_symbols=("O", "H"))
    lmp.box = (*lmp.box[:5], [1, 1, 0], 0)
    with pytest.raises(ValueError, match="full PBC"):
        PeriodicReferenceExternalCoupling(lmp, region, TotalCalculator(), type_symbols=("O", "H"))


def test_periodic_transport_collective_remote_identity_failure():
    lmp, region = bridge_fixture()
    lmp.world_size = 2
    comm = GatherComm([22, 99])
    lmp.set_owned((5,))
    coupling = PeriodicReferenceExternalCoupling(lmp, region, TotalCalculator(),
                                                type_symbols=("O", "H"), comm=comm)
    assert coupling.run(0) == 9.5
    comm.remote_ids = [22, 22]
    with pytest.raises(EvaluationError):
        coupling.run(1)
