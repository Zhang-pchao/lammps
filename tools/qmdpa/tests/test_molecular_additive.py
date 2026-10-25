# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Independent derivative, charge identity and native molecular additive contracts."""

import json
import os
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from test_adaptive import PairPolynomial, displaced
from test_adaptive_cli import providers, torch_state
from test_adaptive_config import load
from test_additive_lammps_bridge import AdditiveWorkerComm
from test_molecular_adaptive import geometry, groups, switch
from test_molecular_config import MolecularDriver
from test_molecular_config import document as mechanical_document

from qmdpa import dispatch
from qmdpa.adaptive_config import build_mixer, molecular_environment_charges
from qmdpa.additive import AdditiveCalculator, CrossLennardJones
from qmdpa.core import EvaluationError
from qmdpa.embedding import EmbeddedEvaluation, FixedEmbeddingCharges
from qmdpa.inference import RAW_ENV
from qmdpa.molecular_additive import AdditivePartition, MolecularAdditiveMixer, subset_charges


class AnalyticEmbedded:
    """A charge-dependent analytic cross potential plus an internal QM control."""

    def __init__(self, charges):
        self.charges = charges
        self.charge, self.multiplicity = 0, 1
        self.xc, self.basis = "hf", "sto-3g"
        self.calls = 0
        self.failed = False

    def evaluate(self, qm, env):
        self.calls += 1
        if self.failed:
            raise EvaluationError("Deliberate embedded failure")
        own = PairPolynomial(.07, -5).evaluate(qm)
        energy = own.energy
        qforce, eforce = own.forces.copy(), np.zeros_like(env.positions)
        for i in range(len(qm.atom_ids)):
            for j, charge in enumerate(self.charges.charges_e):
                delta = qm.positions[i] - env.positions[j]
                energy += .5 * charge * np.dot(delta, delta)
                qforce[i] -= charge * delta
                eforce[j] += charge * delta
        return EmbeddedEvaluation(energy, qm.atom_ids, env.atom_ids, qforce, eforce)


def charges():
    return FixedEmbeddingCharges(tuple(range(1, 9)), [.04, -.04] * 4, "analytic charge table")


def partitions(k=1):
    parts = []
    for subset in switch(k).group_subsets:
        names = ("core",) + subset
        qm_ids = tuple(sorted(a for g in groups() if g.group_id in names for a in g.atom_ids))
        env_ids = tuple(a for a in range(1, 9) if a not in qm_ids)
        calc = AdditiveCalculator(PairPolynomial(.04, -10),
            AnalyticEmbedded(subset_charges(charges(), env_ids)),
            CrossLennardJones({("H", "H"): (.002, 1.2)}, source="analytic LJ control"),
            qm_atom_ids=qm_ids, environment_atom_ids=env_ids)
        parts.append(AdditivePartition("".join(subset), names, calc, 0, 1,
                                      "analytic-embedded", "same-reference"))
    return tuple(parts)


def mixer(k=1, **changes):
    args = dict(groups=groups(), core_group_ids=("core",), partitions=partitions(k),
                switch=switch(k), charges=charges())
    args.update(changes)
    return MolecularAdditiveMixer(**args)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("h", [1e-4, 5e-5])
def test_complete_energy_derivatives_on_every_atom(k, h):
    obj, atoms = mixer(k), geometry()
    result = obj.evaluate_terms(atoms)
    fd = np.zeros_like(atoms.positions)
    for row in range(8):
        for axis in range(3):
            energies = [obj.evaluate(displaced(atoms, row, axis, s * h)).energy for s in (-2, -1, 1, 2)]
            fd[row, axis] = -(energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]) / (12 * h)
    np.testing.assert_allclose(result.total.forces, fd, atol=2e-7, rtol=0)
    np.testing.assert_allclose(result.total.forces.sum(axis=0), 0, atol=1e-12, rtol=0)
    assert np.max(np.abs(result.switching_forces)) > .01
    assert np.max(np.abs(result.total.forces - result.switching_forces - fd)) > .01


def test_complete_partition_sum_has_no_full_baseline_and_common_offset_cancels():
    atoms, obj = geometry(), mixer()
    result = obj.evaluate_terms(atoms)
    energies = np.array([p.calculator.evaluate(atoms).energy for p in obj.partitions])
    assert result.total.energy == pytest.approx(result.weights.weights @ energies, abs=1e-12)
    common = [replace(p, energy_offset_ev=1000., offset_source="common declared shift")
              for p in partitions()]
    shifted = mixer(partitions=common).evaluate(atoms)
    assert shifted.energy - result.total.energy == pytest.approx(1000., abs=1e-11)
    np.testing.assert_allclose(shifted.forces, result.total.forces, atol=5e-12, rtol=0)
    common[0] = replace(common[0], energy_offset_ev=1002.)
    changed = mixer(partitions=common).evaluate(atoms)
    np.testing.assert_allclose(changed.forces - shifted.forces,
                              -2 * result.weights.gradients_per_angstrom[0], atol=1e-12, rtol=0)


def test_reordered_input_and_charge_table_preserve_full_forces():
    atoms, obj = geometry(), mixer()
    base = obj.evaluate(atoms)
    ids = (8, 3, 5, 1, 4, 2, 7, 6)
    table = subset_charges(charges(), ids)
    reordered = mixer(charges=table).evaluate(atoms.select(ids))
    assert reordered.energy == base.energy
    np.testing.assert_array_equal(reordered.forces, base.forces[[a - 1 for a in ids]])
    assert tuple(table.charges_e) == tuple(charges().charges_e[a - 1] for a in ids)


def test_exact_zero_pruning_reactivation_and_coverage_fail_before_work():
    obj, atoms = mixer(), geometry()
    xyz = atoms.positions.copy()
    xyz[2:4] = [[1.8, 0, 0], [1.8, 0, .74]]
    xyz[4:6] += [0, 2., 0]
    xyz[6:8] += [-2., 0, 0]
    pure = replace(atoms, positions=xyz)
    value = obj.evaluate_terms(pure)
    assert value.active_partition_ids == ("a",)
    assert value.partition_totals[1:] == (None, None)
    np.testing.assert_array_equal(value.total.forces, obj.partitions[0].calculator.evaluate(pure).forces)
    obj.evaluate(atoms)
    assert [p.calculator.embedded_backend.calls for p in obj.partitions] == [3, 1, 1]
    before = [p.calculator.environment_backend.calls for p in obj.partitions]
    xyz[2:] *= 3
    with pytest.raises(EvaluationError, match="Too few"):
        obj.evaluate(replace(atoms, positions=xyz))
    assert [p.calculator.environment_backend.calls for p in obj.partitions] == before


@pytest.mark.parametrize("bad", ["values", "source", "radii", "ids", "state", "missing", "coverage"])
def test_invalid_or_mutated_field_is_rejected_before_any_evaluation(bad):
    parts, table = list(partitions()), charges()
    high = parts[-1].calculator.embedded_backend
    if bad == "values":
        high.charges = replace(high.charges, charges_e=-high.charges.charges_e)
    elif bad == "source":
        high.charges = replace(high.charges, source="different")
    elif bad == "radii":
        high.charges = replace(high.charges, radii_angstrom=[.3] * 4)
    elif bad == "ids":
        high.charges = subset_charges(high.charges, reversed(high.charges.atom_ids))
    elif bad == "state":
        high.charge = 1
    elif bad == "missing":
        parts.pop()
    else:
        table = subset_charges(table, tuple(range(1, 8)))
    with pytest.raises((ValueError, TypeError)):
        mixer(partitions=parts, charges=table)
    assert all(p.calculator.environment_backend.calls == 0 for p in parts)


def test_late_provider_failure_returns_no_partial_total_and_retry_is_explicit():
    obj = mixer()
    obj.partitions[-1].calculator.embedded_backend.failed = True
    with pytest.raises(EvaluationError, match="embedded failure"):
        obj.evaluate(geometry())
    obj.partitions[-1].calculator.embedded_backend.failed = False
    np.testing.assert_array_equal(obj.evaluate(geometry()).forces, mixer().evaluate(geometry()).forces)
    obj.partitions[-1].calculator.embedded_backend.charges = replace(
        obj.partitions[-1].calculator.embedded_backend.charges, source="mutated after construction")
    before = [p.calculator.environment_backend.calls for p in obj.partitions]
    with pytest.raises(ValueError, match="charge table"):
        obj.evaluate(geometry())
    assert [p.calculator.environment_backend.calls for p in obj.partitions] == before


def document():
    value = mechanical_document()
    value.pop("full_charge_spin")
    value.pop("region_charge_spin")
    value.update(hamiltonian="finite-molecular-additive", environment_charge_spin=[0, 1],
        embedding={"atom_ids": list(range(1, 9)), "charges_e": [.04, -.04] * 4,
                   "source": "analytic charge table"},
        cross={"form": "lj12-6", "source": "analytic LJ control", "cutoff": None,
               "mixing": "none", "pairs": [{"qm_symbol": "H", "environment_symbol": "H",
                                          "epsilon_ev": .002, "sigma_angstrom": 1.2}]})
    value["qm"].update(xc="hf", basis="sto-3g")
    return value


def test_unified_selector_and_full_charge_table_are_in_restart_identity(tmp_path):
    cfg = load(tmp_path, document())
    desc = dispatch.describe_run(cfg)
    assert desc["assembly"]["native_pair"] == "zero"
    assert desc["provider_contract"]["force_coordinates"] == ["qm", "environment"]
    assert desc["energy_contract"]["partition_terms"] == [
        "low_environment", "embedded_ground", "cross_lj", "declared_offset"]
    assert desc["coupling"]["ee_model"] == "molecular-additive"
    raw = document()
    raw.pop("hamiltonian")
    raw["coupling"] = {"surface": "ground", "embedding": "electrostatic", "ee_model": "molecular-additive"}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with patch("qmdpa.adaptive_config.importlib.metadata.version", return_value="fixture"):
        assert dispatch.load_run_config(tmp_path / "config.json")["physics_sha256"] == cfg["physics_sha256"]
    reordered = document()
    for field in ("atom_ids", "charges_e"):
        reordered["embedding"][field].reverse()
    reordered["partitions"].reverse()
    assert load(tmp_path, reordered)["physics_sha256"] == cfg["physics_sha256"]
    for change in ("charge", "source", "cross", "radius"):
        other = document()
        if change == "charge":
            other["embedding"]["charges_e"][0:2] = [.02, -.02]
        elif change == "source":
            other["embedding"]["source"] = "different explicit source"
        elif change == "cross":
            other["cross"]["pairs"][0]["epsilon_ev"] = .003
        else:
            other["embedding"]["radii_angstrom"] = [.3] * 8
        assert load(tmp_path, other)["physics_sha256"] != cfg["physics_sha256"]


@pytest.mark.parametrize("bad", ["missing", "extra", "duplicate", "charge", "pair", "me-field", "selected"])
def test_invalid_fields_fail_before_backend_allocation(tmp_path, bad):
    raw = document()
    if bad == "missing":
        raw["embedding"]["atom_ids"].pop()
        raw["embedding"]["charges_e"].pop()
    elif bad == "extra":
        raw["embedding"]["unknown"] = True
    elif bad == "duplicate":
        raw["embedding"]["atom_ids"][-1] = 7
    elif bad == "charge":
        raw["embedding"]["charges_e"][2] += .1
    elif bad == "pair":
        raw["cross"]["pairs"] = []
    elif bad == "me-field":
        raw["full_charge_spin"] = [0, 1]
    else:
        raw["provider"]["state"]["method"] = "rhf-tda"
    with patch("qmdpa.backends.DeepMDBackend") as low:
        with pytest.raises((ValueError, TypeError)):
            load(tmp_path, raw)
    low.assert_not_called()


def test_native_factory_reuses_environment_model_and_preserves_complete_force(tmp_path):
    cfg = load(tmp_path, document())
    lmp, torch = MolecularDriver(), torch_state()
    lows, _, low, _ = providers(cfg, torch)
    highs = []
    def high(qm, *, provider, charges):
        obj = AnalyticEmbedded(charges)
        highs.append(obj)
        return obj
    with patch.dict(os.environ, RAW_ENV), \
            patch("qmdpa.inference.prepare_raw_inference", return_value=torch), \
            patch("qmdpa.backends.DeepMDBackend", side_effect=low), \
            patch("qmdpa.providers.make_qm_provider", side_effect=high):
        coupling = dispatch._make_coupling(lmp, None, cfg)
        assert len(lows) == 1 and len(highs) == 3
        energy = coupling.run(0)
        reference = build_mixer(cfg, PairPolynomial(.04, -10),
            [AnalyticEmbedded(c) for c in molecular_environment_charges(cfg)]).evaluate(geometry())
        assert energy == pytest.approx(reference.energy, abs=1e-12)
        np.testing.assert_array_equal(lmp.forces, reference.forces[[a - 1 for a in lmp.tags]])
        assert lows[0].calls == 3
        assert cfg["inference_runtime"]["shared_full_and_region_model"] is False
        highs[-1].failed = True
        with pytest.raises(EvaluationError, match="embedded failure"):
            coupling.run(1)
        assert coupling.last_step == 0
        np.testing.assert_array_equal(lmp.forces, np.zeros((8, 3)))
    comm = AdditiveWorkerComm([(None, None), (None, None)])
    with patch("qmdpa.backends.DeepMDBackend") as low, patch("qmdpa.providers.make_qm_provider") as high:
        worker = dispatch._make_coupling(MolecularDriver(2), comm, cfg)
    low.assert_not_called()
    high.assert_not_called()
    assert worker.calculator is None
