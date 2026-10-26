#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Independent state correspondence and complete selected-family contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_adaptive import PairPolynomial, displaced
from test_excited import T1, T2, reference, runtime_record
from test_molecular_adaptive import geometry, groups, switch

from qmdpa import excited, state_family
from qmdpa.core import EvaluationError, SubtractiveCalculator
from qmdpa.molecular_adaptive import MolecularMechanicalMixer
from qmdpa.selected_adaptive import MolecularSelectedMixer, SelectedMechanicalPartition
from qmdpa.state_family import (
    FamilyTDABackend,
    PartitionState,
    StateFamilyReference,
    pairwise_correspondence,
)


def family():
    atoms = geometry()
    base = reference()
    core = replace(base, qm_geometry=atoms.select((1, 2)))
    parts = tuple((name, replace(base, qm_geometry=atoms.select((1, 2, *ids))))
                  for name, ids in (("a", (3, 4)), ("b", (5, 6)), ("c", (7, 8))))
    return StateFamilyReference(core, parts)


def test_reference_roundtrip_is_canonical_detached_and_immutable():
    original = family()
    record = original.to_record()
    restored = StateFamilyReference.from_record(json.loads(json.dumps(record)))
    assert restored.sha256 == original.sha256
    assert restored.reference("a").qm_geometry.atom_ids == (1, 2, 3, 4)
    record["partitions"].pop()
    assert len(restored.partitions) == 3
    with pytest.raises(ValueError):
        restored.core.transition_density.flags.writeable = True
    with pytest.raises(ValueError, match="Unknown"):
        restored.reference("missing")
    reordered = replace(original, partitions=tuple(reversed(original.partitions)))
    assert reordered.sha256 == original.sha256


@pytest.mark.parametrize("change", ["hash", "unknown", "schema", "order", "member"])
def test_corrupt_or_noncanonical_records_are_rejected(change):
    record = family().to_record()
    if change == "hash":
        record["sha256"] = "f"*64
    else:
        if change == "unknown":
            record["unexpected"] = None
        elif change == "schema":
            record["schema_version"] = True
        elif change == "order":
            record["partitions"].reverse()
        else:
            record["partitions"][0]["other"] = 1
        record["sha256"] = excited._digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        StateFamilyReference.from_record(record)


@pytest.mark.parametrize("change", ["duplicate_id", "same_members", "core_only",
                                   "core_object", "shift_core", "settings", "one", "order"])
def test_reference_rejects_inconsistent_geometry_and_identity(change):
    obj = family()
    parts = list(obj.partitions)
    if change == "duplicate_id":
        parts[1] = ("a", parts[1][1])
    elif change == "same_members":
        parts[1] = ("b", parts[0][1])
    elif change == "core_only":
        parts[0] = ("a", replace(obj.core))
    elif change == "core_object":
        parts[0] = ("a", obj.core)
    elif change == "shift_core":
        ref = parts[0][1]
        parts[0] = ("a", replace(ref, qm_geometry=displaced(ref.qm_geometry, 0, 0, .1)))
    elif change == "settings":
        parts[0] = ("a", replace(parts[0][1], settings=replace(obj.core.settings, basis="sto-3g")))
    elif change == "one":
        parts = parts[:1]
    else:
        ref = parts[0][1]
        parts[0] = ("a", replace(ref, qm_geometry=ref.qm_geometry.select((4, 3, 2, 1))))
    with pytest.raises(ValueError):
        replace(obj, partitions=tuple(parts))


def solved(transitions=(T1, T2), energies=(2., 3.), matrix=None):
    if matrix is None:
        matrix = np.eye(2)
    return SimpleNamespace(
        transitions=tuple(transitions), energies=np.array(energies),
        molecule=matrix, overlap=matrix.T @ matrix, tda_diagnostics={}, ground_energy_ev=-5.)


def state(name, solution, index):
    return PartitionState(name, geometry(), solution, index, "{}")


def runtime():
    return SimpleNamespace(record=runtime_record(), gto=SimpleNamespace(
        intor_cross=lambda integral, left, right: left.T @ right))


def test_bilateral_correspondence_survives_root_permutation_phase_and_ao_basis_change():
    transform = np.array([[1.1, .2], [.1, .9]])
    inv = np.linalg.inv(transform)
    left = state("a", solved(), 0)
    right = state("b", solved((-inv@T2@inv.T, -inv@T1@inv.T), (3., 2.), transform), 1)
    third = state("c", solved((-T1, T2)), 0)
    reports = pairwise_correspondence((left, right, third), reference().settings, runtime())
    assert len(reports) == 6
    assert {(r["source_partition"], r["target_partition"]) for r in reports} == {
        (a, b) for a in ("a", "b", "c") for b in ("a", "b", "c") if a != b}
    assert all(r["overlap"] == pytest.approx(1) for r in reports)
    assert all(r["selected_root"] == (2 if r["target_partition"] == "b" else 1) for r in reports)
    assert pairwise_correspondence((left,), reference().settings, runtime()) == ()


@pytest.mark.parametrize("reason", ["wrong_root", "weak", "ambiguous", "gap", "duplicate", "empty"])
def test_pair_correspondence_rejects_wrong_or_unresolved_states(reason):
    states = [state("a", solved(), 0), state("b", solved(), 0)]
    if reason == "wrong_root":
        states[1] = replace(states[1], selected_index=1)
    elif reason == "weak":
        states[1] = state("b", solved((np.eye(2), T2)), 0)
    elif reason == "ambiguous":
        states[1] = state("b", solved((T1, T1+.01*T2)), 0)
    elif reason == "gap":
        states[1] = state("b", solved(energies=(2., 2.00001)), 0)
    elif reason == "duplicate":
        states[1] = replace(states[1], partition_id="a")
    else:
        states = []
    with pytest.raises((ValueError, EvaluationError)):
        pairwise_correspondence(states, reference().settings, runtime())


@pytest.fixture
def controlled_provider(monkeypatch):
    rt = runtime()
    molecule = SimpleNamespace(intor_symmetric=lambda _: np.eye(2))
    rt.gto.intor_cross = lambda *args: np.eye(2)
    monkeypatch.setattr(excited, "_make_molecule", lambda *args: molecule)
    monkeypatch.setattr(state_family, "_make_molecule", lambda *args: molecule)
    monkeypatch.setattr(state_family, "_load_runtime", lambda *args: rt)
    monkeypatch.setattr(FamilyTDABackend, "_load_method_runtime", lambda *args: rt)
    monkeypatch.setattr(FamilyTDABackend, "_solve_surface", lambda *args: solved())

    def surface(self, solution, index, diagnostics, atoms, environment, charges, forces):
        value = PairPolynomial(.07, -5).evaluate(atoms)
        diagnostics = {**diagnostics, "ground_energy_ev": value.energy-diagnostics["excitation_energy_ev"]}
        return excited.TDASurfaceEvaluation(
            value.energy, atoms.atom_ids, (), value.forces if forces else None, None, diagnostics)

    monkeypatch.setattr(FamilyTDABackend, "_evaluate_selected_surface", surface)
    return rt


def mixer(fam=None):
    fam = family() if fam is None else fam
    low = PairPolynomial(.04)
    parts = tuple(SelectedMechanicalPartition(
        name, ("core", name), SubtractiveCalculator(low, FamilyTDABackend(fam, name)),
        0, 1, "cpu-rhf-tda:"+fam.sha256, fam.sha256) for name, _ in fam.partitions)
    return MolecularSelectedMixer(PairPolynomial(.02, -10), groups=groups(),
        core_group_ids=("core",), partitions=parts, switch=switch(), family=fam)


@pytest.mark.parametrize("h", [1e-4, 5e-5])
def test_full_force_contains_switching_term_and_all_active_pairs(controlled_provider, h):
    obj, atoms = mixer(), geometry()
    terms = obj.evaluate_terms(atoms)
    report = obj.last_state_record
    assert len(report["pair_correspondence"]) == 6
    assert set(report["states"]) == {"a", "b", "c"}
    assert np.max(np.abs(terms.switching_forces)) > .01
    for row in range(8):
        for axis in range(3):
            fd = -(obj.evaluate(displaced(atoms, row, axis, h)).energy
                   - obj.evaluate(displaced(atoms, row, axis, -h)).energy)/(2*h)
            assert fd == pytest.approx(terms.total.forces[row, axis], abs=3e-7)
    order = (8, 3, 5, 1, 4, 2, 7, 6)
    other = obj.evaluate_terms(atoms.select(order))
    np.testing.assert_allclose(other.total.forces,
                              terms.total.forces[[atoms.atom_ids.index(a) for a in order]], atol=1e-12)
    report["states"].clear()
    assert len(obj.last_state_record["states"]) == 3


def test_inactive_states_are_not_reused_after_reactivation(controlled_provider):
    obj, atoms = mixer(), geometry()
    obj.evaluate(atoms)
    xyz = atoms.positions.copy()
    xyz[2:4] = [[1.8, 0, 0], [1.8, 0, .74]]
    xyz[4:] *= 3
    result = obj.evaluate_terms(replace(atoms, positions=xyz))
    assert result.active_partition_ids == ("a",)
    assert obj.last_state_record["pair_correspondence"] == []
    assert obj.partitions[1].calculator.high.last_state is None
    obj.evaluate(atoms)
    assert len(obj.last_state_record["pair_correspondence"]) == 6


@pytest.mark.parametrize("reason", ["provider", "wrong_pair", "coverage", "stale"])
def test_failed_evaluation_publishes_no_old_state_and_recovers(controlled_provider, monkeypatch, reason):
    obj, atoms = mixer(), geometry()
    obj.evaluate(atoms)
    high = obj.partitions[1].calculator.high
    with monkeypatch.context() as patch:
        if reason == "provider":
            def fail(*args):
                raise EvaluationError("deliberate response failure")
            patch.setattr(high, "_solve_surface", fail)
        elif reason == "wrong_pair":
            patch.setattr(high, "_solve_surface", lambda *args: solved((T2, T1)))
            # Both fixed references select root 2 correctly; force a wrong transient map.
            original = high.evaluate
            def wrong(atoms):
                result = original(atoms)
                high.last_state = replace(high.last_state, selected_index=0)
                return result
            patch.setattr(high, "evaluate", wrong)
        elif reason == "stale":
            original = high.evaluate
            def stale(atoms):
                result = original(atoms)
                high.last_state = replace(high.last_state, geometry=displaced(atoms, 0, 0, .1))
                return result
            patch.setattr(high, "evaluate", stale)
        else:
            xyz = atoms.positions.copy()
            xyz[2:] *= 5
            atoms = replace(atoms, positions=xyz)
        with pytest.raises(EvaluationError):
            obj.evaluate(atoms)
    assert obj.last_state_record is None
    assert all(p.calculator.high.last_state is None for p in obj.partitions)
    obj.evaluate(geometry())
    assert len(obj.last_state_record["states"]) == 3


def test_provider_requires_own_and_core_to_select_same_state(controlled_provider):
    fam = family()
    parts = tuple((name, replace(ref, transition_density=T2)) for name, ref in fam.partitions)
    obj = mixer(replace(fam, partitions=parts))
    with pytest.raises(EvaluationError, match="different excitations"):
        obj.evaluate(geometry())
    assert obj.last_state_record is None


def test_ground_entrypoint_and_energy_offsets_remain_forbidden(controlled_provider):
    obj = mixer()
    with pytest.raises(ValueError, match="ground molecular"):
        MolecularMechanicalMixer(obj.full_low, groups=groups(), core_group_ids=("core",),
                                 partitions=obj.partitions, switch=switch())
    with pytest.raises(ValueError, match="energy shifts"):
        replace(obj.partitions[0], energy_offset_ev=1., offset_source="unqualified shift")
    with pytest.raises(ValueError, match="Explicit MechanicalPartition"):
        MolecularMechanicalMixer(obj.full_low, groups=groups(), core_group_ids=("core",),
                                 partitions=(None,), switch=switch())
    with pytest.raises(ValueError, match="complete"):
        MolecularSelectedMixer(obj.full_low, groups=groups(), core_group_ids=("core",),
                               partitions=obj.partitions[:-1], switch=switch(), family=obj.family)
