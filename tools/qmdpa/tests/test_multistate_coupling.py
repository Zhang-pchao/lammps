# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Complete state-set assembly, exact sharing and failure boundaries."""

from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np
import pytest
from test_excited import charges, environment, qm
from test_gap_anchor import Embedded
from test_multistate import setup

from qmdpa import multistate_coupling as hybrid
from qmdpa.additive import AdditiveCalculator, CrossLennardJones
from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.excited import PySCFTDABackend, PySCFTDAElectrostaticBackend
from qmdpa.gap_anchor import GapAnchorCalculator
from qmdpa.multistate import PySCFTDAStateSet


class Low:
    def __init__(self, scale=0.13):
        self.scale = scale
        self.calls = []

    def evaluate(self, geometry):
        self.calls.append(geometry)
        return Evaluation(self.scale * np.sum(geometry.positions ** 2) / 2,
                          -self.scale * geometry.positions)


def full():
    q, e = qm(), environment()
    return Geometry(q.atom_ids + e.atom_ids, q.symbols + e.symbols,
                    np.concatenate((q.positions, e.positions)))


def cross():
    return CrossLennardJones(
        {(a, b): (0.001, 1.1) for a in qm().symbols for b in environment().symbols},
        source="Synthetic assembly control",
    )


def fixture(monkeypatch, mode):
    refs, solves, _ = setup(monkeypatch, embedded=mode != "me")
    states = PySCFTDAStateSet(refs)
    low, region, ground = Low(), Low(0.17), Low(0.23)
    field = Embedded(0.21, 0.07, charges())
    if mode == "me":
        calculator = hybrid.MechanicalStateSet(region, states, full_low_backend=low)
        extra = None
    elif mode == "additive":
        extra = cross()
        calculator = hybrid.AdditiveStateSet(low, states, extra)
    else:
        def factory(full_low, region_low, reference):
            return GapAnchorCalculator(full_low, region_low, ground, field, field,
                                       qm_atom_ids=qm().atom_ids,
                                       environment_atom_ids=environment().atom_ids)
        monkeypatch.setattr(hybrid, "make_rhf_gap_anchor", factory)
        calculator = hybrid.GapAnchorStateSet(low, region, states)
        extra = field
    return calculator, refs, solves, low, region, ground, extra


@pytest.mark.parametrize("mode", ["me", "additive", "anchor"])
def test_complete_totals_match_existing_single_state_assembly_and_share_work(monkeypatch, mode):
    calc, refs, solves, low, region, ground, extra = fixture(monkeypatch, mode)
    # Deliberately interleave the two region orders.
    atoms = full().select((11, 2, 10, 1))
    result = calc.evaluate(atoms)
    assert len(solves) == 1 and len(low.calls) == 1
    assert len(region.calls) == (0 if mode == "additive" else 1)
    assert len(ground.calls) == (1 if mode == "anchor" else 0)
    if mode == "anchor":
        assert len(extra.calls) == 1
    for name, ref in refs.items():
        if mode == "me":
            expected = SubtractiveCalculator(region, PySCFTDABackend(ref.settings, ref),
                                             full_low=low).evaluate(atoms, qm())
        elif mode == "additive":
            selected = PySCFTDAElectrostaticBackend(ref.settings, ref, ref.charges)
            expected = AdditiveCalculator(low, selected, extra, qm_atom_ids=qm().atom_ids,
                                          environment_atom_ids=environment().atom_ids).evaluate(atoms)
        else:
            selected = PySCFTDAElectrostaticBackend(ref.settings, ref, ref.charges)
            expected = GapAnchorCalculator(low, region, ground, extra, selected,
                                           qm_atom_ids=qm().atom_ids,
                                           environment_atom_ids=environment().atom_ids).evaluate(atoms)
        value = result[name]
        assert value.atom_ids == atoms.atom_ids and value.qm_atom_ids == qm().atom_ids
        assert value.environment_atom_ids == (11, 10)
        assert value.hamiltonian == calc.hamiltonian
        assert value.quantum.diagnostics["reference_sha256"] == ref.sha256
        assert value.total.energy == pytest.approx(expected.energy, abs=1e-13)
        np.testing.assert_allclose(value.total.forces, expected.forces, atol=1e-13, rtol=0)
        with pytest.raises(ValueError):
            value.total.forces.flags.writeable = True
        with pytest.raises(FrozenInstanceError):
            value.total = expected
    with pytest.raises(TypeError):
        result["third"] = result["first"]
    n = len(solves)
    before = len(low.calls)
    again = calc.evaluate(full())
    assert len(solves) == n + 1 and len(low.calls) == before + 1
    rows = [atoms.atom_ids.index(atom) for atom in full().atom_ids]
    for name in result:
        np.testing.assert_allclose(again[name].total.forces, result[name].total.forces[rows])


def test_same_backend_only_coalesces_identical_ordered_geometry_and_never_distinct_objects(monkeypatch):
    refs, _, _ = setup(monkeypatch)
    low = Low()
    calc = hybrid.MechanicalStateSet(low, PySCFTDAStateSet(refs))
    calc.evaluate(qm())
    assert len(low.calls) == 1
    calc.evaluate(qm().select((2, 1)))
    assert len(low.calls) == 3  # Both inputs represent the molecule, but order differs.
    other = Low()
    hybrid.MechanicalStateSet(low, PySCFTDAStateSet(refs), full_low_backend=other).evaluate(qm())
    assert len(low.calls) == 4 and len(other.calls) == 1


@pytest.mark.parametrize("mode", ["me", "additive", "anchor"])
@pytest.mark.parametrize("change", ["periodic", "symbols", "missing"])
def test_bad_geometry_is_rejected_before_quantum_or_low_work(monkeypatch, mode, change):
    calc, _, solves, low, _, _, _ = fixture(monkeypatch, mode)
    atoms = full()
    if change == "periodic":
        atoms = replace(atoms, cell=np.eye(3) * 10)
    elif change == "symbols":
        atoms = replace(atoms, symbols=("He",) + atoms.symbols[1:])
    else:
        atoms = atoms.select(atoms.atom_ids[1:])
    with pytest.raises(ValueError):
        calc.evaluate(atoms)
    assert not solves and not low.calls


@pytest.mark.parametrize("mode", ["me", "additive", "anchor"])
def test_full_identity_is_fixed_even_when_me_environment_is_not_a_quantum_field(monkeypatch, mode):
    calc, _, solves, low, _, _, _ = fixture(monkeypatch, mode)
    calc.evaluate(full())
    before = (len(solves), len(low.calls))
    with pytest.raises(ValueError, match="stay unchanged"):
        calc.evaluate(replace(full(), symbols=full().symbols[:-1] + ("He",)))
    assert (len(solves), len(low.calls)) == before


@pytest.mark.parametrize("mode", ["me", "additive", "anchor"])
def test_wrong_embedding_rejected_at_construction(monkeypatch, mode):
    refs, solves, _ = setup(monkeypatch, embedded=mode == "me")
    states = PySCFTDAStateSet(refs)
    with pytest.raises(ValueError, match="embedding model"):
        if mode == "me":
            hybrid.MechanicalStateSet(Low(), states)
        elif mode == "additive":
            hybrid.AdditiveStateSet(Low(), states, cross())
        else:
            hybrid.GapAnchorStateSet(Low(), Low(), states)
    assert not solves


@pytest.mark.parametrize("corruption", ["missing", "no_forces", "identity", "atom_ids"])
def test_incomplete_quantum_sets_are_rejected_before_low_level_calls(monkeypatch, corruption):
    calc, _, solves, low, _, _, _ = fixture(monkeypatch, "additive")
    surfaces = dict(calc.state_set.evaluate_surfaces(qm(), environment()))
    value = surfaces["second"]
    if corruption == "missing":
        del surfaces["second"]
    elif corruption == "no_forces":
        surfaces["second"] = replace(value, qm_forces_ev_per_angstrom=None,
            environment_forces_ev_per_angstrom=None,
            diagnostics=dict(value.diagnostics, forces_evaluated=False))
    elif corruption == "identity":
        surfaces["second"] = replace(value, diagnostics=dict(
            value.diagnostics, reference_sha256="0" * 64))
    else:
        surfaces["second"] = replace(value, qm_atom_ids=(100, 101))
    with patch.object(calc.state_set, "evaluate_surfaces", return_value=surfaces):
        with pytest.raises(EvaluationError):
            calc.evaluate(full())
    assert len(solves) == 1 and not low.calls


@pytest.mark.parametrize("mode", ["me", "additive", "anchor"])
def test_late_failure_returns_no_partial_collection_and_next_call_is_cold(monkeypatch, mode):
    calc, _, solves, low, _, _, _ = fixture(monkeypatch, mode)
    assemble = calc._assemble
    count = []
    def fail_second(*args):
        count.append(None)
        if len(count) == 2:
            raise EvaluationError("Deliberate late assembly failure")
        return assemble(*args)
    with patch.object(calc, "_assemble", side_effect=fail_second):
        with pytest.raises(EvaluationError, match="late assembly"):
            calc.evaluate(full())
    assert len(solves) == 1 and len(low.calls) == 1
    assert len(calc.evaluate(full())) == 2
    assert len(solves) == 2 and len(low.calls) == 2


@pytest.mark.parametrize("method", ["pbe0", "gpu"])
@pytest.mark.parametrize("mode", ["me", "additive", "anchor"])
def test_method_identity_and_anchor_factory_are_not_substituted(monkeypatch, method, mode):
    from qmdpa.multistate import GPU4PySCFTDAStateSet, PySCFPBE0TDAStateSet
    embedded = mode != "me"
    if method == "pbe0":
        from test_pbe0_multistate import references
        from test_pbe0_tda import install_solver
        solves = install_solver(monkeypatch)
        refs = references(embedded)
        states = PySCFPBE0TDAStateSet(refs)
        factory_name = "make_pbe0_gap_anchor"
    else:
        from test_gpu_multistate import setup as gpu_setup
        refs, solves, _ = gpu_setup(monkeypatch, embedded)
        states = GPU4PySCFTDAStateSet(refs)
        factory_name = "make_gpu_rhf_gap_anchor"
    low = Low()
    if mode == "me":
        calc = hybrid.MechanicalStateSet(low, states)
    elif mode == "additive":
        calc = hybrid.AdditiveStateSet(low, states, cross())
    else:
        def factory(full_low, region_low, reference):
            assert reference is refs["first"]
            ground = Embedded(0.2, 0.1, reference.charges)
            return GapAnchorCalculator(full_low, region_low, Low(), ground, ground,
                                       qm_atom_ids=qm().atom_ids,
                                       environment_atom_ids=environment().atom_ids)
        with patch.object(hybrid, factory_name, side_effect=factory) as selected, \
                patch.object(hybrid, "make_rhf_gap_anchor", side_effect=AssertionError("CPU RHF")):
            calc = hybrid.GapAnchorStateSet(low, low, states)
            selected.assert_called_once()
    result = calc.evaluate(full())
    assert len(solves) == 1
    for name, value in result.items():
        assert value.quantum.diagnostics["reference_sha256"] == refs[name].sha256
        assert ("pbe0" if method == "pbe0" else "gpu-rhf") in value.quantum.diagnostics["method"]
