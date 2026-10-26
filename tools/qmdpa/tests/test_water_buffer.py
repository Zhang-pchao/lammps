#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Independent conservative-force and protected-boundary tests."""
from dataclasses import replace

import numpy as np
import pytest
from test_adaptive import PairPolynomial, displaced

from qmdpa.adaptive import MechanicalPartition
from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.selection_config import FixedGroup
from qmdpa.water_buffer import (
    QuinticWaterBufferWeights,
    WaterBoundaryGuard,
    WaterBufferMechanicalMixer,
)


def groups():
    return tuple(FixedGroup(name, tuple(range(3*i+1,3*i+4)), ("O","H","H"))
                 for i,name in enumerate(("core-a","core-b","a","b")))


def geometry():
    return Geometry(tuple(range(1,13)), ("O","H","H")*4, [
        [0,0,0],[.95,0,0],[-.24,.92,0],
        [2.7,.1,0],[3.65,.1,0],[2.46,1.02,0],
        [0,3.5,0],[.95,3.5,0],[-.24,4.42,0],
        [-3.8,0,0],[-2.85,0,0],[-4.04,.92,0]])


def switch():
    return QuinticWaterBufferWeights(1, (("b",10),("a",7)), 3.,5.)


class OxygenGauge(PairPolynomial):
    def __init__(self, shift=0.):
        super().__init__(.035,-15.)
        self.shift=shift

    def evaluate(self, atoms):
        value=super().evaluate(atoms)
        return Evaluation(value.energy+self.shift*atoms.symbols.count("O"),value.forces)


def mixer(*, shift=0., group_table=None, partition_change=None):
    low=PairPolynomial(.02,-10.)
    policy=switch()
    partitions=[]
    for i,subset in enumerate(policy.group_subsets):
        p=MechanicalPartition(str(i),("core-a","core-b")+subset,
                              SubtractiveCalculator(low,OxygenGauge(shift)),0,1,
                              "analytic-water-v1","isolated-water-reference",
                              -len(subset)*(1.25+shift),"reference-fixture")
        if partition_change is not None and i==1:
            p=partition_change(p)
        partitions.append(p)
    return WaterBufferMechanicalMixer(low,groups=groups() if group_table is None else group_table,
        core_group_ids=("core-a","core-b"),partitions=partitions,switch=policy,
        guard=WaterBoundaryGuard(.6,1.25,1.45),
        reference_delta_ev=1.25+shift,reference_source="reference-fixture")


def test_weights_partition_unity_expected_occupancy_and_all_derivatives():
    atoms,policy=geometry(),switch()
    value=policy.evaluate(atoms)
    np.testing.assert_allclose(value.weights.sum(),1,atol=1e-15,rtol=0)
    np.testing.assert_allclose(value.gradients_per_angstrom.sum(axis=0),0,atol=1e-14,rtol=0)
    np.testing.assert_allclose(value.gradients_per_angstrom.sum(axis=1),0,atol=1e-14,rtol=0)
    expected=sum(len(s)*w for s,w in zip(value.group_subsets,value.weights,strict=True))
    assert expected==pytest.approx(value.candidate_weights.sum(),abs=1e-14)
    for atom in range(12):
        for axis in range(3):
            fd=(policy.evaluate(displaced(atoms,atom,axis,1e-6)).weights
                -policy.evaluate(displaced(atoms,atom,axis,-1e-6)).weights)/2e-6
            np.testing.assert_allclose(fd,value.gradients_per_angstrom[:,atom,axis],atol=3e-9,rtol=0)


@pytest.mark.parametrize("h",[1e-4,5e-5])
def test_complete_force_derivatives_include_variable_count_reference(h):
    atoms,obj=geometry(),mixer()
    result=obj.evaluate_terms(atoms)
    assert len(result.active_partition_ids)==4
    assert np.max(np.abs(result.switching_forces))>.01
    for atom in range(12):
        for axis in range(3):
            e=[obj.evaluate(displaced(atoms,atom,axis,sign*h)).energy for sign in (-2,-1,1,2)]
            f=-(e[0]-8*e[1]+8*e[2]-e[3])/(12*h)
            assert f==pytest.approx(result.total.forces[atom,axis],abs=2e-7)
    np.testing.assert_allclose(result.total.forces.sum(axis=0),0,atol=1e-12,rtol=0)


@pytest.mark.parametrize("occupancy",[0,1,2])
def test_pure_empty_partial_and_full_occupancy(occupancy):
    atoms,obj=geometry(),mixer()
    xyz=atoms.positions.copy()
    for i,(lo,hi) in enumerate(((6,9),(9,12))):
        radius=2.9 if i<occupancy else 5.1
        delta=np.array([0,radius,0]) if i==0 else np.array([-radius,0,0])
        xyz[lo:hi]+=delta-xyz[lo]
    result=obj.evaluate_terms(replace(atoms,positions=xyz))
    assert len(result.active_partition_ids)==1
    active=np.argmax(result.weights.weights)
    assert len(result.weights.group_subsets[active])==occupancy
    np.testing.assert_array_equal(result.switching_forces,np.zeros_like(xyz))
    assert sum(x is not None for x in result.corrections)==1
    if occupancy==2:
        high=OxygenGauge().evaluate(replace(atoms,positions=xyz))
        assert result.total.energy==pytest.approx(high.energy-2*1.25,abs=1e-12)
        np.testing.assert_allclose(result.total.forces,high.forces,atol=1e-12,rtol=0)


def test_per_water_energy_gauge_changes_only_the_protected_core_constant():
    atoms=geometry()
    a,b=mixer().evaluate_terms(atoms),mixer(shift=1000.).evaluate_terms(atoms)
    assert b.total.energy-a.total.energy==pytest.approx(2000.,abs=1e-10)
    np.testing.assert_allclose(a.total.forces,b.total.forces,atol=1e-10,rtol=0)


def test_protected_internal_transfer_order_and_translation_are_allowed():
    atoms,obj=geometry(),mixer()
    xyz=atoms.positions.copy()
    xyz[1]=[1.5,.02,0]
    atoms=replace(atoms,positions=xyz)
    value=obj.evaluate(atoms)
    order=tuple(reversed(atoms.atom_ids))
    shuffled=obj.evaluate(atoms.select(order))
    np.testing.assert_allclose(value.forces[::-1],shuffled.forces,atol=1e-12,rtol=0)
    moved=obj.evaluate(replace(atoms,positions=atoms.positions+[11,-4,1]))
    np.testing.assert_allclose(value.forces,moved.forces,atol=1e-12,rtol=0)


@pytest.mark.parametrize("failure",["bond","foreign","core_escape"])
def test_geometric_failures_precede_every_backend_evaluation(failure):
    atoms,obj=geometry(),mixer()
    xyz=atoms.positions.copy()
    if failure=="bond":
        xyz[7,0]+=1.
    elif failure=="foreign":
        xyz[6:9]=[[0,2.25,0],[0,1.3,0],[.92,2.5,0]]
    else:
        xyz[1]=xyz[6]+[.9,0,0]
    with pytest.raises(EvaluationError):
        obj.evaluate(replace(atoms,positions=xyz))
    assert obj.full_low.calls==0


def test_undeclared_outer_water_cannot_enter_support():
    extra=FixedGroup("outer",(13,14,15),("O","H","H"))
    obj=mixer(group_table=groups()+(extra,))
    atoms=geometry()
    xyz=np.vstack([atoms.positions,[[0,-5.5,0],[.95,-5.5,0],[-.24,-4.58,0]]])
    all_atoms=Geometry(tuple(range(1,16)),("O","H","H")*5,xyz)
    obj.evaluate(all_atoms)
    before=obj.full_low.calls
    xyz[12:]+=np.array([0,1.6,0])
    with pytest.raises(EvaluationError,match="undeclared"):
        obj.evaluate(replace(all_atoms,positions=xyz))
    assert obj.full_low.calls==before


def test_partition_offset_cannot_be_mistaken_for_a_common_energy_zero():
    with pytest.raises(ValueError,match="offset"):
        mixer(partition_change=lambda p:replace(p,energy_offset_ev=p.energy_offset_ev+.01))


@pytest.mark.parametrize("n",[1,2,3,6])
def test_bounded_family_includes_every_subset(n):
    policy=QuinticWaterBufferWeights(1,tuple((str(i),i+2) for i in range(n)),3.,5.)
    assert len(policy.group_subsets)==2**n
    assert policy.group_subsets[0]==()
    assert len(policy.group_subsets[-1])==n


def test_exponential_family_is_bounded_before_enumeration():
    with pytest.raises(ValueError,match="six"):
        QuinticWaterBufferWeights(1,tuple((str(i),i+2) for i in range(7)),3.,5.)
