#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit proton ownership topologies and conservative finite-water weights.

Every topology is an integer atom/charge specification, fixed before dynamics.
No nearest-oxygen relabeling enters the Hamiltonian. Mixed transfer endpoints
and all formal ionic defects remain in every active QM fragment.
"""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .adaptive import _integer, _real
from .core import EvaluationError, Geometry, _readonly_array
from .molecular_adaptive import _radial_preferences
from .selection_config import FixedGroup, FixedGroupSelector, _group_id

HAMILTONIAN = "finite-reactive-water-mechanical"


def _quintic(value):
    if value <= 0:
        return 0., 0.
    if value >= 1:
        return 1., 0.
    u = min(value, 1-value)
    tail = u**3 * (10 + u*(-15 + 6*u))
    return (tail if value <= .5 else 1-tail), 30*value**2*(1-value)**2


def _product_with_gradient(factors, derivatives, shape):
    value = float(np.prod(factors))
    gradient = np.zeros(shape)
    for i, derivative in enumerate(derivatives):
        gradient += np.prod([x for j, x in enumerate(factors) if i != j]) * derivative
    return value, gradient


@dataclass(frozen=True)
class ProtonChannel:
    hydrogen_id: int
    donor_group_id: str
    acceptor_group_id: str

    def __post_init__(self):
        object.__setattr__(self, "hydrogen_id", _integer(self.hydrogen_id, "Transfer H ID", 1))
        for name in ("donor_group_id", "acceptor_group_id"):
            object.__setattr__(self, name, _group_id(getattr(self, name)))
        if self.donor_group_id == self.acceptor_group_id:
            raise ValueError("A proton channel requires distinct donor and acceptor groups")


@dataclass(frozen=True)
class ReactivePartition:
    topology: tuple[int, ...]
    qm_group_ids: tuple[str, ...]
    qm_atom_ids: tuple[int, ...]
    group_hydrogens: tuple[tuple[str, tuple[int, ...]], ...]
    group_charges: tuple[tuple[str, int], ...]
    electron_count: int

    @property
    def partition_id(self):
        return "t" + "".join(map(str, self.topology)) + ":" + ",".join(self.qm_group_ids)


@dataclass(frozen=True, eq=False)
class ReactiveWeights:
    weights: np.ndarray
    gradients_per_angstrom: np.ndarray
    ownership_probabilities: np.ndarray
    channel_gaps_angstrom: np.ndarray

    def __post_init__(self):
        for name in ("weights", "gradients_per_angstrom", "ownership_probabilities", "channel_gaps_angstrom"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), name))
        w, g = self.weights, self.gradients_per_angstrom
        if (w.ndim != 1 or not len(w) or g.ndim != 3 or g.shape[0] != len(w)
                or g.shape[2] != 3 or g.shape[1] == 0
                or np.any(w < 0) or np.any(w > 1)
                or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)
                or not np.allclose(g.sum(axis=0), 0., atol=1e-10, rtol=0)):
            raise EvaluationError("Reactive water weights and derivatives must be normalized")


class ReactiveWaterFamily:
    """Complete bounded topology/subset family with neutral QM fragments."""

    def __init__(self, *, groups, pool_group_ids, seed_group_id, channels,
                 inner_radius_angstrom, outer_radius_angstrom,
                 transfer_inner_angstrom, transfer_outer_angstrom,
                 max_partitions=64):
        self.seed_group_id = _group_id(seed_group_id)
        self.coverage = FixedGroupSelector(tuple(groups), (self.seed_group_id,))
        self.groups = self.coverage.groups
        if any(not isinstance(g, FixedGroup) or sorted(g.symbols) != ["H", "H", "O"] for g in self.groups):
            raise ValueError("Declare the initial neutral H2O groups explicitly")
        self.by_group = {g.group_id: g for g in self.groups}
        self.pool = tuple(sorted(_group_id(g) for g in pool_group_ids))
        if (not 2 <= len(self.pool) <= 4 or len(set(self.pool)) != len(self.pool)
                or not set(self.pool) <= set(self.by_group) or self.seed_group_id not in self.pool):
            raise ValueError("Declare two to four distinct pool waters including the seed")
        self.oxygen = {g.group_id: next(a for a, s in zip(g.atom_ids, g.symbols, strict=True) if s == "O")
                       for g in self.groups}
        self.initial_hydrogens = {g.group_id: tuple(a for a, s in zip(g.atom_ids, g.symbols, strict=True) if s == "H")
                                  for g in self.groups}
        self.channels = tuple(channels)
        if (not 1 <= len(self.channels) <= 2
                or any(not isinstance(c, ProtonChannel) for c in self.channels)
                or len({c.hydrogen_id for c in self.channels}) != len(self.channels)):
            raise ValueError("Declare one or two channels with distinct transferable hydrogens")
        self.channels = tuple(sorted(self.channels, key=lambda c: c.hydrogen_id))
        for channel in self.channels:
            if (channel.donor_group_id not in self.pool or channel.acceptor_group_id not in self.pool
                    or channel.hydrogen_id not in self.initial_hydrogens[channel.donor_group_id]):
                raise ValueError("A transfer H must initially belong to its declared pool donor")
        for inner, low, outer, high in (
                ("inner_radius_angstrom", inner_radius_angstrom, "outer_radius_angstrom", outer_radius_angstrom),
                ("transfer_inner_angstrom", transfer_inner_angstrom, "transfer_outer_angstrom", transfer_outer_angstrom)):
            low, high = _real(low, inner), _real(high, outer)
            if not 0 < low < high:
                raise ValueError("Switch intervals require 0 < inner < outer")
            setattr(self, inner, low)
            setattr(self, outer, high)
        self.max_partitions = _integer(max_partitions, "Maximum partition count", 1)
        if self.max_partitions > 64:
            raise ValueError("Reactive water families are bounded to 64 partitions")
        self.topologies = tuple(product((0, 1), repeat=len(self.channels)))
        partitions = []
        self.mandatory = {}
        for topology in self.topologies:
            hydrogens = {g: list(ids) for g, ids in self.initial_hydrogens.items()}
            for moved, channel in zip(topology, self.channels, strict=True):
                if moved:
                    hydrogens[channel.donor_group_id].remove(channel.hydrogen_id)
                    hydrogens[channel.acceptor_group_id].append(channel.hydrogen_id)
            if any(not 1 <= len(ids) <= 3 for ids in hydrogens.values()):
                raise ValueError("Each complete ownership topology must contain OH, H2O or H3O groups")
            charges = {g: len(ids)-2 for g, ids in hydrogens.items()}
            required = {self.seed_group_id} | {g for g, charge in charges.items() if charge}
            self.mandatory[topology] = tuple(sorted(required))
            optional = tuple(g for g in self.pool if g not in required)
            for n in range(len(optional)+1):
                for subset in combinations(optional, n):
                    selected = tuple(sorted(required | set(subset)))
                    atoms = tuple(sorted(a for g in selected for a in [self.oxygen[g], *hydrogens[g]]))
                    electrons = 8*len(selected) + sum(len(hydrogens[g]) for g in selected)
                    if sum(charges[g] for g in selected) != 0 or electrons != 10*len(selected):
                        raise ValueError("Every QM fragment must include all ionic defects and remain neutral")
                    partitions.append(ReactivePartition(
                        topology, selected, atoms,
                        tuple((g, tuple(sorted(ids))) for g, ids in sorted(hydrogens.items())),
                        tuple(sorted(charges.items())), electrons))
        if len(partitions) > self.max_partitions:
            raise ValueError("The complete reactive family exceeds its declared partition bound")
        self.partitions = tuple(partitions)
        self.atom_ids = tuple(sorted(a for g in self.groups for a in g.atom_ids))

    def evaluate(self, geometry):
        if not isinstance(geometry, Geometry) or geometry.periodic:
            raise ValueError("Reactive water requires a finite geometry")
        self.coverage.select(geometry)
        lookup = {a: i for i, a in enumerate(geometry.atom_ids)}
        shape = geometry.positions.shape
        anchors = tuple((g, self.oxygen[g]) for g in self.pool if g != self.seed_group_id)
        radial, dradial, _radii = _radial_preferences(
            geometry, self.oxygen[self.seed_group_id], anchors,
            self.inner_radius_angstrom, self.outer_radius_angstrom)
        preferences = {self.seed_group_id: (1., np.zeros(shape))}
        preferences.update({g: (radial[i], dradial[i]) for i, (g, _) in enumerate(anchors)})
        probabilities, dprobabilities, envelopes, denvelopes, gaps = [], [], [], [], []
        for channel in self.channels:
            gap, derivative = 0., np.zeros(shape)
            h = lookup[channel.hydrogen_id]
            for group, sign in ((channel.donor_group_id, 1), (channel.acceptor_group_id, -1)):
                o = lookup[self.oxygen[group]]
                delta = geometry.positions[o]-geometry.positions[h]
                radius = np.linalg.norm(delta)
                if radius <= 1e-10:
                    raise EvaluationError("A reactive O-H distance is zero")
                gap += sign*radius
                derivative[o] += sign*delta/radius
                derivative[h] -= sign*delta/radius
            inner, outer = self.transfer_inner_angstrom, self.transfer_outer_angstrom
            alpha, dalpha = _quintic((gap+inner)/(2*inner))
            envelope_loss, dloss = _quintic((abs(gap)-inner)/(outer-inner))
            probabilities.append(alpha)
            dprobabilities.append(dalpha/(2*inner)*derivative)
            envelopes.append(1-envelope_loss)
            denvelopes.append(-dloss/(outer-inner)*np.sign(gap)*derivative)
            gaps.append(gap)
        for group in self.pool:
            base, dbase = preferences[group]
            relevant = [i for i, c in enumerate(self.channels)
                        if group in (c.donor_group_id, c.acceptor_group_id)]
            loss, dloss = _product_with_gradient(
                [1-base, *(1-envelopes[i] for i in relevant)],
                [-dbase, *(-denvelopes[i] for i in relevant)], shape)
            preferences[group] = (1-loss, -dloss)
        weights, gradients = [], []
        for partition in self.partitions:
            factors, derivatives = [], []
            for moved, alpha, dalpha in zip(partition.topology, probabilities, dprobabilities, strict=True):
                factors.append(alpha if moved else 1-alpha)
                derivatives.append(dalpha if moved else -dalpha)
            for group in self.pool:
                if group in self.mandatory[partition.topology]:
                    continue
                p, dp = preferences[group]
                inside = group in partition.qm_group_ids
                factors.append(p if inside else 1-p)
                derivatives.append(dp if inside else -dp)
            value, gradient = _product_with_gradient(factors, derivatives, shape)
            weights.append(value)
            gradients.append(gradient)
        return ReactiveWeights(weights, gradients, probabilities, gaps)
