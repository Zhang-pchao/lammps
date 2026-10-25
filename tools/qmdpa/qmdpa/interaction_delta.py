# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite fragment interaction replacement, without a physical DPA decomposition.

Delta W = [E_H(C+B)-E_H(C)-E_H(B)] - [U_L(C+B)-U_L(C)-U_L(B)].

This replaces one explicitly defined low-level cross interaction. Isolated
fragment energies, other interactions and higher-order nonadditivity remain
in the caller's baseline. A shared correction to ground and selected surfaces
does not change their gap. Scientific accuracy requires independent validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .core import Evaluation, Geometry, _atom_ids, _evaluate


@dataclass(frozen=True, eq=False)
class InteractionTriplet:
    """Unsigned pair and isolated fragments, each in its declared atom order."""

    pair: Evaluation
    core: Evaluation
    fragment: Evaluation


@dataclass(frozen=True, eq=False)
class InteractionDeltaTerms:
    """Separate high/low interaction energies and their full-order correction."""

    model: ClassVar[str] = "finite-cross-interaction-delta-v1"
    atom_ids: tuple[int, ...]
    core_atom_ids: tuple[int, ...]
    fragment_atom_ids: tuple[int, ...]
    high: InteractionTriplet
    low: InteractionTriplet
    high_interaction: Evaluation
    low_interaction: Evaluation
    correction: Evaluation


class CrossInteractionDelta:
    """One disjoint finite C/B interaction with explicit fragment backends.

    All six providers are explicit so fragment charge, spin, basis and model
    conditions cannot silently inherit the pair's conditions. The caller must
    pin those identities and use complete molecules or a separately qualified
    boundary prescription. Pair order is C followed by B. Spectators receive
    exactly zero correction. No cutoff, fitted parameter or state dependence
    is introduced here. Backend failures propagate without stale forces.
    """

    def __init__(self, high_pair, high_core, high_fragment,
                 low_pair, low_core, low_fragment, *, core_atom_ids, fragment_atom_ids):
        self._core_ids = _atom_ids(core_atom_ids)
        self._fragment_ids = _atom_ids(fragment_atom_ids)
        if set(self._core_ids).intersection(self._fragment_ids):
            raise ValueError("Core and fragment atom IDs must be disjoint")
        self._pair_ids = self._core_ids + self._fragment_ids
        self._high = (high_pair, high_core, high_fragment)
        self._low = (low_pair, low_core, low_fragment)
        if any(not callable(getattr(v, "evaluate", None)) for v in self._high + self._low):
            raise TypeError("Every pair and fragment requires an explicit energy/force provider")
        self._symbols = None

    @property
    def core_atom_ids(self):
        return self._core_ids

    @property
    def fragment_atom_ids(self):
        return self._fragment_ids

    def _validated_geometries(self, full_geometry):
        if full_geometry.periodic:
            raise ValueError("Interaction replacement requires a nonperiodic geometry")
        pair = full_geometry.select(self._pair_ids)
        if self._symbols is not None and pair.symbols != self._symbols:
            raise ValueError("Element symbols for fixed correction atom IDs must stay unchanged")
        return pair, pair.select(self._core_ids), pair.select(self._fragment_ids)

    def evaluate_terms(self, full_geometry: Geometry) -> InteractionDeltaTerms:
        geometries = self._validated_geometries(full_geometry)
        pair = geometries[0]
        high = InteractionTriplet(*(_evaluate(v, g) for v, g in zip(self._high, geometries)))
        low = InteractionTriplet(*(_evaluate(v, g) for v, g in zip(self._low, geometries)))
        lookup = {atom: i for i, atom in enumerate(full_geometry.atom_ids)}
        pair_rows = [lookup[i] for i in self._pair_ids]
        core_rows = [lookup[i] for i in self._core_ids]
        fragment_rows = [lookup[i] for i in self._fragment_ids]

        def interaction(terms):
            forces = np.zeros_like(full_geometry.positions)
            forces[pair_rows] = terms.pair.forces
            forces[core_rows] -= terms.core.forces
            forces[fragment_rows] -= terms.fragment.forces
            return Evaluation(terms.pair.energy - terms.core.energy - terms.fragment.energy, forces)

        ih, il = interaction(high), interaction(low)
        correction = Evaluation(ih.energy - il.energy, ih.forces - il.forces)
        self._symbols = pair.symbols
        return InteractionDeltaTerms(full_geometry.atom_ids, self._core_ids,
                                     self._fragment_ids, high, low, ih, il, correction)

    def evaluate(self, full_geometry: Geometry) -> Evaluation:
        return self.evaluate_terms(full_geometry).correction


@dataclass(frozen=True, eq=False)
class CorrectedTerms:
    """A baseline and explicit additive correction, both in full input order."""

    atom_ids: tuple[int, ...]
    baseline: Evaluation
    correction: Evaluation
    total: Evaluation


class CorrectedCalculator:
    """Compose a full baseline with a full-order conservative correction.

    This adapter returns a total, never a native pair correction. It supplies no
    electronic state history or native configuration by itself. The caller must
    record the baseline and correction identities separately and qualify the
    combined Hamiltonian before using a driver.
    """

    def __init__(self, baseline, correction):
        if any(not callable(getattr(v, "evaluate", None)) for v in (baseline, correction)):
            raise TypeError("Baseline and correction must supply energy and forces")
        self.baseline = baseline
        self.correction = correction

    def evaluate_terms(self, full_geometry: Geometry) -> CorrectedTerms:
        baseline = _evaluate(self.baseline, full_geometry)
        correction = _evaluate(self.correction, full_geometry)
        total = Evaluation(baseline.energy + correction.energy,
                           baseline.forces + correction.forces)
        return CorrectedTerms(full_geometry.atom_ids, baseline, correction, total)

    def evaluate(self, full_geometry: Geometry) -> Evaluation:
        return self.evaluate_terms(full_geometry).total
