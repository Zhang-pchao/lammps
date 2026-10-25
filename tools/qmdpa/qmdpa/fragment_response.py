# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Standalone fragment interaction and selected-gap response candidates.

No native driver or scientific qualification is implied. Every selected
provider needs its own frozen matching state and complete derivatives.
"""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .additive import _embedded_evaluate
from .core import Evaluation, Geometry, _atom_ids, _evaluate
from .gap_anchor import _same_field
from .interaction_delta import CorrectedCalculator, CrossInteractionDelta


@dataclass(frozen=True, eq=False)
class PairSetTerms:
    model: ClassVar[str] = "finite-fragment-pair-delta-set-v1"
    atom_ids: tuple[int, ...]
    pairs: tuple
    correction: Evaluation


class PairInteractionSet:
    """Sum explicit cross corrections, rejecting double-counted atomic pairs.

    Shared fragments are allowed, for example C-D and B-D. Overlapping atomic
    pair coverage, for example C-D and (C+B)-D, is rejected. This is a finite
    declared pair set, without distance switching or a three-body correction.
    """

    def __init__(self, corrections):
        corrections = tuple(corrections)
        if not corrections or any(type(x) is not CrossInteractionDelta for x in corrections):
            raise TypeError("A nonempty explicit set of CrossInteractionDelta objects is required")
        covered = set()
        for item in corrections:
            pairs = {frozenset((a, b)) for a in item.core_atom_ids for b in item.fragment_atom_ids}
            if pairs.intersection(covered):
                raise ValueError("Interaction corrections double-count an atomic pair")
            covered.update(pairs)
        self.corrections = corrections

    def evaluate_terms(self, geometry: Geometry):
        terms = tuple(item.evaluate_terms(geometry) for item in self.corrections)
        force = np.zeros_like(geometry.positions)
        energy = 0.
        for item in terms:
            force += item.correction.forces
            energy += item.correction.energy
        return PairSetTerms(geometry.atom_ids, terms, Evaluation(energy, force))

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).correction


@dataclass(frozen=True, eq=False)
class FragmentAlignmentTerms:
    """One explicitly selected monomer, reused from a pair evaluation."""

    atom_ids: tuple[int, ...]
    pair_index: int
    side: str
    high: Evaluation
    low: Evaluation
    correction: Evaluation


@dataclass(frozen=True, eq=False)
class AlignedPairSetTerms:
    model: ClassVar[str] = "finite-fragment-pair-delta-one-body-aligned-v1"
    atom_ids: tuple[int, ...]
    pair_set: PairSetTerms
    alignments: tuple[FragmentAlignmentTerms, ...]
    correction: Evaluation


class OneBodyAlignedPairSet(PairInteractionSet):
    """Add explicit E_H(D)-U_L(D) monomer differences to a pair set.

    Each alignment is a (zero-based pair index, "core" or "fragment") tuple.
    It selects the exact high/low monomer providers already used by that pair.
    Selected monomers must not overlap; shared nodes are aligned only once.
    Their raw energy/force objects are reused within this evaluation, with no
    new backend calls or cross-geometry cache. This is a distinct Hamiltonian:
    add it to both surfaces to retain their vertical excitation gap. The caller
    must declare complete molecules and compatible high/low method identities;
    no physical decomposition of a learned potential is assumed.
    """

    def __init__(self, corrections, *, alignments):
        super().__init__(corrections)
        choices = tuple(alignments)
        if not choices:
            raise ValueError("Select at least one explicit monomer for alignment")
        selected = []
        covered = set()
        for choice in choices:
            if not isinstance(choice, (tuple, list)) or len(choice) != 2:
                raise ValueError("Each alignment must select a pair index and monomer side")
            index, side = choice
            if type(index) is not int or not 0 <= index < len(self.corrections):
                raise ValueError("Alignment pair index is outside the declared pair set")
            if side not in ("core", "fragment"):
                raise ValueError("Alignment side must be core or fragment")
            ids = getattr(self.corrections[index], side + "_atom_ids")
            if covered.intersection(ids):
                raise ValueError("Aligned monomers overlap and would double-count a one-body term")
            covered.update(ids)
            selected.append((index, side, ids))
        self._alignments = tuple(selected)

    def evaluate_terms(self, geometry: Geometry):
        pair_set = super().evaluate_terms(geometry)
        energy = pair_set.correction.energy
        force = pair_set.correction.forces.copy()
        rows = {atom: row for row, atom in enumerate(geometry.atom_ids)}
        alignments = []
        for index, side, ids in self._alignments:
            pair = pair_set.pairs[index]
            if getattr(pair, side + "_atom_ids") != ids:
                raise ValueError("Aligned monomer identity changed")
            high, low = getattr(pair.high, side), getattr(pair.low, side)
            aligned_force = np.zeros_like(geometry.positions)
            aligned_force[[rows[atom] for atom in ids]] = high.forces - low.forces
            correction = Evaluation(high.energy - low.energy, aligned_force)
            alignments.append(FragmentAlignmentTerms(ids, index, side, high, low, correction))
            energy += correction.energy
            force += correction.forces
        return AlignedPairSetTerms(geometry.atom_ids, pair_set, tuple(alignments),
                                   Evaluation(energy, force))


@dataclass(frozen=True, eq=False)
class ClusterGapTerms:
    model: ClassVar[str] = "finite-cluster-minus-field-gap-v1"
    atom_ids: tuple[int, ...]
    pair_ground: Evaluation
    pair_selected: Evaluation
    embedded_ground: object
    embedded_selected: object
    pair_gap: Evaluation
    correction: Evaluation


class ClusterGapDelta:
    """Delta Omega = Omega(C+D;0) - Omega(C;v_D), with all derivatives.

    Both selected inputs are total selected-state energies, not excitation
    energies. Separate ground terms are subtracted explicitly. This correction
    belongs only to the selected surface; adding it to ground would change the
    method. Whole-fragment identities, same quantum settings and physical state
    correspondence must be pinned by the caller. The explicit native route records both
    auxiliary selected states; standalone callers retain their own records.
    """

    def __init__(self, pair_ground, pair_selected, embedded_ground,
                 embedded_selected, *, core_atom_ids, fragment_atom_ids):
        self.core_atom_ids = _atom_ids(core_atom_ids)
        self.fragment_atom_ids = _atom_ids(fragment_atom_ids)
        if set(self.core_atom_ids).intersection(self.fragment_atom_ids):
            raise ValueError("Cluster core and fragment must be disjoint")
        self.pair_atom_ids = self.core_atom_ids + self.fragment_atom_ids
        self.pair_ground, self.pair_selected = pair_ground, pair_selected
        self.embedded_ground, self.embedded_selected = embedded_ground, embedded_selected
        if any(not callable(getattr(b, "evaluate", None))
               for b in (pair_ground, pair_selected, embedded_ground, embedded_selected)):
            raise TypeError("Every cluster-gap term needs an explicit complete energy/force provider")
        field = getattr(embedded_ground, "charges", None)
        if (not _same_field(field, getattr(embedded_selected, "charges", None))
                or field.atom_ids != self.fragment_atom_ids):
            raise ValueError("Embedded gap terms must have the same fixed fragment field")
        self._field = field
        self._symbols = None

    def _validated_geometries(self, geometry):
        if geometry.periodic:
            raise ValueError("Cluster-gap correction requires a finite geometry")
        if (not _same_field(self._field, self.embedded_ground.charges)
                or not _same_field(self._field, self.embedded_selected.charges)):
            raise ValueError("The cluster-gap fixed field changed")
        pair = geometry.select(self.pair_atom_ids)
        if self._symbols is not None and pair.symbols != self._symbols:
            raise ValueError("Cluster-gap element identity changed")
        core = pair.select(self.core_atom_ids)
        fragment = pair.select(self.fragment_atom_ids)
        return pair, core, fragment

    def evaluate_terms(self, geometry: Geometry):
        pair, core, fragment = self._validated_geometries(geometry)
        ground = _evaluate(self.pair_ground, pair)
        selected = ground if self.pair_selected is self.pair_ground else _evaluate(self.pair_selected, pair)
        eg = _embedded_evaluate(self.embedded_ground, core, fragment, "Cluster embedded ground")
        es = eg if self.embedded_selected is self.embedded_ground else _embedded_evaluate(
            self.embedded_selected, core, fragment, "Cluster embedded selected")
        pair_gap = Evaluation(selected.energy-ground.energy, selected.forces-ground.forces)
        rows = {atom: i for i, atom in enumerate(geometry.atom_ids)}
        force = np.zeros_like(geometry.positions)
        force[[rows[i] for i in self.pair_atom_ids]] += pair_gap.forces
        force[[rows[i] for i in self.core_atom_ids]] -= (
            es.qm_forces_ev_per_angstrom-eg.qm_forces_ev_per_angstrom)
        force[[rows[i] for i in self.fragment_atom_ids]] -= (
            es.environment_forces_ev_per_angstrom-eg.environment_forces_ev_per_angstrom)
        correction = Evaluation(pair_gap.energy-(es.energy_ev-eg.energy_ev), force)
        self._symbols = pair.symbols
        return ClusterGapTerms(geometry.atom_ids, ground, selected, eg, es, pair_gap, correction)

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).correction


class ReducedFragmentResponseCalculator(CorrectedCalculator):
    """Evaluate the existing total after cancelling identical signed terms.

    Only the concrete anchor, pair-set and cluster-gap composition is accepted.
    Provider object identity and ordered region IDs define equality; equal
    settings or equal numerical results alone never authorize cancellation.
    The three selected providers are always evaluated, even at zero net weight,
    so their state/failure records remain mandatory. No cross-geometry cache is
    added. evaluate_terms retains the expanded diagnostic ledger.
    """

    def __init__(self, anchor, pair_set, cluster_gap):
        from .gap_anchor import GapAnchorCalculator

        if (type(anchor) is not GapAnchorCalculator
                or type(pair_set) not in (PairInteractionSet, OneBodyAlignedPairSet)
                or type(cluster_gap) is not ClusterGapDelta):
            raise TypeError("Reduced evaluation requires the explicit fragment composition")
        super().__init__(CorrectedCalculator(anchor, pair_set), cluster_gap)
        self.last_execution = None

    def evaluate(self, geometry):
        self.last_execution = None
        anchor, pairs, gap = self.baseline.baseline, self.baseline.correction, self.correction
        ordered, qm, environment = anchor._validated_geometries(geometry)
        pair_geometries = [item._validated_geometries(geometry) for item in pairs.corrections]
        cluster, core, fragment = gap._validated_geometries(geometry)
        terms = []

        def add(backend, selected, coefficient, field=None, *, required=False):
            terms.append((backend, selected, field, coefficient, required))

        add(anchor.full_low_backend, ordered, 1)
        add(anchor.region_low_backend, qm, -1)
        add(anchor.isolated_ground_backend, qm, 1)
        add(anchor.embedded_ground_backend, qm, -1, environment)
        add(anchor.embedded_selected_backend, qm, 1, environment, required=True)
        for item, geometries in zip(pairs.corrections, pair_geometries):
            for backend, selected, coefficient in zip(item._high, geometries, (1, -1, -1)):
                add(backend, selected, coefficient)
            for backend, selected, coefficient in zip(item._low, geometries, (-1, 1, 1)):
                add(backend, selected, coefficient)
        if type(pairs) is OneBodyAlignedPairSet:
            for index, side, ids in pairs._alignments:
                item = pairs.corrections[index]
                if getattr(item, side + "_atom_ids") != ids:
                    raise ValueError("Aligned monomer identity changed")
                which = 1 if side == "core" else 2
                add(item._high[which], pair_geometries[index][which], 1)
                add(item._low[which], pair_geometries[index][which], -1)
        add(gap.pair_ground, cluster, -1)
        add(gap.pair_selected, cluster, 1, required=True)
        add(gap.embedded_ground, core, 1, fragment)
        add(gap.embedded_selected, core, -1, fragment, required=True)

        grouped = {}
        for backend, selected, field, coefficient, required in terms:
            key = (id(backend), selected.atom_ids, None if field is None else field.atom_ids)
            if key not in grouped:
                grouped[key] = [backend, selected, field, 0, False]
            grouped[key][3] += coefficient
            grouped[key][4] = grouped[key][4] or required
        rows = {atom: i for i, atom in enumerate(geometry.atom_ids)}
        energy, forces, evaluated, skipped = 0., np.zeros_like(geometry.positions), 0, 0
        for backend, selected, field, coefficient, required in grouped.values():
            if not coefficient and not required:
                skipped += 1
                continue
            if field is None:
                result = _evaluate(backend, selected)
                energy += coefficient * result.energy
                forces[[rows[i] for i in selected.atom_ids]] += coefficient * result.forces
            else:
                result = _embedded_evaluate(backend, selected, field, "Reduced fragment term")
                energy += coefficient * result.energy_ev
                forces[[rows[i] for i in selected.atom_ids]] += (
                    coefficient * result.qm_forces_ev_per_angstrom)
                forces[[rows[i] for i in field.atom_ids]] += (
                    coefficient * result.environment_forces_ev_per_angstrom)
            evaluated += 1
        total = Evaluation(energy, forces)
        for item, geometries in zip(pairs.corrections, pair_geometries):
            item._symbols = geometries[0].symbols
        gap._symbols = cluster.symbols
        self.last_execution = {
            "expanded_terms": len(terms), "unique_terms": len(grouped),
            "evaluated_terms": evaluated, "cancelled_terms": skipped,
            "required_selected_terms": sum(row[4] for row in grouped.values()),
        }
        return total
