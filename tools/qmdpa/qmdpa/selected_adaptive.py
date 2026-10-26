# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Conservative selected-surface mixing with an explicit common excitation."""

import copy
from dataclasses import dataclass

import numpy as np

from .adaptive import MechanicalPartition
from .core import EvaluationError
from .molecular_adaptive import MolecularMechanicalMixer
from .state_family import FamilyTDABackend, StateFamilyReference, pairwise_correspondence

HAMILTONIAN = "finite-molecular-selected-mechanical"
STATE_KIND = "common-core-rhf-tda"


@dataclass(frozen=True, eq=False)
class SelectedMechanicalPartition(MechanicalPartition):
    """Same subtraction and charge contract, with an explicitly selected provider."""

    electronic_state: str = STATE_KIND

    def _validate_electronic_state(self):
        if self.electronic_state != STATE_KIND:
            raise ValueError("The selected molecular partition requires its common-core TDA state")
        if not isinstance(self.calculator.high, FamilyTDABackend):
            raise TypeError("Selected partitions require a prepared family TDA provider")
        if self.energy_offset_ev != 0 or self.offset_source is not None:
            raise ValueError("Selected molecular partitions do not admit energy shifts")

    def validate_provider_state(self):
        super().validate_provider_state()
        self._validate_electronic_state()
        high = self.calculator.high
        if (self.qm_charge != 0 or self.qm_multiplicity != 1
                or high.partition_id != self.partition_id
                or self.method_identity != "cpu-rhf-tda:" + high.family.sha256
                or self.energy_reference != high.family.sha256):
            raise ValueError("Selected partition identity differs from its prepared family")


class MolecularSelectedMixer(MolecularMechanicalMixer):
    """Reuse all complete-family weights/derivatives; validate every active state pair."""

    hamiltonian = HAMILTONIAN

    def __init__(self, full_low, *, groups, core_group_ids, partitions, switch, family):
        if not isinstance(family, StateFamilyReference):
            raise TypeError("A prepared state family is required")
        partitions = tuple(partitions)
        if any(type(p) is not SelectedMechanicalPartition for p in partitions):
            raise TypeError("Explicit selected-state partitions are required")
        if any(p.calculator.high.family is not family for p in partitions):
            raise ValueError("All providers must share the same immutable family")
        super().__init__(full_low, groups=groups, core_group_ids=core_group_ids,
                         partitions=partitions, switch=switch)
        if {p.partition_id for p in self.partitions} != {name for name, _ in family.partitions}:
            raise ValueError("Prepared state references must cover the complete partition family")
        for part, atoms in zip(self.partitions, self.qm_atom_ids):
            if family.reference(part.partition_id).qm_geometry.atom_ids != atoms:
                raise ValueError("Prepared and declared partition atom IDs differ")
        by_group = {g.group_id: g for g in self.groups}
        core = tuple(sorted(atom for name in self.core_group_ids for atom in by_group[name].atom_ids))
        if family.core.qm_geometry.atom_ids != core:
            raise ValueError("The common state reference must cover exactly the fixed core")
        self.family = family
        self._last_record = None

    @property
    def last_state_record(self):
        return copy.deepcopy(self._last_record)

    def evaluate_terms(self, geometry):
        self._last_record = None
        for partition in self.partitions:
            partition.calculator.high.last_state = None
        try:
            terms = super().evaluate_terms(geometry)
            active = []
            for partition in self.partitions:
                if partition.partition_id not in terms.active_partition_ids:
                    continue
                state = partition.calculator.high.last_state
                expected = geometry.select(self.family.reference(partition.partition_id).qm_geometry.atom_ids)
                if (state is None or state.partition_id != partition.partition_id
                        or state.geometry.atom_ids != expected.atom_ids
                        or state.geometry.symbols != expected.symbols
                        or not np.array_equal(state.geometry.positions, expected.positions)):
                    raise EvaluationError("Active partition has no current successful state identity")
                active.append(state)
            runtime = self.partitions[0].calculator.high._runtime
            pairs = pairwise_correspondence(active, self.family.core.settings, runtime)
            self._last_record = {
                "schema_version": 1, "hamiltonian": HAMILTONIAN,
                "family_sha256": self.family.sha256,
                "common_core_reference_sha256": self.family.core.sha256,
                "atom_ids": list(geometry.atom_ids), "positions_angstrom": geometry.positions.tolist(),
                "partition_ids": list(terms.partition_ids), "qm_atom_ids": [list(x) for x in terms.qm_atom_ids],
                "weights": terms.weights.weights.tolist(),
                "active_partition_ids": list(terms.active_partition_ids),
                "states": {s.partition_id: s.diagnostics for s in active}, "pair_correspondence": list(pairs),
                "potential_ev": terms.total.energy, "forces_ev_angstrom": terms.total.forces.tolist(),
                "switching_forces_ev_angstrom": terms.switching_forces.tolist(),
            }
            return terms
        except BaseException:
            self._last_record = None
            for partition in self.partitions:
                partition.calculator.high.last_state = None
            raise
