# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""A common localized TDA character across explicit finite QM partitions.

Transition-density overlaps compare operators across AO bases. They are not
signed wavefunction overlaps, derivative couplings or electronic populations.
"""

import json
from dataclasses import dataclass

from .core import EvaluationError, Geometry
from .excited import (
    PySCFTDABackend,
    TDAReference,
    TDASettings,
    _digest,
    _load_runtime,
    _make_molecule,
    _reference_from_solved,
    _solve,
    build_tda_reference,
    select_tda_state,
    transition_overlap,
)
from .selection_config import _group_id


@dataclass(frozen=True, eq=False)
class StateFamilyReference:
    """Immutable common-core reference and a complete prepared partition set."""

    core: TDAReference
    partitions: tuple[tuple[str, TDAReference], ...]

    def __post_init__(self):
        if type(self.core) is not TDAReference:
            raise TypeError("The common core requires a CPU RHF/TDA reference")
        settings = self.core.settings
        if settings.charge != 0 or settings.multiplicity != 1 or self.core.charges is not None:
            raise ValueError("The state family requires isolated neutral singlets")
        partitions = tuple(sorted(((_group_id(name), ref) for name, ref in self.partitions), key=lambda item: item[0]))
        if not 2 <= len(partitions) <= 64 or len({name for name, _ in partitions}) != len(partitions):
            raise ValueError("A state family requires two to 64 distinct prepared partitions")
        identities = {}
        memberships = []
        for index, reference in enumerate((self.core, *(ref for _, ref in partitions))):
            if (type(reference) is not TDAReference or reference.settings != settings
                    or reference.runtime != self.core.runtime or reference.charges is not None):
                raise ValueError("Every family reference must share method, runtime and boundary")
            geom = reference.qm_geometry
            if geom.periodic or geom.atom_ids != tuple(sorted(geom.atom_ids)):
                raise ValueError("Family reference atoms must be finite and sorted by stable ID")
            for atom, symbol, xyz in zip(geom.atom_ids, geom.symbols, geom.positions):
                identity = (symbol, tuple(xyz))
                if atom in identities and identities[atom] != identity:
                    raise ValueError("Shared reference atoms have different symbols or coordinates")
                identities[atom] = identity
            if index > 0:
                if not set(self.core.qm_geometry.atom_ids) < set(geom.atom_ids):
                    raise ValueError("Each partition must contain the complete common core and an environment group")
                memberships.append(geom.atom_ids)
        if len(set(memberships)) != len(memberships):
            raise ValueError("Prepared partition memberships must be distinct")
        object.__setattr__(self, "partitions", partitions)

    def reference(self, partition_id):
        for name, reference in self.partitions:
            if name == partition_id:
                return reference
        raise ValueError("Unknown prepared partition identity")

    def to_record(self):
        payload = {"schema_version": 1, "model": "finite-common-core-rhf-tda-family",
                   "core": self.core.to_record(),
                   "partitions": [{"partition_id": name, "reference": ref.to_record()}
                                  for name, ref in self.partitions]}
        return {**payload, "sha256": _digest(payload)}

    @property
    def sha256(self):
        return self.to_record()["sha256"]

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict) or set(record) != {"schema_version", "model", "core", "partitions", "sha256"}:
            raise ValueError("State-family reference has missing or unknown fields")
        if (type(record["schema_version"]) is not int or record["schema_version"] != 1
                or record["model"] != "finite-common-core-rhf-tda-family"
                or _digest({k: v for k, v in record.items() if k != "sha256"}) != record["sha256"]):
            raise ValueError("State-family reference identity is invalid")
        if not isinstance(record["partitions"], list) or any(
                not isinstance(row, dict) or set(row) != {"partition_id", "reference"}
                for row in record["partitions"]):
            raise ValueError("Every partition reference requires its explicit ID")
        result = cls(TDAReference.from_record(record["core"]),
                     tuple((row["partition_id"], TDAReference.from_record(row["reference"]))
                           for row in record["partitions"]))
        if result.sha256 != record["sha256"]:
            raise ValueError("State-family reference is not canonical")
        return result


def _scores(runtime, density, molecule, overlap, solved):
    cross = runtime.gto.intor_cross("int1e_ovlp", molecule, solved.molecule)
    return [transition_overlap(density, transition, overlap, solved.overlap, cross)
            for transition in solved.transitions]


@dataclass(frozen=True)
class PartitionState:
    """Transient successful provider result; never serialized as an executable object."""

    partition_id: str
    geometry: Geometry
    solved: object
    selected_index: int
    diagnostics_json: str

    @property
    def diagnostics(self):
        return json.loads(self.diagnostics_json)


def pairwise_correspondence(states, settings, runtime):
    """Check both directions for every active pair, including competing roots."""
    states = tuple(states)
    if not states or len({s.partition_id for s in states}) != len(states):
        raise ValueError("Distinct successful active partition states are required")
    reports = []
    for left in states:
        for right in states:
            if left.partition_id == right.partition_id:
                continue
            scores = _scores(runtime, left.solved.transitions[left.selected_index],
                             left.solved.molecule, left.solved.overlap, right.solved)
            selected, diagnostics = select_tda_state(scores, right.solved.energies, settings)
            if selected != right.selected_index:
                raise EvaluationError("Active partitions disagree on the selected excitation")
            reports.append({"source_partition": left.partition_id,
                            "target_partition": right.partition_id, **diagnostics})
    return tuple(reports)


class FamilyTDABackend(PySCFTDABackend):
    """Existing analytic response with own-reference and common-core admission."""

    def __init__(self, family, partition_id):
        if not isinstance(family, StateFamilyReference):
            raise TypeError("A prepared StateFamilyReference is required")
        self.family = family
        self.partition_id = _group_id(partition_id)
        reference = family.reference(partition_id)
        super().__init__(family.core.settings, reference)
        self._family_identity = family.sha256
        self._core_molecule = _make_molecule(family.core.qm_geometry, self.settings, self._runtime)
        self._core_overlap = self._core_molecule.intor_symmetric("int1e_ovlp")
        self.last_state = None
        self._pending = None

    @property
    def charge(self):
        return self.settings.charge

    @property
    def multiplicity(self):
        return self.settings.multiplicity

    @property
    def basis(self):
        return self.settings.basis

    @property
    def xc(self):
        return self.settings.xc

    def _select_surface(self, solved, forces):
        selected, diagnostics = super()._select_surface(solved, forces)
        scores = _scores(self._runtime, self.family.core.transition_density,
                         self._core_molecule, self._core_overlap, solved)
        common, core_diagnostics = select_tda_state(scores, solved.energies, self.settings)
        if selected != common:
            raise EvaluationError("Partition reference and common core select different excitations")
        diagnostics.update(family_sha256=self._family_identity, partition_id=self.partition_id,
                           common_core_reference_sha256=self.family.core.sha256,
                           common_core=core_diagnostics)
        self._pending = (solved, selected)
        return selected, diagnostics

    def evaluate_surface(self, qm_geometry, *, forces=True):
        self.last_state = self._pending = None
        try:
            if (self.family.sha256 != self._family_identity
                    or self.reference is not self.family.reference(self.partition_id)
                    or self.settings != self.family.core.settings
                    or _load_runtime(self.settings).record != self.family.core.runtime):
                raise EvaluationError("State-family provider identity changed")
            result = super().evaluate_surface(qm_geometry, forces=forces)
            solved, selected = self._pending
            self.last_state = PartitionState(self.partition_id, qm_geometry, solved, selected,
                                             json.dumps(dict(result.diagnostics), sort_keys=True, allow_nan=False))
            return result
        except BaseException:
            self.last_state = self._pending = None
            raise


def build_state_family_reference(geometry, *, core_atom_ids, partitions, settings, initial_root, label):
    """Prepare one core seed and select each finite partition by actual AO overlap."""
    if not isinstance(geometry, Geometry) or geometry.periodic or type(settings) is not TDASettings:
        raise ValueError("A finite geometry and explicit CPU RHF/TDA settings are required")
    core = geometry.select(tuple(sorted(core_atom_ids)))
    anchor = build_tda_reference(core, settings, initial_root=initial_root, label=label)
    runtime = _load_runtime(settings)
    molecule = _make_molecule(core, settings, runtime)
    overlap = molecule.intor_symmetric("int1e_ovlp")
    refs = []
    states = []
    for name, ids in partitions:
        name = _group_id(name)
        atoms = geometry.select(tuple(sorted(ids)))
        solved = _solve(atoms, settings, runtime)
        scores = _scores(runtime, anchor.transition_density, molecule, overlap, solved)
        selected, diagnostics = select_tda_state(scores, solved.energies, settings)
        reference = _reference_from_solved(atoms, settings, solved, runtime, reference_type=TDAReference,
                                           initial_root=selected+1, label=label+":"+name)
        refs.append((name, reference))
        states.append(PartitionState(name, atoms, solved, selected, json.dumps(diagnostics)))
    family = StateFamilyReference(anchor, tuple(refs))
    pairwise_correspondence(states, settings, runtime)
    return family


def prepare_family(request_path, output):
    """Prepare a portable common-core family from an explicit geometry request."""
    from pathlib import Path

    from .adaptive_config import _fields
    from .config import file_digest
    from .selection_config import read_json

    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    request = read_json(request_path.read_text())
    _fields(request, ("schema_version", "geometry", "core_atom_ids", "partitions",
                      "settings", "initial_root", "label"), name="State-family request")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("State-family request requires schema_version integer 1")
    _fields(request["geometry"], ("atom_ids", "symbols", "positions_angstrom"), name="Geometry")
    if not isinstance(request["partitions"], list) or len(request["partitions"]) < 2:
        raise ValueError("State-family request requires multiple explicit partitions")
    for part in request["partitions"]:
        _fields(part, ("partition_id", "atom_ids"), name="Prepared partition")
    if output.exists():
        raise FileExistsError("State-family preparation will not replace an existing reference")
    geometry = Geometry(tuple(request["geometry"]["atom_ids"]), tuple(request["geometry"]["symbols"]),
                        request["geometry"]["positions_angstrom"])
    reference = build_state_family_reference(
        geometry, core_atom_ids=request["core_atom_ids"],
        partitions=tuple((p["partition_id"], p["atom_ids"]) for p in request["partitions"]),
        settings=TDASettings(**request["settings"]), initial_root=request["initial_root"],
        label=request["label"])
    with output.open("x") as stream:
        stream.write(json.dumps(reference.to_record(), indent=2, allow_nan=False)+"\n")
    return {"path": str(output), "sha256": file_digest(output),
            "family_sha256": reference.sha256,
            "qualification": "Prepared identity only; derivatives and dynamics require separate tests"}
