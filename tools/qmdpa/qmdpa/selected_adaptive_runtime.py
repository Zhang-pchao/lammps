# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Bind complete selected-family correspondence to each accepted native frame."""

import copy

import numpy as np

from .core import Geometry
from .excited import _digest, select_tda_state
from .molecular_adaptive import QuinticMolecularWeights
from .selected_adaptive import HAMILTONIAN
from .state_family import StateFamilyReference
from .state_records import geometry_digest, validate_state_record


def _selection(diagnostic, settings):
    index, checked = select_tda_state(diagnostic["candidate_overlaps"],
                                     diagnostic["candidate_excitation_energies_ev"], settings)
    if _digest({name: diagnostic.get(name) for name in checked}) != _digest(checked):
        raise ValueError("Stored state correspondence disagrees with its candidate scores")
    return index


def validate_frame(frame, config):
    record = frame.get("selected_family")
    if config.get("hamiltonian") != HAMILTONIAN:
        if record is not None:
            raise ValueError("Unexpected selected-family record in this Hamiltonian")
        return
    fields = {"schema_version", "hamiltonian", "family_sha256", "common_core_reference_sha256",
              "atom_ids", "positions_angstrom", "partition_ids", "qm_atom_ids", "weights",
              "active_partition_ids", "states", "pair_correspondence", "potential_ev",
              "forces_ev_angstrom", "switching_forces_ev_angstrom", "step", "sha256"}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("An accepted selected-family frame requires its complete correspondence record")
    if (record["sha256"] != _digest({k: v for k, v in record.items() if k != "sha256"})
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["hamiltonian"] != HAMILTONIAN
            or type(record["step"]) is not int or record["step"] != frame["step"]):
        raise ValueError("Selected-family record identity or step differs")
    family = StateFamilyReference.from_record(config["family_reference"])
    if (record["family_sha256"] != family.sha256
            or record["common_core_reference_sha256"] != family.core.sha256):
        raise ValueError("Selected-family reference differs from the configured identity")
    full = config["full_region"]
    atoms = {a["id"]: a for a in frame["atoms"]}
    if len(atoms) != len(frame["atoms"]) or set(atoms) != set(full["atom_ids"]):
        raise ValueError("Selected-family frame has invalid native atom ownership")
    ordered = [atoms[a] for a in full["atom_ids"]]
    geometry = Geometry(tuple(full["atom_ids"]), tuple(full["symbols"]), [a["x"] for a in ordered])
    if (record["atom_ids"] != list(geometry.atom_ids)
            or any(a["image"] != [0, 0, 0] for a in ordered)
            or not np.array_equal(record["positions_angstrom"], geometry.positions)):
        raise ValueError("Selected-family coordinates or finite atom identities differ")
    policy = QuinticMolecularWeights(**{k: v for k, v in config["switch"].items() if k != "method"})
    weights = policy.evaluate(geometry)
    parts = config["partitions"]
    ids = [p["partition_id"] for p in parts]
    # Configuration admission already places partitions in switch subset order.
    active = [name for i, name in enumerate(ids)
              if weights.weights[i] != 0 or np.any(weights.gradients_per_angstrom[i])]
    if (record["partition_ids"] != ids or record["active_partition_ids"] != active
            or record["qm_atom_ids"] != [list(family.reference(name).qm_geometry.atom_ids) for name in ids]
            or not np.array_equal(record["weights"], weights.weights)
            or not isinstance(record["states"], dict) or set(record["states"]) != set(active)):
        raise ValueError("Selected-family weights, coverage or active state records differ")
    for name in active:
        reference = family.reference(name)
        diagnostic = record["states"][name]
        provider_record = {
            "schema_version": 1, "method": "rhf-tda", "reference_sha256": reference.sha256,
            "geometry_sha256": geometry_digest(geometry.select(reference.qm_geometry.atom_ids)),
            "selected_root": diagnostic["selected_root"],
            "provider_energy_ev": diagnostic["ground_energy_ev"]+diagnostic["excitation_energy_ev"],
            "diagnostics": diagnostic}
        validate_state_record(provider_record, reference)
        common = _selection(diagnostic["common_core"], family.core.settings)
        if (diagnostic.get("family_sha256") != family.sha256 or diagnostic.get("partition_id") != name
                or diagnostic.get("common_core_reference_sha256") != family.core.sha256
                or common+1 != diagnostic["selected_root"]):
            raise ValueError("Selected-family own and common-core states disagree")
    expected = {(a, b) for a in active for b in active if a != b}
    seen = set()
    if not isinstance(record["pair_correspondence"], list):
        raise TypeError("Selected-family correspondence requires all directed active pairs")
    for pair in record["pair_correspondence"]:
        if not isinstance(pair, dict):
            raise TypeError("Invalid pair correspondence")
        key = (pair.get("source_partition"), pair.get("target_partition"))
        if key not in expected or key in seen:
            raise ValueError("Missing, repeated or foreign active pair correspondence")
        seen.add(key)
        selected = _selection(pair, family.core.settings)
        target = record["states"][key[1]]
        if (selected+1 != target["selected_root"]
                or tuple(pair["candidate_excitation_energies_ev"]) != tuple(target["candidate_excitation_energies_ev"])):
            raise ValueError("Active pair correspondence selects a different target state")
    if seen != expected:
        raise ValueError("Selected-family correspondence does not cover all directed active pairs")
    force = np.asarray(record["forces_ev_angstrom"])
    switching = np.asarray(record["switching_forces_ev_angstrom"])
    if (force.shape != geometry.positions.shape or switching.shape != force.shape
            or not np.isfinite(force).all() or not np.isfinite(switching).all()
            or not np.isfinite(record["potential_ev"])
            or abs(record["potential_ev"]-frame["potential_eV"]) > 1e-7
            or not np.allclose(force, [a["f"] for a in ordered], atol=1e-6, rtol=0)):
        raise ValueError("Selected-family and native total energy/forces differ")


def attach_frame(frame, coupling, config):
    if config.get("hamiltonian") != HAMILTONIAN:
        return
    if coupling.failure is not None or coupling.last_step != frame["step"] or coupling.comm.rank != 0:
        raise ValueError("Selected-family frame requires a successful current owner callback")
    record = copy.deepcopy(coupling.selected_family_observer.last_state_record)
    if not isinstance(record, dict):
        raise ValueError("Selected-family callback did not publish a successful state")  # noqa: TRY004 - Accepted-frame contract.
    record["step"] = frame["step"]
    record["sha256"] = _digest(record)
    frame["selected_family"] = record
    validate_frame(frame, config)


def compare_restart(frame, previous, config):
    validate_frame(frame, config)
    validate_frame(previous, config)
    if config.get("hamiltonian") != HAMILTONIAN:
        return
    actual, expected = frame["selected_family"], previous["selected_family"]
    for key in ("family_sha256", "common_core_reference_sha256", "atom_ids", "partition_ids",
                "qm_atom_ids", "weights", "active_partition_ids"):
        if actual[key] != expected[key]:
            raise ValueError("Recomputed selected-family membership or identity differs")
    for name in actual["states"]:
        if actual["states"][name]["selected_root"] != expected["states"][name]["selected_root"]:
            raise ValueError("Recomputed selected-family selected root differs")
    # Existing native restart checks independently compare complete E/F and x/v.
    # No electronic amplitude or adaptive history is propagated by this model.
