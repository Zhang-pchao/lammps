# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit finite fragment-response input; validation imports no quantum engine."""

import math
from copy import deepcopy
from dataclasses import asdict

from .embedding import FixedEmbeddingCharges
from .fragment_response import ClusterGapTerms
from .interaction_delta_config import INTERACTION_MODEL, validate_interaction_delta
from .providers import load_tda_reference, validate_provider, validate_provider_regions

FRAGMENT_GAP_HAMILTONIAN = "finite-electrostatic-gap-anchor-fragment-response"
FRAGMENT_MODEL = "finite-fragment-response-v1"


def _pair_coverage(pairs):
    covered = set()
    for pair in pairs:
        selected = {frozenset((a, b)) for a in pair["core_region"]["atom_ids"]
                    for b in pair["fragment_region"]["atom_ids"]}
        if covered.intersection(selected):
            raise ValueError("Fragment corrections double-count an atomic pair")
        covered.update(selected)


def validate_fragment_pairs(pairs, alignments, qm, environment, method):
    """Validate shared ground/selected pair coverage and one-body ownership."""
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Declare a nonempty fragment interaction pair set")
    pairs = [validate_interaction_delta(p, qm, environment, method) for p in pairs]
    _pair_coverage(pairs)
    if not isinstance(alignments, list):
        raise ValueError("Declare one-body alignments explicitly, or use an empty list")
    covered = set()
    for choice in alignments:
        if (not isinstance(choice, dict) or set(choice) != {"pair_index", "side"}
                or type(choice["pair_index"]) is not int
                or not 0 <= choice["pair_index"] < len(pairs)
                or choice["side"] != "fragment"):
            raise ValueError("Native one-body alignment must select an environment fragment by pair index")
        ids = pairs[choice["pair_index"]]["fragment_region"]["atom_ids"]
        if covered.intersection(ids):
            raise ValueError("One-body alignment double-counts an environment atom")
        covered.update(ids)
    return pairs, deepcopy(alignments)


def validate_fragment_response(value, qm, environment, primary_provider, charges, *, config_path=None):
    """Pin complete regions, one-body coverage and two independently named states.

    This first native route uses matched PBE0 neutral singlets. Only environment
    monomers can be aligned: the baseline has already replaced its QM monomer.
    Inline references define fixed state identities, not initial coordinates.
    State correspondence is an explicit, hash-bound scientific declaration.
    """
    required = {"model", "pairs", "alignments", "cluster_gap"}
    if not isinstance(value, dict) or set(value) != required or value["model"] != FRAGMENT_MODEL:
        raise ValueError("Fragment response requires its model, pairs, alignments and cluster_gap")
    if primary_provider["state"]["method"] != "pbe0-tda":
        raise ValueError("Native fragment response requires the matched PBE0 route")
    primary = load_tda_reference("pbe0-tda", primary_provider["state"]["reference"])
    pairs, alignments = validate_fragment_pairs(
        value["pairs"], value["alignments"], qm, environment, "pbe0-tda")
    gap = value["cluster_gap"]
    required_gap = {"model", "core_region", "fragment_region",
                    "pair_provider", "field_provider", "state_correspondence"}
    if not isinstance(gap, dict) or set(gap) != required_gap or gap["model"] != ClusterGapTerms.model:
        raise ValueError("Declare the explicit cluster-minus-field gap, regions and both providers")
    regions = validate_interaction_delta({
        "model": INTERACTION_MODEL, "ground_method": "matched-pbe0",
        "core_region": gap["core_region"], "fragment_region": gap["fragment_region"],
        "charge_spin": {name: [0, 1] for name in ("pair", "core", "fragment")},
    }, qm, environment, "pbe0-tda")
    core, fragment = regions["core_region"], regions["fragment_region"]
    pair_region = {key: core[key] + fragment[key] for key in ("atom_ids", "symbols", "molecules")}
    by_id = dict(zip(charges.atom_ids, charges.charges_e))
    field = FixedEmbeddingCharges(fragment["atom_ids"],
        [by_id[i] for i in fragment["atom_ids"]], charges.source)
    if not math.isclose(math.fsum(field.charges_e), 0., abs_tol=1e-8, rel_tol=0):
        raise ValueError("The native cluster response requires a neutral fixed fragment field")
    normalized = deepcopy(value)
    normalized["pairs"] = pairs
    references = {}
    for name, embedded, region in (("pair", False, pair_region), ("field", True, core)):
        specification = gap[name + "_provider"]
        if config_path is not None:
            from .state_inputs import resolve_provider_reference

            specification = resolve_provider_reference(specification, config_path)
        provider = validate_provider(specification, electrostatic=embedded)
        if provider["state"]["method"] != "pbe0-tda":
            raise ValueError("Every fragment response state requires matched PBE0")
        reference = load_tda_reference("pbe0-tda", provider["state"]["reference"])
        # A different matrix capacity is necessary for different region sizes.
        # All physical settings and all remaining numerical controls match.
        def settings(ref):
            return {k: v for k, v in asdict(ref.settings).items()
                    if k != "tda_dense_max_dimension"}
        if settings(reference) != settings(primary) or reference.runtime_json != primary.runtime_json:
            raise ValueError("Fragment state settings or runtime differ from the primary state")
        validate_provider_regions(provider, region, periodic=False,
            environment_region=fragment if embedded else None, charges=field if embedded else None)
        normalized["cluster_gap"][name + "_provider"] = provider
        references[name] = reference
    declaration = gap["state_correspondence"]
    identities = {"anchor_sha256": primary.sha256, "pair_sha256": references["pair"].sha256,
                  "field_sha256": references["field"].sha256}
    if (not isinstance(declaration, dict) or set(declaration) != {"source", *identities}
            or not isinstance(declaration["source"], str) or not declaration["source"].strip()
            or any(declaration[name] != sha for name, sha in identities.items())):
        raise ValueError("State correspondence must name its evidence and bind all three references")
    return normalized
