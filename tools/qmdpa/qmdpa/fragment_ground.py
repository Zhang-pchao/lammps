# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Ground surface of the explicit finite fragment interaction model.

Reuse the existing subtractive baseline and fragment differences. No selected
state, embedding field or new physical term is introduced by this native route.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .charges import _invalid_constant, _unique_object
from .config import _validate_config as validate_mechanical
from .config import validate_region
from .core import Evaluation, SubtractiveCalculator, _atom_ids, _evaluate
from .fragment_response_config import validate_fragment_pairs
from .interaction_delta import CorrectedTerms

GROUND_FRAGMENT_HAMILTONIAN = "finite-mechanical-fragment-response"
GROUND_FRAGMENT_MODEL = "finite-ground-fragment-interactions-v1"


def _ground_settings(qm):
    from .pbe0_tda import PBE0TDASettings

    options = dict(qm)
    if options.pop("nlc_grid_level", None) is not None:
        raise ValueError("Matched ground fragment terms do not support NLC")
    # Reuse exactly the existing ground settings and source pin. No TDA
    # calculation or selected-state reference is constructed.
    # Use the same explicit profile validator as every matched selected term.
    # A smaller basis alone cannot silently change its required grid profile.
    return PBE0TDASettings(**options)


def load_ground_fragment_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(), object_pairs_hook=_unique_object,
                     parse_constant=_invalid_constant)
    return _validate_config(raw, path)


def _validate_config(raw, path):
    """Reuse mechanical input validation, then bind the complete pair ledger."""
    if not isinstance(raw, dict) or raw.get("hamiltonian") != GROUND_FRAGMENT_HAMILTONIAN:
        raise ValueError("Declare the finite ground fragment Hamiltonian explicitly")
    required = {"qm_region", "environment_region", "fragment_interactions",
                "full_charge_spin", "region_charge_spin", "qm"}
    if not required <= set(raw):
        raise ValueError("Ground fragments require both regions and explicit interaction terms")
    request = deepcopy(raw)
    environment = request.pop("environment_region")
    fragments = request.pop("fragment_interactions")
    reuse = request.pop("reuse_ground_terms", False)
    if type(reuse) is not bool:
        raise ValueError("reuse_ground_terms must be an explicit boolean")
    if "region" in request:
        raise ValueError("Use qm_region for the ground fragment route")
    request["region"] = request.pop("qm_region")
    request["hamiltonian"] = "fixed-region-mechanical"
    if ("temperature_K" in request) != ("velocity_seed" in request):
        raise ValueError("temperature_K and velocity_seed must be supplied together")
    config = validate_mechanical(request, path)
    if config["periodic"] is not False or Path(config["model"]).suffix != ".pt2":
        raise ValueError("Ground fragment dynamics requires finite boundaries and a compiled .pt2 model")
    if config["provider"]["engine"] != "pyscf":
        raise ValueError("Matched ground fragments currently require the CPU PySCF provider")
    if config["provider"]["state"]["method"] != "ground":
        raise ValueError("Ground fragment propagation cannot carry a selected-state provider")
    for key in ("full_charge_spin", "region_charge_spin"):
        if (config[key] != [0, 1] or any(type(v) is not int for v in config[key])):
            raise ValueError("Ground fragment terms require explicit neutral singlets [0, 1]")
    _ground_settings(config["qm"])
    qm = config.pop("region")
    validate_region(environment)
    if not set(environment["symbols"]) <= set(config["type_symbols"]):
        raise ValueError("Environment symbols must occur in type_symbols")
    if set(qm["atom_ids"]).intersection(environment["atom_ids"]):
        raise ValueError("QM and environment regions must be disjoint")
    if (not isinstance(fragments, dict)
            or set(fragments) != {"model", "pairs", "alignments"}
            or fragments["model"] != GROUND_FRAGMENT_MODEL):
        raise ValueError("Ground fragments require model, pairs and alignments only")
    pairs, alignments = validate_fragment_pairs(
        fragments["pairs"], fragments["alignments"], qm, environment, "pbe0-ground")
    config.update(hamiltonian=GROUND_FRAGMENT_HAMILTONIAN, qm_region=qm,
                  environment_region=environment, reuse_ground_terms=reuse,
                  fragment_interactions={"model": GROUND_FRAGMENT_MODEL,
                                         "pairs": pairs, "alignments": alignments})
    config["assembly"] = {"pair_style": "zero",
                          "external_quantity": "total-energy-and-all-forces",
                          "model_format": "pt2", "ground_method": "rks-xc-pbe0",
                          "selected_state_terms": []}
    if reuse:
        config["assembly"]["ground_result_reuse"] = "last-successful-exact-input"
    physics = {k: v for k, v in config.items() if k not in {
        "model", "data_file", "steps", "sample_interval", "temperature_K",
        "velocity_seed", "physics_sha256"}}
    config["physics_sha256"] = hashlib.sha256(json.dumps(
        physics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return config


class GroundFragmentCalculator:
    """A finite full-system total from existing baseline/correction evaluators."""

    def __init__(self, baseline, correction, *, qm_atom_ids, environment_atom_ids):
        if not isinstance(baseline, SubtractiveCalculator):
            raise TypeError("Ground fragment baseline must be the subtractive calculator")
        if not callable(getattr(correction, "evaluate", None)):
            raise TypeError("Ground fragment correction must supply complete energy and forces")
        self.baseline, self.correction = baseline, correction
        self.qm_atom_ids = _atom_ids(qm_atom_ids)
        self.environment_atom_ids = _atom_ids(environment_atom_ids)
        if set(self.qm_atom_ids).intersection(self.environment_atom_ids):
            raise ValueError("QM and environment regions must be disjoint")
        self._ids = self.qm_atom_ids + self.environment_atom_ids
        self._symbols = None

    def evaluate_terms(self, geometry):
        if geometry.periodic or set(geometry.atom_ids) != set(self._ids):
            raise ValueError("Ground fragment regions must cover the complete finite geometry")
        ordered = geometry.select(self._ids)
        if self._symbols is not None and ordered.symbols != self._symbols:
            raise ValueError("Fixed ground fragment element identities changed")
        baseline = self.baseline.evaluate(ordered, ordered.select(self.qm_atom_ids))
        correction = _evaluate(self.correction, ordered)
        rows = {atom: i for i, atom in enumerate(self._ids)}
        order = [rows[atom] for atom in geometry.atom_ids]
        baseline = Evaluation(baseline.energy, baseline.forces[order])
        correction = Evaluation(correction.energy, correction.forces[order])
        total = Evaluation(baseline.energy + correction.energy,
                           baseline.forces + correction.forces)
        self._symbols = ordered.symbols
        return CorrectedTerms(geometry.atom_ids, baseline, correction, total)

    def evaluate(self, geometry):
        return self.evaluate_terms(geometry).total


def _zero_pair(lmp, config):
    from .gap_anchor_cli import _zero_pair as setup

    setup(lmp, config)


def _ground_fragment_coupling(lmp, comm, config):
    from .backends import DeepMDBackend
    from .fragment_response import OneBodyAlignedPairSet, PairInteractionSet
    from .interaction_delta import CrossInteractionDelta
    from .lammps import AdditiveExternalCoupling
    from .pbe0_tda import PySCFPBE0GroundBackend
    from .region import FixedRegion
    from .runtime import _root_call

    if config["hamiltonian"] != GROUND_FRAGMENT_HAMILTONIAN:
        raise ValueError("Ground fragment driver requires its explicit Hamiltonian")
    qm, environment = config["qm_region"], config["environment_region"]
    region = FixedRegion(tuple(qm["atom_ids"] + environment["atom_ids"]),
                         tuple(qm["symbols"] + environment["symbols"]))
    types = tuple(config["type_symbols"])
    _root_call(comm, lambda: AdditiveExternalCoupling._validate_native_assembly(
        lmp, region, types))
    calculator = None

    def build():
        nonlocal calculator
        settings = _ground_settings(config["qm"])
        reuse = config.get("reuse_ground_terms", False)
        low = DeepMDBackend(config["model"], charge_spin=config["region_charge_spin"])
        full = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"])
        providers = {}

        def ground(ids):
            key = tuple(ids)
            if not reuse:
                return PySCFPBE0GroundBackend(settings)
            if key not in providers:
                providers[key] = PySCFPBE0GroundBackend(
                    settings, reuse_identical_geometry=True)
            return providers[key]

        baseline = SubtractiveCalculator(low, ground(qm["atom_ids"]), full_low=full)
        spec = config["fragment_interactions"]
        pairs = []
        for pair in spec["pairs"]:
            core, fragment = pair["core_region"]["atom_ids"], pair["fragment_region"]["atom_ids"]
            pairs.append(CrossInteractionDelta(
                ground(core + fragment), ground(core), ground(fragment), low, low, low,
                core_atom_ids=core, fragment_atom_ids=fragment))
        choices = tuple((v["pair_index"], v["side"]) for v in spec["alignments"])
        correction = (OneBodyAlignedPairSet(tuple(pairs), alignments=choices)
                      if choices else PairInteractionSet(tuple(pairs)))
        calculator = GroundFragmentCalculator(
            baseline, correction, qm_atom_ids=qm["atom_ids"],
            environment_atom_ids=environment["atom_ids"])

    _root_call(comm, build)
    return AdditiveExternalCoupling(
        lmp, region, calculator, type_symbols=types, comm=comm,
        fix_id="qmdpa_ground_fragments")
