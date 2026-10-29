#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Select an existing fixed-region Hamiltonian from an explicit configuration.

This module owns no energy, force, state or checkpoint implementation. Loading
runs on the existing native owner's collective configuration path; all ranks
then use the same normalized Hamiltonian. Validation alone runs no evaluator.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .charges import _invalid_constant, _unique_object
from .fragment_ground import GROUND_FRAGMENT_HAMILTONIAN
from .fragment_response_config import FRAGMENT_GAP_HAMILTONIAN
from .interaction_delta_config import CORRECTED_GAP_HAMILTONIAN

MECHANICAL = "fixed-region-mechanical"
MOLECULAR = "finite-molecular-mechanical"
MOLECULAR_EE = "finite-molecular-additive"
WATER_BUFFER = "finite-water-buffer-mechanical"
REACTIVE_WATER = "finite-reactive-water-mechanical"
REACTIVE_EE = "finite-reactive-water-additive"
REFERENCE = "finite-electrostatic-reference-subtractive"
PERIODIC_EE = "periodic-electrostatic-reference-subtractive"
EHRENFEST = "finite-two-state-ehrenfest-mechanical"
SELECTED_FAMILY = "finite-molecular-selected-mechanical"
_ROUTES = {
    SELECTED_FAMILY: ("selected_adaptive_config", "load_selected_adaptive_config",
                      "selected_adaptive_cli", "_zero_pair", "_selected_adaptive_coupling"),
    EHRENFEST: ("ehrenfest_config", "load_ehrenfest_config", "ehrenfest_cli",
                 "_zero_pair", "_ehrenfest_coupling"),
    PERIODIC_EE: ("periodic_ee_config", "load_periodic_ee_config", "periodic_ee_cli",
                   "_zero_pair", "_periodic_ee_coupling"),
    REACTIVE_WATER: ("reactive_config", "load_reactive_config", "reactive_cli",
                     "_zero_pair", "_reactive_coupling"),
    REFERENCE: ("reference_config", "load_reference_config", "reference_cli",
                "_zero_pair", "_reference_coupling"),
    MOLECULAR: ("adaptive_config", "load_adaptive_config", "adaptive_cli",
                "_zero_pair", "_adaptive_coupling"),
    MECHANICAL: ("config", "load_config", "cli", "_mechanical_pair", "_mechanical_coupling"),
    "finite-electrostatic-additive": (
        "additive_config", "load_additive_config", "additive_cli", "_zero_pair", "_additive_coupling"
    ),
    "finite-electrostatic-gap-anchor": (
        "gap_anchor_config", "load_gap_anchor_config", "gap_anchor_cli",
        "_zero_pair", "_gap_anchor_coupling"
    ),
}
_ROUTES[REACTIVE_EE] = _ROUTES[REACTIVE_WATER]
_ROUTES[MOLECULAR_EE] = _ROUTES[MOLECULAR]
_ROUTES[WATER_BUFFER] = _ROUTES[MOLECULAR]
_ROUTES[CORRECTED_GAP_HAMILTONIAN] = _ROUTES["finite-electrostatic-gap-anchor"]
_ROUTES[FRAGMENT_GAP_HAMILTONIAN] = _ROUTES["finite-electrostatic-gap-anchor"]
_ROUTES[GROUND_FRAGMENT_HAMILTONIAN] = (
    "fragment_ground", "load_ground_fragment_config", "fragment_ground",
    "_zero_pair", "_ground_fragment_coupling"
)


def _route(hamiltonian):
    if not isinstance(hamiltonian, str) or hamiltonian not in _ROUTES:
        raise ValueError(
            "Unsupported Hamiltonian. Choose "
            + ", ".join(_ROUTES)
            + "; no model is selected by fallback."
        )
    return _ROUTES[hamiltonian]


def _function(module, name):
    return getattr(importlib.import_module("."+module, __package__), name)


def _normalize_selection(raw):
    """Consume an optional user selector without changing the physical identity."""
    from copy import deepcopy

    config = deepcopy(raw)
    if "coupling" not in config:
        return config
    choice = config.pop("coupling")
    required = {"surface", "embedding"}
    if (not isinstance(choice, dict) or not required <= set(choice)
            or set(choice) - required - {"ee_model", "me_model"}):
        raise ValueError("coupling requires surface, embedding and an optional matching me_model or ee_model")
    surface, embedding = choice["surface"], choice["embedding"]
    if choice == {"surface": "selected", "embedding": "mechanical", "me_model": "molecular-common-core"}:
        if "hamiltonian" in config and config["hamiltonian"] != SELECTED_FAMILY:
            raise ValueError("Selected-family selector conflicts with the explicit Hamiltonian")
        config["hamiltonian"] = SELECTED_FAMILY
        return config
    if choice == {"surface": "two-state", "embedding": "mechanical", "me_model": "ehrenfest"}:
        if "hamiltonian" in config and config["hamiltonian"] != EHRENFEST:
            raise ValueError("Ehrenfest selector conflicts with the explicit Hamiltonian")
        config["hamiltonian"] = EHRENFEST
        return config
    if surface == "constrained":
        if choice != {"surface": "constrained", "embedding": "electrostatic", "ee_model": "additive"}:
            raise ValueError("Native cDFT currently requires fixed additive electrostatic embedding")
        if config.get("provider", {}).get("state", {}).get("method") != "uks-hirshfeld-cdft":
            raise ValueError("Constrained coupling requires an explicit cDFT provider")
        if config.get("hamiltonian", "finite-electrostatic-additive") != "finite-electrostatic-additive":
            raise ValueError("Constrained coupling conflicts with the Hamiltonian")
        config["hamiltonian"] = "finite-electrostatic-additive"
        return config
    if surface not in ("ground", "selected"):
        raise ValueError("coupling.surface must be ground or selected")
    if embedding == "mechanical":
        if "ee_model" in choice:
            raise ValueError("Mechanical coupling cannot specify ee_model")
        model = choice.get("me_model")
        if model not in (None, "fragment-response", "molecular", "water-buffer", "reactive-water"):
            raise ValueError("Mechanical me_model must be fragment-response, molecular, water-buffer or reactive-water when supplied")
        if model in ("fragment-response", "molecular", "water-buffer", "reactive-water") and surface != "ground":
            raise ValueError("Mechanical fragment response currently requires the ground surface")
        hamiltonian = {"fragment-response": GROUND_FRAGMENT_HAMILTONIAN,
                       "molecular": MOLECULAR, "water-buffer": WATER_BUFFER,
                       "reactive-water": REACTIVE_WATER, None: MECHANICAL}[model]
    elif embedding == "electrostatic":
        if "me_model" in choice:
            raise ValueError("Electrostatic coupling cannot specify me_model")
        model = choice.get("ee_model")
        if model in ("reference-subtractive", "periodic-reference-subtractive") and surface != "ground":
            raise ValueError("Reference-subtractive dynamics currently requires ground state")
        if model not in ("reference-subtractive", "periodic-reference-subtractive", "additive", "molecular-additive", "reactive-water-additive", "gap-anchor", "gap-anchor-cross-delta", "gap-anchor-fragment-response"):
            raise ValueError("Electrostatic coupling requires explicit ee_model additive, gap-anchor, gap-anchor-cross-delta or gap-anchor-fragment-response")
        if model == "gap-anchor" and surface == "ground":
            raise ValueError("Ground gap-anchor reduces to GS-ME; select mechanical explicitly")
        if model in ("gap-anchor-cross-delta", "gap-anchor-fragment-response") and surface == "ground":
            raise ValueError("The corrected native driver currently requires a selected PBE0 state")
        if model in ("molecular-additive", "reactive-water-additive") and surface != "ground":
            raise ValueError("Molecular additive coupling requires the ground surface")
        hamiltonian = (REACTIVE_EE if model == "reactive-water-additive" else
                       PERIODIC_EE if model == "periodic-reference-subtractive" else
                       MOLECULAR_EE if model == "molecular-additive" else "finite-electrostatic-" + model)
    else:
        raise ValueError("coupling.embedding must be mechanical or electrostatic")
    if "hamiltonian" in config and config["hamiltonian"] != hamiltonian:
        raise ValueError("coupling conflicts with the explicit Hamiltonian")
    provider = config.get("provider")
    state = provider.get("state") if isinstance(provider, dict) else None
    method = state.get("method") if isinstance(state, dict) else None
    if surface == "selected" and method not in ("rhf-tda", "pbe0-tda"):
        raise ValueError("Selected coupling requires an explicit supported provider.state reference")
    if surface == "ground" and provider is not None and method != "ground":
        raise ValueError("Ground coupling conflicts with provider.state")
    config["hamiltonian"] = hamiltonian
    return config


def _describe_selection(config):
    if config["hamiltonian"] == SELECTED_FAMILY:
        return {"surface": "selected", "embedding": "mechanical", "me_model": "molecular-common-core"}
    if config["hamiltonian"] == EHRENFEST:
        return {"surface": "two-state", "embedding": "mechanical", "me_model": "ehrenfest"}
    if config["provider"]["state"]["method"] == "uks-hirshfeld-cdft":
        return {"surface": "constrained", "embedding": "electrostatic", "ee_model": "additive"}
    mechanical = config["hamiltonian"] in (MECHANICAL, GROUND_FRAGMENT_HAMILTONIAN, MOLECULAR, WATER_BUFFER, REACTIVE_WATER)
    result = {"surface": "ground" if config["provider"]["state"]["method"] == "ground" else "selected",
              "embedding": "mechanical" if mechanical else "electrostatic"}
    if config["hamiltonian"] == GROUND_FRAGMENT_HAMILTONIAN:
        result["me_model"] = "fragment-response"
    if config["hamiltonian"] == MOLECULAR:
        result["me_model"] = "molecular"
    if config["hamiltonian"] == WATER_BUFFER:
        result["me_model"] = "water-buffer"
    if config["hamiltonian"] == REACTIVE_WATER:
        result["me_model"] = "reactive-water"
    if not mechanical:
        result["ee_model"] = ("reactive-water-additive" if config["hamiltonian"] == REACTIVE_EE else
                              "periodic-reference-subtractive" if config["hamiltonian"] == PERIODIC_EE else
                              "molecular-additive" if config["hamiltonian"] == MOLECULAR_EE
                              else config["hamiltonian"].removeprefix("finite-electrostatic-"))
    return result


def _energy_contract(config):
    """Describe existing terms; no backend calls or extra electronic calculations."""
    hamiltonian = config["hamiltonian"]
    selected = config["provider"]["state"]["method"] != "ground"
    if hamiltonian == EHRENFEST:
        return {"native_baseline": "zero", "external_energy": "complete-expectation-potential",
                "state_total": "U_i=DPA(full)+E_i(Q)-DPA(Q)",
                "potential": "sum_i |c_i|^2 U_i",
                "force_definition": "population-weighted-state-forces-minus-2-Re(conj(c0)*c1)-gap-d01",
                "derivative_coupling": "d01=<0|grad 1>; environment rows zero",
                "electronic_generator": "diag(U)-i*hbar*(v.d)",
                "scope": "Two fixed-reference real singlets; no ETF, hopping, SOC or adaptive region"}
    if hamiltonian == SELECTED_FAMILY:
        return {"native_baseline": "zero", "external_energy": "complete-total",
                "potential": "DPA(full)+sum_a w_a*(E_selected,a-DPA(Qa))",
                "force_definition": "weighted-complete-forces-minus-energy-times-weight-gradients",
                "partition_policy": config["switch"],
                "correspondence": "own-reference-and-common-core-and-all-directed-active-pairs",
                "scope": "Finite neutral CPU RHF/TDA, fixed core, complete same-composition fixed-k family"}
    if hamiltonian == MOLECULAR_EE:
        return {
            "total_terms": [{"name": "normalized_weighted_complete_additive_totals",
                             "coefficient": 1, "force_coordinates": "all"}],
            "partition_terms": ["low_environment", "embedded_ground", "cross_lj", "declared_offset"],
            "native_baseline": "zero", "external_energy": "complete-total",
            "force_definition": "weighted-complete-forces-minus-energy-times-weight-gradients",
            "partition_policy": config["switch"],
            "scope": "Fixed-charge complete fixed-occupancy family; no excited-state map or mutual polarization",
        }
    if hamiltonian == REACTIVE_EE:
        return {
            "total_terms": [{"name": "weighted_reactive_additive_totals",
                             "coefficient": 1, "force_coordinates": "all"}],
            "partition_terms": ["low_environment", "embedded_ground", "cross_lj", "water_offset"],
            "native_baseline": "zero", "external_energy": "complete-total",
            "force_definition": "weighted-complete-forces-minus-energy-times-weight-gradients",
            "partition_policy": config["reactive_region"],
            "reference": "isolated-water-difference-per-additional-QM-oxygen",
            "scope": "Finite declared proton channels; fixed charges; nonempty neutral environment",
        }
    if hamiltonian == REACTIVE_WATER:
        return {
            "total_terms": [{"name": "low_full", "coefficient": 1, "force_coordinates": "all"},
                            {"name": "weighted_reactive_ground_corrections", "coefficient": 1,
                             "force_coordinates": "all"}],
            "native_baseline": "zero", "external_energy": "complete-total",
            "force_definition": "weighted-complete-forces-minus-energy-times-weight-gradients",
            "partition_policy": config["reactive_region"],
            "reference": "isolated-water-difference-per-additional-QM-oxygen",
            "scope": "Declared finite proton channels; all ionic defects and mixed transfer endpoints in neutral QM",
        }
    if hamiltonian in (MOLECULAR, WATER_BUFFER):
        return {
            "total_terms": [{"name": "low_full", "coefficient": 1, "force_coordinates": "all"},
                            {"name": "normalized_weighted_ground_corrections", "coefficient": 1,
                             "force_coordinates": "all"}],
            "native_baseline": "zero", "external_energy": "complete-total",
            "force_definition": "weighted-complete-forces-minus-energy-times-weight-gradients",
            "partition_policy": config["switch"],
            "scope": ("Protected aqueous core with variable complete-water subsets and fixed reference offsets"
                      if hamiltonian == WATER_BUFFER else
                      "Complete fixed-occupancy family; no reactive regrouping or excited-state map"),
        }
    if hamiltonian == MECHANICAL:
        quantum_term = ("periodic_ground" if config["qm"].get("boundary") == "periodic-same-cell"
                        else "supercell_ground" if config["provider"]["engine"] == "abacus"
                        else "isolated_selected" if selected else "isolated_ground")
        terms = [("low_full", 1, "AB"), ("low_qm", -1, "A"),
                 (quantum_term, 1, "A")]
        baseline = "low_full"
    elif hamiltonian == GROUND_FRAGMENT_HAMILTONIAN:
        terms = [("low_full", 1, "AB"), ("low_qm", -1, "A"),
                 ("isolated_ground", 1, "A"), ("fragment_pair_deltas", 1, "declared"),
                 ("declared_environment_one_body", 1, "declared")]
        baseline = "zero"
    elif hamiltonian == PERIODIC_EE:
        terms = [("low_full", 1, "AB"), ("low_qm", -1, "A"),
                 ("periodic_embedded_ground", 1, "AB"), ("periodic_gaussian_reference", -1, "AB")]
        baseline = "zero"
    elif hamiltonian == REFERENCE:
        terms = [("low_full", 1, "AB"), ("low_qm", -1, "A"),
                 ("embedded_ground", 1, "AB"), ("reference_cross_coulomb", -1, "AB")]
        baseline = "zero"
    elif hamiltonian == "finite-electrostatic-additive":
        terms = [("low_environment", 1, "B"),
                 ("embedded_selected" if selected else "embedded_ground", 1, "AB"),
                 ("cross_lj", 1, "AB")]
        baseline = "zero"
    else:
        terms = [("low_full", 1, "AB"), ("low_qm", -1, "A"),
                 ("isolated_ground", 1, "A"), ("embedded_gap", 1, "AB")]
        baseline = "zero"
        if hamiltonian == CORRECTED_GAP_HAMILTONIAN:
            terms.append(("cross_interaction_delta", 1, "CB"))
        if hamiltonian == FRAGMENT_GAP_HAMILTONIAN:
            terms.extend([("fragment_pair_deltas", 1, "declared"),
                          ("declared_environment_one_body", 1, "declared"),
                          ("cluster_pair_gap", 1, "CD"), ("cluster_field_gap", -1, "CD")])
    regions = {"A": "qm", "B": "environment", "AB": "complete-system"}
    if hamiltonian in (FRAGMENT_GAP_HAMILTONIAN, GROUND_FRAGMENT_HAMILTONIAN):
        regions.update({"declared": "explicit pairs and optional aligned environment monomers",
                        "CD": "declared cluster core and environment fragment"})
    if hamiltonian == CORRECTED_GAP_HAMILTONIAN:
        regions["CB"] = "declared correction core and environment fragment"
    return {"total_terms": [{"name": name, "coefficient": sign, "force_coordinates": region}
                            for name, sign, region in terms],
            "native_baseline": baseline, "external_energy": "total-minus-native-baseline",
            "force_definition": "negative-coordinate-gradient-of-each-signed-energy-term",
            "regions": regions,
            "scope": "Formula contract only; evaluated components and scientific accuracy require evidence"}


def load_run_config(path):
    """Validate one declared route with its existing strict configuration loader."""
    path = Path(path).resolve()
    original = path.read_bytes()
    raw = json.loads(original, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(raw, dict):
        raise ValueError("Run configuration must be a JSON object")  # noqa: TRY004 - Configuration API.
    has_gap = "electronic_gap" in raw
    gap = raw.pop("electronic_gap", None)
    has_selector = "coupling" in raw
    has_bias = "bias" in raw
    bias = raw.pop("bias", None)
    has_sampling = "sampling" in raw
    sampling = raw.pop("sampling", None)
    has_thermostat = "thermostat" in raw
    thermostat = raw.pop("thermostat", None)
    has_execution = "execution" in raw
    execution = raw.pop("execution", None)
    raw = _normalize_selection(raw)
    # Legacy schema-1 mechanical configurations predate an explicit model ID.
    hamiltonian = raw.get("hamiltonian", MECHANICAL)
    if hamiltonian == EHRENFEST and any((has_bias, has_sampling, has_thermostat, has_execution)):
        raise ValueError("Ehrenfest currently admits plain finite NVE without sampling or execution extensions")
    if raw.get("provider", {}).get("state", {}).get("method") == "uks-hirshfeld-cdft":
        if hamiltonian != "finite-electrostatic-additive" or any(
                (has_gap, has_bias, has_sampling, has_thermostat, has_execution)):
            raise ValueError("Native cDFT currently requires plain finite additive NVE without bias extensions")
    module, loader, _, _, _ = _route(hamiltonian)
    config = (_function(module, "_validate_config")(raw, path) if has_gap or has_selector or has_bias or has_sampling or has_thermostat or has_execution
              else _function(module, loader)(path))
    if path.read_bytes() != original:
        raise ValueError("Configuration changed while selecting and validating its Hamiltonian")
    if config.get("hamiltonian") != hamiltonian:
        raise ValueError("Validated Hamiltonian differs from the declared model")
    if has_bias:
        from .bias import bind_bias

        bind_bias(config, bias)
    if has_thermostat:
        from .dynamics import bind_thermostat

        bind_thermostat(config, thermostat)
    if has_sampling:
        from .sampling import bind_sampling

        bind_sampling(config, sampling, path.parent)
    from .inference import COMPRESSED_PROFILE

    if config.get("inference_profile") == COMPRESSED_PROFILE and not has_execution:
        raise ValueError("Compressed reactive dynamics requires explicit environment-domain execution")
    if has_execution:
        from .execution import bind_execution

        bind_execution(config, execution)
    if has_gap:
        from .electronic_gap import bind

        bind(config, gap, path.parent)
    return config


def _setup_pair(lmp, config):
    if "electronic_gap" in config:
        from .additive_cli import _zero_pair

        return _zero_pair(lmp, config)
    _, _, module, setup, _ = _route(config["hamiltonian"])
    return _function(module, setup)(lmp, config)


def _make_coupling(lmp, comm, config, *, provider_resources=None):
    if "electronic_gap" in config:
        from .electronic_gap import make_coupling

        if provider_resources is not None:
            raise ValueError("Electronic gap owns its existing CPU providers")
        return make_coupling(lmp, comm, config)
    _, _, module, _, factory = _route(config["hamiltonian"])
    if provider_resources is not None:
        from .execution import uses_native_domains

        abacus = config["hamiltonian"] in (MECHANICAL, PERIODIC_EE) and config["provider"]["engine"] == "abacus"
        if not (abacus or uses_native_domains(config)):
            raise ValueError("External provider resources require an admitted owned backend")
        return _function(module, factory)(lmp, comm, config, provider_resources=provider_resources)
    return _function(module, factory)(lmp, comm, config)


def run(config_path, output, *, restart=None):
    """Reuse the same owner-read/MPI, failure and accepted-checkpoint lifecycle."""
    from .cli import _run

    return _run(
        config_path, output, restart=restart, config_loader=load_run_config,
        setup_pair=_setup_pair, make_coupling=_make_coupling,
    )


def describe_run(config):
    """Summarize validated choices without claiming numerical qualification."""
    hamiltonian = config["hamiltonian"]
    _route(hamiltonian)
    mechanical = hamiltonian == MECHANICAL
    regions = (
        {"full": config["full_region"], "qm": config["qm_region"]}
        if hamiltonian == EHRENFEST else
        {"full": config["full_region"], "groups": config["groups"],
         "reactive_region": config["reactive_region"], "partitions": config["partitions"]}
        if hamiltonian in (REACTIVE_WATER, REACTIVE_EE) else
        {"full": config["full_region"], "groups": config["groups"],
         "core_group_ids": config["core_group_ids"], "partitions": config["partitions"]}
        if hamiltonian in (MOLECULAR, MOLECULAR_EE, WATER_BUFFER, SELECTED_FAMILY) else
        {"qm": config["region"]}
        if mechanical else
        {"qm": config["qm_region"], "environment": config["environment_region"]}
    )
    description = {
        "schema_version": 1,
        "status": "configuration-valid",
        "hamiltonian": hamiltonian,
        "coupling": _describe_selection(config),
        "energy_contract": _energy_contract(config),
        "assembly": {
            "native_pair": "deepmd" if mechanical else "zero",
            "external_quantity": (
                "qm-minus-region-low-correction" if mechanical else "complete-total-energy-and-forces"
            ),
        },
        "periodic": config["periodic"],
        "regions": regions,
        "provider_contract": config["provider_contract"],
        "qm": config["qm"],
        "model": {"path": config["model"], "sha256": config["model_sha256"]},
        "data": {"path": config["data_file"], "sha256": config["data_sha256"]},
        "physics_sha256": config["physics_sha256"],
        "software": config["software"],
        "dynamics": {key: config[key] for key in ("timestep_fs", "steps", "sample_interval")},
        "qualification": "Configuration only; numerical, scientific and performance evidence is separate",
    }
    if "electronic_gap" in config:
        description["electronic_gap"] = config["electronic_gap"]
        description["assembly"] = {"native_pair": "zero", "external_quantity": "sampled-surface-plus-static-gap-restraint"}
        description["energy_contract"]["native_baseline"] = "zero"
        description["energy_contract"]["external_energy"] = "complete-sampled-total-plus-gap-restraint"
    if mechanical and config["qm"].get("boundary") == "periodic-same-cell":
        description["boundaries"] = {
            "full_low": "periodic-3d", "region_low": "periodic-same-cell",
            "qm": "periodic-same-cell", "embedding": "none",
            "cell_angstrom": config["qm"]["supercell_angstrom"],
            "stress": "unsupported",
        }
    elif mechanical and config["periodic"]:
        description["boundaries"] = {
            "full_low": "periodic-3d", "region_low": "isolated-molecule",
            "qm": "isolated-molecule", "embedding": "none",
            "coordinate_mapping": "native-unwrapped-images",
            "cell": "fixed-native-cell", "all_qm_cancellation": "not-guaranteed",
            "stress": "unsupported",
        }
    if hamiltonian == SELECTED_FAMILY:
        description["state_family"] = {
            "reference_sha256": config["family_reference"]["sha256"],
            "metric": "unsigned normalized cross-AO transition-density operator overlap",
            "electronic_propagation": False}
    if hamiltonian == EHRENFEST:
        description["electronic_dynamics"] = {
            "algorithm": "two-state-ehrenfest", "initial_amplitudes": config["initial_amplitudes"],
            "reference_sha256": config["pair_reference"]["sha256"],
            "nuclear_momentum": "not separately conserved by the uncorrected NAC convention"}
    if hamiltonian == PERIODIC_EE:
        description["embedding"] = config["qm"]["periodic_field"]
        description["reference"] = config["reference"]
        description["boundaries"] = {
            "full_low": "periodic-3d", "region_low": "periodic-same-cell",
            "qm": "periodic-same-cell", "embedding": "neutral-gaussian-3d-mean-zero",
            "cell_angstrom": config["qm"]["supercell_angstrom"], "stress": "unsupported",
            "empty_environment": "use-explicit-same-cell-mechanical-route",
        }
    if hamiltonian == REFERENCE:
        description["embedding"] = config["embedding"]
        description["reference"] = config["reference"]
    if hamiltonian in (WATER_BUFFER, REACTIVE_WATER, REACTIVE_EE):
        description["buffer_reference"] = config["buffer_reference"]
        description["water_guard"] = config["water_guard"]
    if hamiltonian in (MOLECULAR_EE, REACTIVE_EE):
        description["embedding"] = config["embedding"]
        description["cross"] = config["cross"]
    if "interaction_delta" in config:
        description["interaction_delta"] = config["interaction_delta"]
    if "fragment_interactions" in config:
        description["fragment_interactions"] = config["fragment_interactions"]
    if "fragment_response" in config:
        description["fragment_response"] = config["fragment_response"]
        description["selected_state_terms"] = ["anchor", "cluster_pair", "cluster_field"]
    if "bias" in config:
        from .bias import describe_bias

        description["bias"] = describe_bias(config["bias"])
        description["unbiased_physics_sha256"] = config["unbiased_physics_sha256"]
        description["energy_contract"]["total_terms"].append({
            "name": "static_distance_bias", "coefficient": 1,
            "force_coordinates": "both restrained atoms",
        })
        description["assembly"]["additional_native_bias"] = "spring couple"
    if "sampling" in config:
        description["sampling"] = config["sampling"]
        description["assembly"]["additional_native_bias"] = "plumed"
        description["energy_contract"]["total_terms"].append({
            "name": "plumed_bias", "coefficient": 1,
            "force_coordinates": "declared CV atoms",
        })
    if "thermostat" in config:
        description["dynamics"]["thermostat"] = config["thermostat"]
    if "execution" in config:
        description["execution"] = config["execution"]
        from .execution import uses_native_domains

        if uses_native_domains(config):
            description["assembly"]["environment_pair"] = config["execution"]["pair_style"]
        else:
            description["assembly"]["native_pair"] = config["execution"]["pair_style"]
    return description
