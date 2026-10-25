# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Optional fragment configuration and all-state acceptance, without real engines."""

import builtins
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_dispatch import save

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import NativeDriver, config, load
from test_pbe0_gap_native import converted
from test_pbe0_provider_state import pbe0_diagnostics, request
from test_state_records import Backend, evaluate, frame

from qmdpa import dispatch, gap_anchor_cli
from qmdpa.core import EvaluationError, Geometry
from qmdpa.excited import select_tda_state
from qmdpa.fragment_response import ClusterGapDelta, OneBodyAlignedPairSet
from qmdpa.fragment_response_config import FRAGMENT_GAP_HAMILTONIAN, FRAGMENT_MODEL
from qmdpa.interaction_delta import CorrectedCalculator
from qmdpa.interaction_delta_config import INTERACTION_MODEL
from qmdpa.pbe0_tda import PBE0TDAReference
from qmdpa.runtime import _attach_state, _compare_restart, _validate_restart
from qmdpa.state_records import (
    RecordingTDAProvider,
    StateRecordGroup,
    configured_auxiliary_providers,
    validate_frame_state,
)

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def select(config):
    anchor = converted(config)
    geometry = Geometry(
        anchor.qm_geometry.atom_ids + anchor.environment_geometry.atom_ids,
        anchor.qm_geometry.symbols + anchor.environment_geometry.symbols,
        np.vstack((anchor.qm_geometry.positions, anchor.environment_geometry.positions)),
    )
    pair = PBE0TDAReference(
        geometry,
        anchor.settings,
        anchor.transition_density,
        anchor.seed_root,
        anchor.excitation_energy_ev,
        anchor.reference_gap_ev,
        "Synthetic pair identity only",
        anchor.runtime_json,
    )
    field = replace(anchor, label="Synthetic auxiliary field identity only")
    config["hamiltonian"] = FRAGMENT_GAP_HAMILTONIAN
    config["fragment_response"] = {
        "model": FRAGMENT_MODEL,
        "pairs": [
            {
                "model": INTERACTION_MODEL,
                "ground_method": "matched-pbe0",
                "core_region": copy.deepcopy(config["qm_region"]),
                "fragment_region": copy.deepcopy(config["environment_region"]),
                "charge_spin": {name: [0, 1] for name in ("pair", "core", "fragment")},
            }
        ],
        "alignments": [{"pair_index": 0, "side": "fragment"}],
        "cluster_gap": {
            "model": "finite-cluster-minus-field-gap-v1",
            "core_region": copy.deepcopy(config["qm_region"]),
            "fragment_region": copy.deepcopy(config["environment_region"]),
            "pair_provider": request(pair),
            "field_provider": request(field),
            "state_correspondence": {
                "source": "Synthetic declaration; no scientific claim",
                "anchor_sha256": anchor.sha256,
                "pair_sha256": pair.sha256,
                "field_sha256": field.sha256,
            },
        },
    }
    return anchor, pair, field


def observers(references):
    result = []
    for ref in references:
        backend = Backend.__new__(Backend)
        backend.reference, backend.calls, backend.failure = ref, [], False
        backend.diagnostic = pbe0_diagnostics(ref)
        result.append(RecordingTDAProvider(backend))
    return result


def group_fixture(config):
    refs = select(config)
    primary, pair, field = observers(refs)
    group = StateRecordGroup(primary, {"cluster_pair": pair, "cluster_field": field})
    for item in (primary, pair, field):
        evaluate(item)
    return refs, group


def attach(config, refs, group):
    result = frame(refs[0])
    _attach_state(
        result,
        SimpleNamespace(failure=None, last_step=result["step"], state_observer=group),
        config,
    )
    return result


def test_explicit_identity_selection_and_total_contract(tmp_path, config):
    select(config)
    normalized = load(tmp_path, config)
    assert normalized["assembly"]["pair_style"] == "zero"
    assert normalized["assembly"]["selected_state_terms"] == [
        "anchor",
        "cluster_pair",
        "cluster_field",
    ]
    selected = copy.deepcopy(config)
    selected.pop("hamiltonian")
    selected["coupling"] = {
        "surface": "selected",
        "embedding": "electrostatic",
        "ee_model": "gap-anchor-fragment-response",
    }
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        assert dispatch.load_run_config(save(tmp_path, selected)) == normalized
    description = dispatch.describe_run(normalized)
    assert description["fragment_response"] == normalized["fragment_response"]
    assert [(v["name"], v["coefficient"]) for v in description["energy_contract"]["total_terms"]][
        -2:
    ] == [("cluster_pair_gap", 1), ("cluster_field_gap", -1)]
    original_hash = normalized["physics_sha256"]
    config["steps"] = 0
    assert load(tmp_path, config)["physics_sha256"] == original_hash
    config["fragment_response"]["alignments"] = []
    assert load(tmp_path, config)["physics_sha256"] != original_hash


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "unannounced",
        "both",
        "model",
        "empty-pairs",
        "double-pair",
        "double-one-body",
        "qm-one-body",
        "bool-index",
        "outside",
        "split-molecule",
        "pair-state",
        "field-state",
        "correspondence",
        "extra-field",
    ],
)
def test_ambiguous_or_unmatched_model_rejects_before_backend(tmp_path, config, change):
    select(config)
    response = config["fragment_response"]
    gap = response["cluster_gap"]
    if change == "missing":
        config.pop("fragment_response")
    elif change == "unannounced":
        config["hamiltonian"] = "finite-electrostatic-gap-anchor"
    elif change == "both":
        config["interaction_delta"] = copy.deepcopy(response["pairs"][0])
    elif change == "model":
        response["model"] = "implicit-polarization"
    elif change == "empty-pairs":
        response["pairs"] = []
    elif change == "double-pair":
        response["pairs"] *= 2
    elif change == "double-one-body":
        response["alignments"] *= 2
    elif change == "qm-one-body":
        response["alignments"][0]["side"] = "core"
    elif change == "bool-index":
        response["alignments"][0]["pair_index"] = False
    elif change == "outside":
        gap["core_region"] = copy.deepcopy(config["environment_region"])
    elif change == "split-molecule":
        gap["core_region"]["molecules"] = [[i] for i in gap["core_region"]["atom_ids"]]
    elif change == "pair-state":
        gap["pair_provider"] = copy.deepcopy(gap["field_provider"])
    elif change == "field-state":
        gap["field_provider"] = copy.deepcopy(gap["pair_provider"])
    elif change == "correspondence":
        gap["state_correspondence"]["pair_sha256"] = "0" * 64
    else:
        response["fitted_offset"] = 1.0
    with patch("qmdpa.backends.DeepMDBackend") as low, pytest.raises(ValueError):
        load(tmp_path, config)
    low.assert_not_called()


def test_provider_settings_and_capacity_are_explicit(tmp_path, config):
    _, pair, _ = select(config)
    gap = config["fragment_response"]["cluster_gap"]

    def change(pinned):
        gap["pair_provider"] = request(pinned)
        gap["state_correspondence"]["pair_sha256"] = pinned.sha256

    capacity = replace(pair, settings=replace(pair.settings, tda_dense_max_dimension=1024))
    change(capacity)
    load(tmp_path, config)  # Region-specific solver capacity is not a method change.
    change(replace(pair, settings=replace(pair.settings, conv_tol=5e-13)))
    with pytest.raises(ValueError, match="settings"):
        load(tmp_path, config)


def test_configuration_does_not_import_quantum_engine(tmp_path, config):
    select(config)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf")):
            raise AssertionError("Unexpected engine import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        load(tmp_path, config)


def test_native_factory_uses_existing_components_and_all_recorders(config):
    anchor, pair, field = select(config)
    baseline = SimpleNamespace(
        embedded_selected_backend=SimpleNamespace(reference=anchor, charges=anchor.charges),
        evaluate=lambda g: None,
    )
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.gap_anchor.make_pbe0_gap_anchor", return_value=baseline),
        patch("qmdpa.pbe0_tda.PySCFPBE0GroundBackend"),
        patch(
            "qmdpa.pbe0_tda.PySCFPBE0ElectrostaticGroundBackend",
            return_value=SimpleNamespace(charges=field.charges, evaluate=lambda *a: None),
        ),
        patch("qmdpa.pbe0_tda.PySCFPBE0TDABackend", return_value=SimpleNamespace(reference=pair)),
        patch(
            "qmdpa.pbe0_tda.PySCFPBE0TDAElectrostaticBackend",
            return_value=SimpleNamespace(reference=field, charges=field.charges),
        ),
    ):
        bridge = gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    assert isinstance(bridge.calculator, CorrectedCalculator)
    assert isinstance(bridge.calculator.correction, ClusterGapDelta)
    assert isinstance(bridge.calculator.baseline.correction, OneBodyAlignedPairSet)
    assert bridge.calculator.baseline.baseline is baseline
    assert bridge.state_observer.primary is baseline.embedded_selected_backend
    assert (
        bridge.state_observer.auxiliary["cluster_pair"]
        is bridge.calculator.correction.pair_selected
    )
    assert (
        bridge.state_observer.auxiliary["cluster_field"]
        is bridge.calculator.correction.embedded_selected
    )
    assert low.call_count == 2


def test_three_records_are_detached_same_evaluation_and_report_without_solves(config):
    refs, group = group_fixture(config)
    snapshot = attach(config, refs, group)
    validate_frame_state(snapshot, config)
    assert snapshot["electronic_state"] == group.primary.accepted_record(frame(refs[0]))
    records = snapshot["auxiliary_electronic_states"]
    assert records["cluster_pair"]["reference_sha256"] == refs[1].sha256
    assert records["cluster_field"]["reference_sha256"] == refs[2].sha256
    records["cluster_pair"]["diagnostics"]["ground_energy_ev"] = 999.0
    validate_frame_state(attach(config, refs, group), config)
    assert [len(p.backend.calls) for p in (group.primary, *group.auxiliary.values())] == [1, 1, 1]


@pytest.mark.parametrize("role", ["cluster_pair", "cluster_field"])
@pytest.mark.parametrize("damage", ["missing", "geometry", "reference", "root", "assembly"])
def test_each_auxiliary_record_is_required_before_restart_binary_io(tmp_path, config, role, damage):
    refs, group = group_fixture(config)
    snapshot = attach(config, refs, group)
    records = snapshot["auxiliary_electronic_states"]
    record = records[role]
    if damage == "missing":
        del records[role]
    elif damage == "geometry":
        record["geometry_sha256"] = "0" * 64
    elif damage == "reference":
        record["reference_sha256"] = "0" * 64
    elif damage == "root":
        record["selected_root"] = 2
    else:
        record["diagnostics"]["tda_matrix_assembly"] = "foreign"
    config["physics_sha256"] = "test"
    checkpoint = {
        "physics_sha256": "test",
        "frame": snapshot,
        "restart_file": "missing.bin",
        "restart_sha256": "0" * 64,
    }
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):
        _validate_restart(path, config)


@pytest.mark.parametrize("role", ["cluster_pair", "cluster_field"])
def test_auxiliary_reselection_cannot_hide_in_unchanged_total(config, role):
    refs, group = group_fixture(config)
    original = attach(config, refs, group)
    provider = group.auxiliary[role]
    _, chosen = select_tda_state([0.0, 1.0, 0.0], [2.0, 3.0, 4.0], provider.reference.settings)
    provider.backend.diagnostic.update(chosen, excitation_energy_ev=3.0)
    evaluate(provider)
    changed = attach(config, refs, group)
    assert changed["potential_eV"] == original["potential_eV"]
    with pytest.raises(ValueError, match="different auxiliary"):
        _compare_restart(changed, {"frame": original}, config)


def test_auxiliary_provider_energy_cannot_compensate_in_total(config):
    refs, group = group_fixture(config)
    old = attach(config, refs, group)
    new = copy.deepcopy(old)
    record = new["auxiliary_electronic_states"]["cluster_pair"]
    record["provider_energy_ev"] += 0.1
    record["diagnostics"]["ground_energy_ev"] += 0.1
    with pytest.raises(ValueError, match="auxiliary provider energy"):
        _compare_restart(new, {"frame": old}, config)


@pytest.mark.parametrize("role", ["cluster_pair", "cluster_field"])
def test_failed_auxiliary_invalidates_all_records_and_never_attaches_partial_frame(config, role):
    refs, group = group_fixture(config)
    provider = group.auxiliary[role]
    provider.backend.failure = True
    with pytest.raises(EvaluationError, match="injected"):
        evaluate(provider)
    target = frame(refs[0])
    with pytest.raises(EvaluationError, match="No successful"):
        _attach_state(
            target,
            SimpleNamespace(failure=None, last_step=target["step"], state_observer=group),
            config,
        )
    assert "electronic_state" not in target and "auxiliary_electronic_states" not in target
    for item in (group.primary, *group.auxiliary.values()):
        with pytest.raises(EvaluationError, match="No successful"):
            item.accepted_record(target)


def test_group_clear_and_legacy_frame_boundary(config):
    refs, group = group_fixture(config)
    snapshot = attach(config, refs, group)
    legacy = copy.deepcopy(config)
    del legacy["fragment_response"]
    legacy["hamiltonian"] = "finite-electrostatic-gap-anchor"
    with pytest.raises(ValueError, match="does not declare"):
        validate_frame_state(snapshot, legacy)
    assert configured_auxiliary_providers(legacy) == {}
    group.clear()
    with pytest.raises(EvaluationError, match="No successful"):
        attach(config, refs, group)
