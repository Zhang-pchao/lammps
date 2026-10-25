# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Distinct native configuration/transport boundaries; no real MD claims."""

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_additive_lammps_bridge import AdditiveDriver, TotalCalculator
from test_excited_config import provider_request, reference, region

from qmdpa import gap_anchor_cli
from qmdpa.core import EvaluationError
from qmdpa.gap_anchor_config import load_gap_anchor_config
from qmdpa.lammps import GapAnchorExternalCoupling
from qmdpa.region import FixedRegion


@pytest.fixture
def config(tmp_path):
    pinned = reference(embedded=True)
    (tmp_path / "model.pt2").write_bytes(b"test identity only")
    (tmp_path / "atoms.data").write_text("test data identity only")
    return {
        "schema_version": 1,
        "hamiltonian": "finite-electrostatic-gap-anchor",
        "data_file": "atoms.data",
        "model": "model.pt2",
        "model_sha256": hashlib.sha256(b"test identity only").hexdigest(),
        "type_symbols": ["H", "O"],
        "periodic": False,
        "qm_region": region(pinned.qm_geometry),
        "environment_region": region(pinned.environment_geometry),
        "embedding": pinned.to_record()["charges"],
        "full_charge_spin": [0, 1],
        "region_charge_spin": [0, 1],
        "qm": asdict(pinned.settings),
        "provider": {**provider_request(pinned), "reuse_identical_geometry": True},
        "timestep_fs": 0.1,
        "steps": 20,
    }


def load(tmp_path, config):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        return load_gap_anchor_config(p)


def test_resolved_assembly_and_budget_independent_identity(tmp_path, config):
    first = load(tmp_path, config)
    assert first["assembly"]["pair_style"] == "zero"
    assert first["assembly"]["external_quantity"] == "total-energy-and-all-forces"
    assert first["provider_contract"]["force_coordinates"] == ["qm", "environment"]
    config.update(steps=10, sample_interval=2)
    assert load(tmp_path, config)["physics_sha256"] == first["physics_sha256"]
    config["timestep_fs"] *= 0.5
    assert load(tmp_path, config)["physics_sha256"] != first["physics_sha256"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("hamiltonian", "finite-electrostatic-additive"),
        ("periodic", True),
        ("schema_version", True),
        ("full_charge_spin", [0, 3]),
        ("region_charge_spin", [False, 1]),
        ("steps", -1),
        ("sample_interval", 0),
        ("timestep_fs", float("nan")),
        ("cross", {}),
        ("temperature_K", 300),
        ("velocity_seed", 42),
        ("inference_profile", "raw"),
    ],
)
def test_unsupported_or_ambiguous_inputs_fail(tmp_path, config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        load(tmp_path, config)


@pytest.mark.parametrize(
    "change",
    ["model", "data", "field_source", "charges", "order", "symbols", "qm_settings", "reference"],
)
def test_source_state_and_field_identity_are_bound(tmp_path, config, change):
    first = load(tmp_path, config)
    if change == "model":
        (tmp_path / "model.pt2").write_bytes(b"changed")
    elif change == "data":
        (tmp_path / "atoms.data").write_text("changed")
        assert load(tmp_path, config)["physics_sha256"] != first["physics_sha256"]
        return
    elif change == "field_source":
        config["embedding"]["source"] = "another field"
    elif change == "charges":
        config["embedding"]["charges_e"][0] += 0.1
    elif change == "order":
        config["embedding"]["atom_ids"].reverse()
    elif change == "symbols":
        config["environment_region"]["symbols"][0] = "H"
    elif change == "qm_settings":
        config["qm"]["basis"] = "sto-3g"
    else:
        config["provider"]["state"]["reference"]["transition_density"][0][1] = 0.0
    with pytest.raises((ValueError, EvaluationError)):
        load(tmp_path, config)


def test_field_and_state_change_together_changes_restart_identity(tmp_path, config):
    first = load(tmp_path, config)
    old = reference(embedded=True)
    changed = replace(old, charges=replace(old.charges, source="new declared field source"))
    config["provider"] = provider_request(changed)
    config["embedding"] = changed.to_record()["charges"]
    assert load(tmp_path, config)["physics_sha256"] != first["physics_sha256"]


def test_raw_model_and_duplicate_json_are_rejected(tmp_path, config):
    (tmp_path / "model.pt").write_bytes(b"test identity only")
    config["model"] = "model.pt"
    with pytest.raises(ValueError, match="compiled .pt2"):
        load(tmp_path, config)
    config["model"] = "model.pt2"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config).replace('"steps": 20', '"steps": 20, "steps": 10'))
    with pytest.raises(ValueError):
        load_gap_anchor_config(path)


def attach(lmp, calc=None):
    return GapAnchorExternalCoupling(
        lmp,
        FixedRegion((22, 5, 99), ("O", "H", "O")),
        TotalCalculator() if calc is None else calc,
        type_symbols=("O", "H"),
    )


def test_total_transport_has_distinct_fix_and_maps_all_ids_once():
    lmp = AdditiveDriver()
    bridge = attach(lmp)
    assert bridge.run(0) == 9.5
    assert lmp.energy_calls[-1] == ("qmdpa_gap_anchor", 9.5)
    np.testing.assert_array_equal(lmp.forces, [[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    lmp.set_owned((99, 22, 5))
    bridge.run(1)
    np.testing.assert_array_equal(lmp.forces, [[7, 8, 9], [1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize("field", ["pair_style", "bond_style", "angle_style", "kspace_style"])
def test_native_force_sources_are_rejected(field):
    lmp = AdditiveDriver()
    lmp.styles[field] = "forbidden"
    with pytest.raises(ValueError):
        attach(lmp)


def test_changed_assembly_and_failure_clear_complete_force():
    lmp = AdditiveDriver()
    bridge = attach(lmp)
    bridge.run(0)
    lmp.styles["pair_style"] = "deepmd"
    with pytest.raises(EvaluationError):
        bridge.run(1)
    assert bridge.last_step == 0 and lmp.energy_calls[-1][1] == 0
    np.testing.assert_array_equal(lmp.forces, 0)


def test_pair_setup_and_shared_accepted_checkpoint_lifecycle(tmp_path):
    lmp = SimpleNamespace(commands_list=Mock())
    gap_anchor_cli._zero_pair(lmp, {})
    lmp.commands_list.assert_called_once_with(["pair_style zero 1.0", "pair_coeff * *"])
    with patch("qmdpa.cli._run", return_value="accepted") as driver:
        assert gap_anchor_cli.run("config", "out", restart="checkpoint") == "accepted"
    assert driver.call_args.kwargs["config_loader"] is load_gap_anchor_config
    assert driver.call_args.kwargs["restart"] == "checkpoint"
    assert driver.call_args.kwargs["make_coupling"] is gap_anchor_cli._gap_anchor_coupling


class NativeDriver(AdditiveDriver):
    def __init__(self):
        super().__init__(tags=(6, 2, 4, 1, 5))
        self.natoms = 5
        self.types[:] = [1, 1, 2, 1, 1]


def test_configuration_preflight_precedes_backend_allocation(config):
    lmp = NativeDriver()
    lmp.styles["pair_style"] = "deepmd"
    with patch("qmdpa.backends.DeepMDBackend") as backend, pytest.raises(RuntimeError):
        gap_anchor_cli._gap_anchor_coupling(lmp, None, config)
    backend.assert_not_called()


def test_selected_state_observer_forwards_the_actual_fixed_field(config):
    pinned = reference(embedded=True)
    backend = SimpleNamespace(reference=pinned, charges=pinned.charges)
    calculator = SimpleNamespace(embedded_selected_backend=backend)
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.gap_anchor.make_rhf_gap_anchor", return_value=calculator) as build,
    ):
        bridge = gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    assert low.call_count == 2 and build.call_args.kwargs["reuse_identical_geometry"] is True
    assert bridge.state_observer is calculator.embedded_selected_backend
    assert bridge.state_observer.charges is backend.charges
    backend.charges = replace(pinned.charges, source="mutation")
    assert bridge.state_observer.charges.source == "mutation"
    assert bridge.state_observer.reference.sha256 == pinned.sha256
