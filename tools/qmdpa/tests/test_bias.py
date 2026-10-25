# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Bias configuration, callback safety and accepted-state identity contracts."""

from copy import deepcopy

import numpy as np
import pytest
from test_cli import accepted_frame
from test_dispatch import additive, mechanical, save
from test_gap_anchor_native import config as gap_config
from test_lammps_bridge import ConstantCorrection, Driver

from qmdpa import dispatch
from qmdpa.bias import (
    NativeDistanceBias,
    bind_bias,
    compare_bias_restart,
    normalize_bias,
    validate_bias_frame,
)
from qmdpa.core import EvaluationError
from qmdpa.lammps import ExternalCoupling
from qmdpa.region import FixedRegion
from qmdpa.runtime import _compare_restart, _validate_restart

__all__ = ["gap_config"]


def restraint(ids=(1, 2), k=0.2, target=0.5):
    return {"form": "harmonic-distance", "restraints": [
        {"atom_ids": list(ids), "k_ev_per_angstrom2": k, "distance_angstrom": target}
    ]}


@pytest.mark.parametrize("kind", ["mechanical", "additive", "gap"])
def test_unified_loader_binds_bias_without_changing_coupling(
    tmp_path, gap_config, kind, monkeypatch
):
    # These unevaluated configurations have synthetic dependency identities.
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-version")
    raw = {"mechanical": mechanical, "additive": additive}.get(kind)
    raw = raw(tmp_path) if raw is not None else gap_config
    before = dispatch.load_run_config(save(tmp_path, raw))
    raw["bias"] = restraint()
    after = dispatch.load_run_config(save(tmp_path, raw))
    assert after["unbiased_physics_sha256"] == before["physics_sha256"]
    assert after["physics_sha256"] != before["physics_sha256"]
    assert {k: v for k, v in after.items() if k not in
            {"bias", "unbiased_physics_sha256", "physics_sha256"}} == {
                k: v for k, v in before.items() if k != "physics_sha256"
            }
    description = dispatch.describe_run(after)
    assert description["energy_contract"]["total_terms"][-1]["name"] == "static_distance_bias"
    assert description["assembly"]["native_pair"] == ("deepmd" if kind == "mechanical" else "zero")


@pytest.mark.parametrize("change", [
    ("k_ev_per_angstrom2", 0.3), ("distance_angstrom", 0.6), ("atom_ids", [1, 3]),
])
def test_parameters_change_physics_identity(tmp_path, change):
    raw = mechanical(tmp_path)
    raw["bias"] = restraint()
    before = dispatch.load_run_config(save(tmp_path, raw))
    raw["bias"]["restraints"][0][change[0]] = change[1]
    assert dispatch.load_run_config(save(tmp_path, raw))["physics_sha256"] != before["physics_sha256"]


def test_pair_orientation_and_budget_do_not_change_physics(tmp_path):
    raw = mechanical(tmp_path)
    raw["bias"] = restraint()
    before = dispatch.load_run_config(save(tmp_path, raw))
    raw["bias"]["restraints"][0]["atom_ids"].reverse()
    raw.update(steps=10, sample_interval=3)
    assert dispatch.load_run_config(save(tmp_path, raw))["physics_sha256"] == before["physics_sha256"]


@pytest.mark.parametrize("key,value", [
    ("atom_ids", [1, 1]), ("atom_ids", [True, 2]), ("atom_ids", [0, 2]),
    ("atom_ids", [1, 2, 3]), ("atom_ids", [1.0, 2]), ("atom_ids", "1 2"),
    ("k_ev_per_angstrom2", True), ("k_ev_per_angstrom2", 0), ("k_ev_per_angstrom2", -1),
    ("k_ev_per_angstrom2", float("nan")), ("k_ev_per_angstrom2", "0.2"),
    ("distance_angstrom", 0), ("distance_angstrom", float("inf")),
    ("distance_angstrom", "1; clear"), ("extra", 1),
])
def test_invalid_parameters_are_rejected(key, value):
    raw = restraint()
    raw["restraints"][0][key] = value
    with pytest.raises(ValueError):
        normalize_bias(raw, {"hamiltonian": dispatch.MECHANICAL, "periodic": False})


@pytest.mark.parametrize("raw", [None, False, [], {}, {"form": "other", "restraints": []},
                                {"form": "harmonic-distance", "restraints": []}])
def test_malformed_bias_is_not_treated_as_disabled(raw):
    with pytest.raises(ValueError):
        normalize_bias(raw, {"hamiltonian": dispatch.MECHANICAL, "periodic": False})


def test_duplicate_pairs_and_group_overflow_fail():
    raw = restraint()
    raw["restraints"].append({**raw["restraints"][0], "atom_ids": [2, 1]})
    config = {"hamiltonian": dispatch.MECHANICAL, "periodic": False}
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_bias(raw, config)
    raw["restraints"] = [restraint((2*i+1, 2*i+2))["restraints"][0] for i in range(16)]
    with pytest.raises(ValueError, match="31"):
        normalize_bias(raw, config)


@pytest.mark.parametrize("hamiltonian,periodic", [
    (dispatch.MECHANICAL, True), (dispatch.MOLECULAR, False), (dispatch.MOLECULAR_EE, False),
    ("finite-electrostatic-gap-anchor-cross-delta", False),
])
def test_unsupported_composition_rejected(hamiltonian, periodic):
    with pytest.raises(ValueError, match="finite fixed-region"):
        normalize_bias(restraint(), {"hamiltonian": hamiltonian, "periodic": periodic})


class NativeDriver(Driver):
    def __init__(self):
        super().__init__()
        self.box = (*self.box[:5], [0, 0, 0], 0)
        self.positions[:] = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
        self.groups = {"qmdpa_bias_atom_5"}
        self.extracted = []

    def has_style(self, category, name):
        return category == "fix" and name == "spring"

    def has_id(self, category, name):
        return category == "group" and name in self.groups

    def extract_fix(self, fix, style, kind, axis=0):
        self.extracted.append((fix, style, kind, axis))
        return 0.025 if kind == 0 else [0.1, 0, 0][axis]


def config(ids=(5, 99)):
    return {"bias": restraint(ids), "hamiltonian": dispatch.MECHANICAL, "periodic": False,
            "physics_sha256": "test"}


def test_native_setup_records_energy_and_recreates_groups():
    driver = NativeDriver()
    bias = NativeDistanceBias(driver, config())
    assert bias.fix_ids == ["qmdpa_bias_0"]
    assert driver.commands[0] == "group qmdpa_bias_atom_5 clear"
    assert "spring couple qmdpa_bias_atom_99" in driver.commands[-2]
    assert driver.commands[-1] == "fix_modify qmdpa_bias_0 energy yes"


def test_environment_atom_guard_and_undefined_later_step_use_failure_protocol():
    driver = NativeDriver()
    coupling = ExternalCoupling(driver, FixedRegion((22, 5), ("O", "H")), ConstantCorrection())
    bias = NativeDistanceBias(driver, config())
    coupling.native_bias = bias
    coupling.run(0)
    assert coupling.statistics()["calculator_attempts"] == 1
    driver.positions[1] = driver.positions[0]
    with pytest.raises(EvaluationError, match="Bias distance"):
        coupling.run(1)
    assert coupling.statistics()["calculator_attempts"] == 1
    assert driver.timeout and not driver.swallowed


@pytest.mark.parametrize("change", ["missing", "nan", "images"])
def test_bias_guard_checks_all_owners_before_quantum(change):
    driver = NativeDriver()
    coupling = ExternalCoupling(driver, FixedRegion((22, 5), ("O", "H")), ConstantCorrection())
    coupling.native_bias = NativeDistanceBias(driver, config((5, 999) if change == "missing" else (5, 99)))
    if change == "nan":
        driver.positions[1, 0] = np.nan
    if change == "images":
        driver.decode_image_flags = lambda value: [1, 0, 0]
    with pytest.raises(EvaluationError):
        coupling.run(0)
    assert coupling.statistics()["calculator_attempts"] == 0
    assert driver.timeout and not driver.swallowed


def test_bias_guard_handles_split_ownership_and_empty_rank():
    bias = NativeDistanceBias(NativeDriver(), config())
    bias.validate_packets([[], [(5, [0, 0, 0])], [(99, [2, 0, 0])]])
    with pytest.raises(ValueError, match="multiple owners"):
        bias.validate_packets([[(5, [0, 0, 0]), (5, [0, 0, 0]), (99, [2, 0, 0])]])


def biased_frame():
    result = accepted_frame()
    result["potential_eV"] += 0.025
    NativeDistanceBias(NativeDriver(), config((1, 2))).capture(result, None)
    return result


def test_native_scalar_and_vector_are_observed_and_unbiased_energy_preserved():
    frame = biased_frame()
    assert frame["bias"]["energy_eV"] == 0.025
    assert frame["bias"]["unbiased_potential_eV"] == -10.0
    assert frame["bias"]["restraints"][0]["force_on_first_eV_per_angstrom"] == [0.1, 0, 0]
    _compare_restart(frame, {"frame": deepcopy(frame)}, config((1, 2)))


@pytest.mark.parametrize("change", ["missing", "energy", "unbiased", "id", "distance", "force"])
def test_invalid_bias_record_cannot_be_accepted(change):
    frame = biased_frame()
    if change == "missing":
        frame.pop("bias")
    elif change in ("energy", "unbiased"):
        frame["bias"][{"energy": "energy_eV", "unbiased": "unbiased_potential_eV"}[change]] += 1
    elif change == "id":
        frame["bias"]["restraints"][0]["atom_ids"] = [1, 3]
    elif change == "distance":
        frame["bias"]["restraints"][0]["distance_angstrom"] += 0.1
    else:
        frame["bias"]["restraints"][0]["force_on_first_eV_per_angstrom"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_bias_frame(frame, config((1, 2)))


def test_no_bias_config_rejects_biased_frame_and_recomputed_force_changes():
    reference = biased_frame()
    with pytest.raises(ValueError, match="unbiased"):
        validate_bias_frame(reference, {})
    frame = deepcopy(reference)
    frame["bias"]["restraints"][0]["force_on_first_eV_per_angstrom"][0] += 1e-4
    with pytest.raises(ValueError, match="Restart bias force"):
        compare_bias_restart(frame, reference, config((1, 2)))


def test_restart_preflight_requires_bias_record_and_same_physics(tmp_path):
    import hashlib
    import json

    cfg = config((1, 2))
    bind_bias(cfg, cfg["bias"])
    binary = tmp_path / "restart.bin"
    binary.write_bytes(b"test restart only")
    checkpoint = {"physics_sha256": cfg["physics_sha256"], "frame": biased_frame(),
                  "restart_file": binary.name,
                  "restart_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    assert _validate_restart(path, cfg)["frame"] == checkpoint["frame"]
    checkpoint["frame"].pop("bias")
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="bias record"):
        _validate_restart(path, cfg)
