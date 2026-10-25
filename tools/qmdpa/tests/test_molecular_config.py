# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Strict molecular configuration and existing collective native lifecycle."""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest
from test_adaptive import PairPolynomial
from test_adaptive_cli import providers, torch_state
from test_adaptive_config import document as legacy
from test_adaptive_config import load
from test_additive_lammps_bridge import AdditiveDriver, AdditiveWorkerComm
from test_molecular_adaptive import geometry, groups

from qmdpa import dispatch
from qmdpa.adaptive_config import build_mixer
from qmdpa.core import EvaluationError
from qmdpa.inference import RAW_ENV


def document():
    data = legacy()
    data.update(
        hamiltonian="finite-molecular-mechanical", type_symbols=["H"],
        core_group_ids=["core"],
        full_region={"atom_ids": list(range(1, 9)), "symbols": ["H"] * 8,
                     "molecules": [list(g.atom_ids) for g in groups()]},
        groups=[{"group_id": g.group_id, "atom_ids": list(g.atom_ids), "symbols": list(g.symbols)}
                for g in groups()],
        switch={"method": "quintic-molecular-occupancy", "center_atom_id": 1,
                "candidate_anchors": [["c", 7], ["a", 3], ["b", 5]], "occupancy": 1,
                "inner_radius_angstrom": 2., "outer_radius_angstrom": 3.},
    )
    base = data["partitions"][0]
    data["partitions"] = [dict(base, partition_id=name, qm_group_ids=["core", name])
                          for name in ("c", "a", "b")]
    return data


def test_unified_config_describes_total_native_assembly_and_canonical_family(tmp_path, monkeypatch):
    # Keep the direct and dispatched loads on the same synthetic identity.
    monkeypatch.setattr("importlib.metadata.version", lambda name: "fixture")
    cfg = load(tmp_path, document())
    path = tmp_path / "config.json"
    via_unified = dispatch.load_run_config(path)
    assert via_unified["switch"] == cfg["switch"]
    assert [p["partition_id"] for p in cfg["partitions"]] == ["a", "b", "c"]
    assert cfg["switch"]["candidate_anchors"] == [["a", 3], ["b", 5], ["c", 7]]
    assert cfg["switch"]["max_partitions"] == 64
    description = dispatch.describe_run(via_unified)
    assert description["assembly"] == {"native_pair": "zero",
                                        "external_quantity": "complete-total-energy-and-forces"}
    assert description["coupling"] == {"surface": "ground", "embedding": "mechanical",
                                       "me_model": "molecular"}
    assert description["energy_contract"]["native_baseline"] == "zero"
    assert description["regions"]["core_group_ids"] == ["core"]
    raw = document()
    del raw["hamiltonian"]
    raw["coupling"] = {"surface": "ground", "embedding": "mechanical", "me_model": "molecular"}
    path.write_text(json.dumps(raw))
    selected = dispatch.load_run_config(path)
    assert selected["physics_sha256"] == via_unified["physics_sha256"]


def test_family_order_is_canonical_but_switch_and_offset_change_restart_identity(tmp_path):
    original = load(tmp_path, document())["physics_sha256"]
    reordered = document()
    for key in ("partitions", "groups", "core_group_ids"):
        reordered[key].reverse()
    reordered["switch"]["candidate_anchors"].reverse()
    assert load(tmp_path, reordered)["physics_sha256"] == original
    for field in ("outer_radius_angstrom", "inner_radius_angstrom"):
        changed = document()
        changed["switch"][field] += .1
        assert load(tmp_path, changed)["physics_sha256"] != original
    changed = document()
    changed["partitions"][0].update(energy_offset_ev=.1, offset_source="declared alternative")
    assert load(tmp_path, changed)["physics_sha256"] != original


@pytest.mark.parametrize("bad", ["family", "anchor", "core", "count", "field", "selected", "selector"])
def test_invalid_molecular_settings_never_load_backends(tmp_path, bad):
    value = document()
    if bad == "family":
        value["partitions"].pop()
    elif bad == "anchor":
        value["switch"]["candidate_anchors"][0][1] = 5
    elif bad == "core":
        del value["core_group_ids"]
    elif bad == "count":
        value["switch"]["occupancy"] = 2
    elif bad == "field":
        value["switch"]["threshold"] = .1
    elif bad == "selected":
        value["provider"]["state"]["method"] = "rhf-tda"
    else:
        value["coupling"] = {"surface": "selected", "embedding": "mechanical", "me_model": "molecular"}
    with patch("qmdpa.backends.DeepMDBackend") as backend:
        if bad == "selector":
            load(tmp_path, document())
            (tmp_path / "config.json").write_text(json.dumps(value))
            with pytest.raises(ValueError):
                dispatch.load_run_config(tmp_path / "config.json")
        else:
            with pytest.raises((TypeError, ValueError)):
                load(tmp_path, value)
    backend.assert_not_called()


class MolecularDriver(AdditiveDriver):
    def __init__(self, world_size=1):
        ids = (8, 3, 5, 1, 4, 2, 7, 6)
        super().__init__(tags=ids, world_size=world_size)
        self.natoms, self.ntypes = 8, 1
        self.types[:] = 1
        self.positions[:] = geometry().select(ids).positions


def test_native_factory_builds_family_once_and_publishes_complete_conservative_total(tmp_path):
    config = load(tmp_path, document())
    lmp, torch = MolecularDriver(), torch_state()
    lows, highs, low, high = providers(config, torch)
    with patch.dict(os.environ, RAW_ENV), \
            patch("qmdpa.inference.prepare_raw_inference", return_value=torch), \
            patch("qmdpa.backends.DeepMDBackend", side_effect=low), \
            patch("qmdpa.providers.make_qm_provider", side_effect=high):
        coupling = dispatch._make_coupling(lmp, None, config)
        assert len(lows) == 1 and len(highs) == 3
        energy = coupling.run(0)
        expected = build_mixer(config, PairPolynomial(.04, -10),
            [PairPolynomial(.07, -5) for _ in range(3)]).evaluate(geometry())
        assert energy == pytest.approx(expected.energy, abs=1e-12)
        np.testing.assert_allclose(lmp.forces,
            expected.forces[[geometry().atom_ids.index(a) for a in lmp.tags]], atol=1e-12, rtol=0)
        assert lows[0].calls == 4 and all(p.calls == 1 for p in highs)
        lmp.positions[lmp.tags > 2] *= 3
        with pytest.raises(EvaluationError, match="Too few"):
            coupling.run(1)
        assert coupling.last_step == 0 and lmp.energy_calls[-1][1] == 0
        np.testing.assert_array_equal(lmp.forces, np.zeros((8, 3)))


def test_worker_rank_allocates_no_model_or_quantum_providers(tmp_path):
    config = load(tmp_path, document())
    comm = AdditiveWorkerComm([(None, None), (None, None)])
    with patch("qmdpa.backends.DeepMDBackend") as low, \
            patch("qmdpa.providers.make_qm_provider") as high:
        coupling = dispatch._make_coupling(MolecularDriver(2), comm, config)
    low.assert_not_called()
    high.assert_not_called()
    assert coupling.calculator is None


def test_old_two_partition_input_does_not_accept_new_policy_fields(tmp_path):
    value = legacy()
    value["core_group_ids"] = ["core"]
    with pytest.raises(ValueError, match="unsupported"):
        load(tmp_path, value)
