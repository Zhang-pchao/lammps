# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Ground fragment input, limiting identity, derivatives and native assembly."""

import builtins
import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import NativeDriver, config

from qmdpa import dispatch, fragment_ground
from qmdpa.core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from qmdpa.fragment_ground import (
    GROUND_FRAGMENT_HAMILTONIAN,
    GROUND_FRAGMENT_MODEL,
    GroundFragmentCalculator,
)
from qmdpa.fragment_response import OneBodyAlignedPairSet
from qmdpa.interaction_delta import CrossInteractionDelta
from qmdpa.interaction_delta_config import INTERACTION_MODEL
from qmdpa.state_records import configured_auxiliary_providers, configured_reference

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def ground_request(config):
    request = copy.deepcopy(config)
    request.pop("embedding")
    request.pop("provider")
    request["hamiltonian"] = GROUND_FRAGMENT_HAMILTONIAN
    request["qm"] = {
        "xc": "pbe0",
        "basis": "def2-svp",
        "charge": 0,
        "multiplicity": 1,
        "conv_tol": 1e-12,
        "conv_tol_grad": 1e-8,
        "max_cycle": 100,
        "grid_level": 4,
        "grid_response": True,
        "threads": 1,
    }
    request["fragment_interactions"] = {
        "model": GROUND_FRAGMENT_MODEL,
        "pairs": [
            {
                "model": INTERACTION_MODEL,
                "ground_method": "matched-pbe0",
                "core_region": copy.deepcopy(request["qm_region"]),
                "fragment_region": copy.deepcopy(request["environment_region"]),
                "charge_spin": {k: [0, 1] for k in ("pair", "core", "fragment")},
            }
        ],
        "alignments": [{"pair_index": 0, "side": "fragment"}],
    }
    return request


def load(tmp_path, request):
    path = tmp_path / "ground.json"
    path.write_text(json.dumps(request))
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        return dispatch.load_run_config(path)


def test_explicit_ground_selector_and_formula(tmp_path, config):
    raw = ground_request(config)
    direct = load(tmp_path, raw)
    raw.pop("hamiltonian")
    raw["coupling"] = {
        "surface": "ground",
        "embedding": "mechanical",
        "me_model": "fragment-response",
    }
    chosen = load(tmp_path, raw)
    assert chosen == direct
    description = dispatch.describe_run(chosen)
    assert description["coupling"] == raw["coupling"]
    assert description["assembly"]["native_pair"] == "zero"
    assert description["energy_contract"]["native_baseline"] == "zero"
    names = [v["name"] for v in description["energy_contract"]["total_terms"]]
    assert names == [
        "low_full",
        "low_qm",
        "isolated_ground",
        "fragment_pair_deltas",
        "declared_environment_one_body",
    ]
    assert configured_reference(chosen) is None
    assert configured_auxiliary_providers(chosen) == {}
    assert "selected_state_terms" not in description


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "selected",
        "field",
        "cluster",
        "pair-overlap",
        "monomer-overlap",
        "qm-align",
        "outside",
        "method",
        "periodic",
        "raw-model",
        "symbol",
        "loose-scf",
        "grid",
        "reuse-type",
        "velocity-only",
        "wrong-selector",
    ],
)
def test_ground_input_rejects_ambiguity_before_engine(tmp_path, config, damage):
    raw = ground_request(config)
    terms = raw["fragment_interactions"]
    if damage == "missing":
        raw.pop("fragment_interactions")
    elif damage == "selected":
        raw["provider"] = copy.deepcopy(config["provider"])
    elif damage == "field":
        raw["embedding"] = copy.deepcopy(config["embedding"])
    elif damage == "cluster":
        terms["cluster_gap"] = {}
    elif damage == "pair-overlap":
        terms["pairs"] *= 2
    elif damage == "monomer-overlap":
        terms["alignments"] *= 2
    elif damage == "qm-align":
        terms["alignments"][0]["side"] = "core"
    elif damage == "outside":
        terms["pairs"][0]["fragment_region"] = raw["qm_region"]
    elif damage == "method":
        raw["qm"]["xc"] = "pbe"
    elif damage == "periodic":
        raw["periodic"] = True
    elif damage == "raw-model":
        (tmp_path / "model.pt").write_bytes((tmp_path / "model.pt2").read_bytes())
        raw["model"] = "model.pt"
    elif damage == "symbol":
        raw["environment_region"]["symbols"][0] = "C"
    elif damage == "loose-scf":
        raw["qm"]["conv_tol"] = 1e-6
    elif damage == "grid":
        raw["qm"]["grid_response"] = False
    elif damage == "reuse-type":
        raw["reuse_ground_terms"] = 1
    elif damage == "velocity-only":
        raw["velocity_seed"] = 42
    else:
        raw["coupling"] = {
            "surface": "selected",
            "embedding": "mechanical",
            "me_model": "fragment-response",
        }
    with patch("qmdpa.pbe0_tda.PySCFPBE0GroundBackend") as high, pytest.raises(ValueError):
        load(tmp_path, raw)
    high.assert_not_called()


def test_restart_identity_binds_correction_and_keeps_run_length_independent(tmp_path, config):
    raw = ground_request(config)
    first = load(tmp_path, raw)
    raw.update(steps=1, sample_interval=2)
    assert load(tmp_path, raw)["physics_sha256"] == first["physics_sha256"]
    raw["fragment_interactions"]["alignments"] = []
    assert load(tmp_path, raw)["physics_sha256"] != first["physics_sha256"]
    raw = ground_request(config)
    raw["reuse_ground_terms"] = True
    assert load(tmp_path, raw)["physics_sha256"] != first["physics_sha256"]


def test_validation_needs_no_quantum_or_selected_reference(tmp_path, config):
    raw = ground_request(config)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pyscf", "gpu4pyscf")):
            raise AssertionError("Unexpected scientific engine import")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        load(tmp_path, raw)


class Quadratic:
    def __init__(self, diagonal, coupling):
        self.diagonal, self.coupling, self.failure = diagonal, coupling, False

    def evaluate(self, geometry):
        if self.failure:
            raise EvaluationError("injected ground failure")
        x = geometry.positions
        total = np.sum(x, axis=0)
        energy = self.diagonal * np.sum(x * x) + self.coupling * np.dot(total, total)
        return Evaluation(energy, -2 * self.diagonal * x - 2 * self.coupling * total)


def calculator(qm=(22,), env=(99,)):
    low, high = Quadratic(0.5, 0.3), Quadratic(1.2, 0.7)
    delta = CrossInteractionDelta(
        high, high, high, low, low, low, core_atom_ids=(22,), fragment_atom_ids=env
    )
    terms = OneBodyAlignedPairSet((delta,), alignments=((0, "fragment"),))
    result = GroundFragmentCalculator(
        SubtractiveCalculator(low, high), terms, qm_atom_ids=qm, environment_atom_ids=env
    )
    return result, high


def test_complete_two_fragment_replacement_equals_full_high_and_preserves_order():
    calc, high = calculator()
    geom = Geometry((99, 22), ("O", "H"), [[0.7, -0.1, 0.4], [-0.2, 0.3, -0.6]])
    expected = high.evaluate(geom)
    actual = calc.evaluate(geom)
    assert actual.energy == pytest.approx(expected.energy, abs=1e-14)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=1e-14, rtol=0)


def test_partial_pair_coverage_has_full_conservative_force():
    calc, _ = calculator(qm=(22, 5))
    geom = Geometry(
        (99, 5, 22), ("O", "H", "O"), [[0.7, -0.1, 0.4], [-0.2, 0.3, -0.6], [0.1, 0.8, -0.2]]
    )
    result = calc.evaluate_terms(geom)
    for row in range(3):
        for axis in range(3):
            plus, minus = geom.positions.copy(), geom.positions.copy()
            plus[row, axis] += 1e-5
            minus[row, axis] -= 1e-5
            ep = calc.evaluate(Geometry(geom.atom_ids, geom.symbols, plus)).energy
            em = calc.evaluate(Geometry(geom.atom_ids, geom.symbols, minus)).energy
            assert -(ep - em) / 2e-5 == pytest.approx(result.total.forces[row, axis], abs=1e-9)
    np.testing.assert_allclose(
        result.total.forces, result.baseline.forces + result.correction.forces
    )


def test_membership_element_and_failed_ground_result_are_not_accepted():
    calc, high = calculator()
    geom = Geometry((22, 99), ("O", "H"), [[0.1, 0.2, 0.3], [0.3, 0.1, -0.1]])
    calc.evaluate(geom)
    with pytest.raises(ValueError):
        calc.evaluate(Geometry((22, 98), geom.symbols, geom.positions))
    with pytest.raises(ValueError):
        calc.evaluate(Geometry(geom.atom_ids, ("H", "O"), geom.positions))
    high.failure = True
    with pytest.raises(EvaluationError, match="injected"):
        calc.evaluate(geom)


def test_native_factory_has_no_selected_or_field_provider(tmp_path, config):
    raw = ground_request(config)
    raw["reuse_ground_terms"] = True
    chosen = load(tmp_path, raw)
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.pbe0_tda.PySCFPBE0GroundBackend") as high,
        patch("qmdpa.pbe0_tda.PySCFPBE0TDABackend") as state,
        patch("qmdpa.pbe0_tda.PySCFPBE0ElectrostaticGroundBackend") as field,
    ):
        bridge = fragment_ground._ground_fragment_coupling(NativeDriver(), None, chosen)
    assert isinstance(bridge.calculator, GroundFragmentCalculator)
    assert isinstance(bridge.calculator.correction, OneBodyAlignedPairSet)
    assert bridge.state_observer is None
    assert low.call_count == 2
    assert high.call_count == 3
    state.assert_not_called()
    field.assert_not_called()
