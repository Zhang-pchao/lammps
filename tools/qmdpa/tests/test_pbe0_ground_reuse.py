# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Exact ground reuse keeps method, input, failure and result ownership contracts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_fragment_response_native import select

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import NativeDriver, config, load
from test_pbe0_tda import pbe0_reference

from qmdpa import gap_anchor_cli
from qmdpa import pbe0_tda as api
from qmdpa.core import Evaluation, EvaluationError, Geometry
from qmdpa.embedding import EmbeddedEvaluation

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def setup(monkeypatch, embedded, enabled=True):
    ref = pbe0_reference(embedded)

    def initialize(self, **options):
        self.__dict__.update(options)
        self._identity = None
        self.calls = 0
        self.fail = False

    def embedded_initialize(self, charges, **options):
        initialize(self, **options)
        self.charges = charges

    def calculation(self, *geometries):
        self.calls += 1
        if self.fail:
            raise EvaluationError("intentional fresh calculation failure")
        q = geometries[0]
        if len(geometries) == 1:
            return Evaluation(float(np.sum(q.positions**2)), -2 * q.positions)
        e = geometries[1]
        self._identity = (q.atom_ids, q.symbols, e.atom_ids, e.symbols)
        return EmbeddedEvaluation(
            float(np.sum(q.positions**2) + np.sum(e.positions**2)),
            q.atom_ids,
            e.atom_ids,
            -2 * q.positions,
            -2 * e.positions,
            {"nested": {"values": [1.0, 2.0]}},
        )

    monkeypatch.setattr(api, "_load_runtime", lambda settings: SimpleNamespace(record=ref.runtime))
    monkeypatch.setattr(api.PySCFBackend, "__init__", initialize)
    monkeypatch.setattr(api.PySCFElectrostaticBackend, "__init__", embedded_initialize)
    monkeypatch.setattr(api.PySCFBackend, "evaluate", calculation)
    monkeypatch.setattr(api.PySCFElectrostaticBackend, "evaluate", calculation)
    backend = (
        api.PySCFPBE0ElectrostaticGroundBackend(
            ref.settings, ref.charges, reuse_identical_geometry=enabled
        )
        if embedded
        else api.PySCFPBE0GroundBackend(ref.settings, reuse_identical_geometry=enabled)
    )
    args = (ref.qm_geometry, ref.environment_geometry) if embedded else (ref.qm_geometry,)
    return backend, args


def moved(geometry):
    positions = geometry.positions.copy()
    positions[0, 0] = np.nextafter(positions[0, 0], np.inf)
    return Geometry(geometry.atom_ids, geometry.symbols, positions)


@pytest.mark.parametrize("embedded", [False, True])
def test_identical_only_last_input_and_detached_result(monkeypatch, embedded):
    b, args = setup(monkeypatch, embedded)
    first = b.evaluate(*args)
    second = b.evaluate(*args)
    assert first is not second and b.calls == 1
    if embedded:
        first.diagnostics["nested"]["values"][0] = 99.0
        assert b.evaluate(*args).diagnostics["nested"]["values"][0] == 1.0
        assert not np.shares_memory(
            first.qm_forces_ev_per_angstrom, second.qm_forces_ev_per_angstrom
        )
    else:
        assert not np.shares_memory(first.forces, second.forces)
    for i in range(len(args)):
        changed = list(args)
        changed[i] = moved(args[i])
        before = b.calls
        b.evaluate(*changed)
        b.evaluate(*args)
        assert b.calls == before + 2  # No rounded key or older-geometry fallback.
    b.clear_reuse()
    before = b.calls
    b.evaluate(*args)
    assert b.calls == before + 1


@pytest.mark.parametrize("embedded", [False, True])
def test_default_off_and_failed_request_clear_old_success(monkeypatch, embedded):
    b, args = setup(monkeypatch, embedded, False)
    b.evaluate(*args)
    b.evaluate(*args)
    assert b.calls == 2 and b.reuse_statistics()["hits"] == 0
    b, args = setup(monkeypatch, embedded)
    b.evaluate(*args)
    b.fail = True
    changed = (moved(args[0]), *args[1:])
    with pytest.raises(EvaluationError, match="intentional"):
        b.evaluate(*changed)
    assert b._last_ground_result is None
    b.fail = False
    b.evaluate(*args)
    assert b.calls == 3 and b.reuse_statistics()["failures"] == 1


@pytest.mark.parametrize(
    "change", ["settings", "basis", "runtime", "runtime-record", "extra", "charges"]
)
def test_changed_provider_cannot_hit_cached_result(monkeypatch, change):
    b, args = setup(monkeypatch, True)
    b.evaluate(*args)
    if change == "settings":
        b.settings = replace(b.settings, conv_tol=5e-13)
    elif change == "basis":
        b.basis = "sto-3g"
    elif change == "runtime":
        b._runtime = SimpleNamespace(record=b._runtime.record)
    elif change == "runtime-record":
        b._runtime.record["pyscf_version"] = "foreign"
    elif change == "extra":
        b.include_qm_dipole = True
    else:
        b.charges = replace(b.charges, source="foreign field source")
    with pytest.raises(EvaluationError, match="configuration changed"):
        b.evaluate(*args)
    assert b.calls == 1 and b._last_ground_result is None


def test_embedded_membership_and_invalid_geometry_clear_reuse(monkeypatch):
    b, args = setup(monkeypatch, True)
    b.evaluate(*args)
    q = args[0]
    changed = Geometry(q.atom_ids[::-1], q.symbols[::-1], q.positions[::-1])
    with pytest.raises(ValueError, match="membership"):
        b.evaluate(changed, args[1])
    assert b._last_ground_result is None
    b.evaluate(*args)
    with pytest.raises(ValueError):
        b.evaluate(args[0])  # Missing environment cannot reuse a complete field result.
    assert b._last_ground_result is None


@pytest.mark.parametrize("value", [1, "true", None])
def test_non_boolean_reuse_rejects_before_quantum_runtime(monkeypatch, value):
    ref = pbe0_reference(True)
    monkeypatch.setattr(api, "_load_runtime", lambda s: pytest.fail("Unexpected runtime"))
    with pytest.raises(TypeError):
        api.PySCFPBE0GroundBackend(ref.settings, reuse_identical_geometry=value)
    with pytest.raises(TypeError):
        api.PySCFPBE0ElectrostaticGroundBackend(
            ref.settings, ref.charges, reuse_identical_geometry=value
        )


def test_native_option_is_explicit_and_cannot_change_hamiltonian(tmp_path, config):
    select(config)
    disabled = load(tmp_path, config)
    config["reuse_ground_terms"] = True
    enabled = load(tmp_path, config)
    assert enabled["hamiltonian"] == disabled["hamiltonian"]
    assert enabled["provider"] == disabled["provider"]
    assert enabled["fragment_response"] == disabled["fragment_response"]
    assert enabled["qm"] == disabled["qm"]
    assert enabled["physics_sha256"] != disabled["physics_sha256"]
    assert enabled["assembly"]["ground_result_reuse"] == "last-successful-exact-input"
    config["reuse_ground_terms"] = 1
    with pytest.raises(ValueError, match="boolean"):
        load(tmp_path, config)


def test_native_shares_only_identical_declared_ground_fragments(config):
    anchor, pair, field = select(config)
    config["reuse_ground_terms"] = True
    baseline = SimpleNamespace(
        isolated_ground_backend=SimpleNamespace(evaluate=lambda g: None),
        embedded_selected_backend=SimpleNamespace(reference=anchor, charges=anchor.charges),
        evaluate=lambda g: None,
    )

    def high(settings, **options):
        assert options == {"reuse_identical_geometry": True}
        return SimpleNamespace(evaluate=lambda g: None, settings=settings)

    with (
        patch("qmdpa.backends.DeepMDBackend"),
        patch("qmdpa.gap_anchor.make_pbe0_gap_anchor", return_value=baseline) as factory,
        patch.object(api, "PySCFPBE0GroundBackend", side_effect=high) as grounds,
        patch.object(
            api,
            "PySCFPBE0ElectrostaticGroundBackend",
            return_value=SimpleNamespace(charges=field.charges, evaluate=lambda *a: None),
        ),
        patch.object(api, "PySCFPBE0TDABackend", return_value=SimpleNamespace(reference=pair)),
        patch.object(
            api,
            "PySCFPBE0TDAElectrostaticBackend",
            return_value=SimpleNamespace(reference=field, charges=field.charges),
        ),
    ):
        bridge = gap_anchor_cli._gap_anchor_coupling(NativeDriver(), None, config)
    delta = bridge.calculator.correction
    pair_term = bridge.calculator.baseline.correction.corrections[0]
    assert pair_term._high[0] is delta.pair_ground
    assert pair_term._high[1] is baseline.isolated_ground_backend
    assert pair_term._high[2] is not delta.pair_ground
    assert grounds.call_count == 2  # Pair and environment; core already in anchor.
    assert factory.call_args.kwargs["reuse_ground_terms"] is True
