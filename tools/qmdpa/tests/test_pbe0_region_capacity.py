# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit PBE0 region capacity never inherits a smaller reference identity."""

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

# Register shared pytest fixtures consumed through test function arguments.
from test_excited_config import case, load
from test_pbe0_provider_state import convert_case, observer, request
from test_pbe0_tda import pbe0_reference
from test_state_records import frame

from qmdpa import excited
from qmdpa import pbe0_tda as pbe0
from qmdpa.core import EvaluationError
from qmdpa.providers import make_qm_provider
from qmdpa.state_records import configured_reference, validate_frame_state

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['case']


@pytest.mark.parametrize(
    "value", [True, 256.0, 1024.0, "1024", 2048.0, "2048", 0, 255, 257, 512, 1023, 1025, 2047, 2049]
)
def test_only_declared_integer_resource_caps_are_accepted(value):
    with pytest.raises((ValueError, TypeError)):
        pbe0.PBE0TDASettings(tda_dense_max_dimension=value)


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("cap", [1024, 2048])
def test_reference_roundtrip_preserves_default_and_separates_explicit_cap(cap, embedded):
    small = pbe0_reference(embedded)
    explicit_default = replace(small, settings=pbe0.PBE0TDASettings(tda_dense_max_dimension=256))
    assert small.settings.tda_dense_max_dimension == 256
    assert explicit_default.to_record() == small.to_record()
    large = replace(small, settings=replace(small.settings, tda_dense_max_dimension=cap))
    loaded = pbe0.PBE0TDAReference.from_record(large.to_record())
    assert loaded.to_record() == large.to_record()
    assert loaded.sha256 != small.sha256
    assert loaded.settings.tda_dense_max_dimension == cap
    for settings, reference in [(large.settings, small), (small.settings, large)]:
        target = pbe0.PySCFPBE0TDAElectrostaticBackend if embedded else pbe0.PySCFPBE0TDABackend
        args = (settings, reference, reference.charges) if embedded else (settings, reference)
        with patch("qmdpa.pbe0_tda._load_runtime") as runtime:
            with pytest.raises(ValueError, match="settings differ"):
                target(*args)
        runtime.assert_not_called()


@pytest.mark.parametrize("cap", [1024, 2048])
def test_dispatch_and_restart_physics_identity_separate_caps(cap, tmp_path, case):
    embedded, small, configuration = convert_case(case)
    original = load(tmp_path, (embedded, small, configuration))
    large = replace(small, settings=replace(small.settings, tda_dense_max_dimension=cap))
    updated = copy.deepcopy(configuration)
    updated.update(qm=asdict(large.settings), provider=request(large))
    resolved = load(tmp_path, (embedded, large, updated))
    assert configured_reference(resolved).to_record() == large.to_record()
    assert resolved["physics_sha256"] != original["physics_sha256"]
    target = "PySCFPBE0TDAElectrostaticBackend" if embedded else "PySCFPBE0TDABackend"
    with (
        patch("qmdpa.pbe0_tda." + target) as backend,
        patch("qmdpa.state_records.RecordingTDAProvider"),
    ):
        make_qm_provider(updated["qm"], provider=updated["provider"], charges=large.charges)
    assert backend.call_args.args[0] == large.settings
    assert backend.call_args.args[1].to_record() == large.to_record()
    with patch("qmdpa.pbe0_tda._load_runtime") as runtime:
        with pytest.raises(ValueError):
            make_qm_provider(asdict(small.settings), provider=request(large), charges=large.charges)
    runtime.assert_not_called()


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("cap", [1024, 2048])
def test_saved_state_cannot_inherit_a_different_cap(cap, embedded):
    small, _, recording = observer(embedded)
    recording.evaluate(small.qm_geometry, small.environment_geometry)
    saved = frame(small)
    saved["electronic_state"] = recording.accepted_record(saved)
    validate_frame_state(saved, {"periodic": False, "provider": request(small)})
    large = replace(small, settings=replace(small.settings, tda_dense_max_dimension=cap))
    with pytest.raises(ValueError, match="reference"):
        validate_frame_state(saved, {"periodic": False, "provider": request(large)})


@pytest.mark.parametrize("cap,nocc,nvir", [(256, 13, 49), (1024, 18, 68), (2048, 30, 80)])
def test_actual_dimension_guard_precedes_operator_setup(cap, nocc, nvir):
    td = SimpleNamespace(_scf=SimpleNamespace(mo_occ=np.array([2] * nocc + [0] * nvir)))
    td.gen_vind = lambda: pytest.fail("Oversize TDA must reject before operator setup")
    with pytest.raises(EvaluationError, match="declared cap"):
        excited._dense_tda(td, pbe0.PBE0TDASettings(tda_dense_max_dimension=cap))


@pytest.mark.parametrize("nocc,nvir,cap", [(13, 49, 1024), (18, 68, 2048)])
def test_explicit_cap_executes_declared_official_operator_contract(nocc, nvir, cap):
    # A synthetic diagonal operator verifies capacity, batching and selected-root
    # residual checks. It does not qualify any quantum gradient or geometry.
    dimension = nocc * nvir
    diagonal = np.linspace(0.1, 0.9, dimension)
    batches = []

    def action(vectors):
        batches.append(vectors.shape)
        return vectors * diagonal

    td = SimpleNamespace(
        _scf=SimpleNamespace(mo_occ=np.array([2] * nocc + [0] * nvir)),
        positive_eig_threshold=1e-3,
        gen_vind=lambda: (action, None),
    )
    result = excited._dense_tda(td, pbe0.PBE0TDASettings(tda_dense_max_dimension=cap))
    np.testing.assert_allclose(td.e, diagonal[:3], rtol=0, atol=1e-15)
    assert result["tda_dimension"] == dimension and result["tda_dense_max_dimension"] == cap
    assert sum(n for n, _ in batches[:-1]) == dimension
    assert max(n for n, _ in batches[:-1]) <= 16
    assert batches[-1] == (3, dimension)
    assert max(result["tda_matrix_residuals_hartree"]) < 1e-14
    assert all(x.shape == (nocc, nvir) and y == 0 for x, y in td.xy)


def test_pbe0_capacity_does_not_expand_the_rhf_route():
    assert excited.TDASettings(tda_dense_max_dimension=1024).tda_dense_max_dimension == 1024
    with pytest.raises(ValueError, match="hard dimension cap of 1024"):
        excited.TDASettings(tda_dense_max_dimension=2048)
