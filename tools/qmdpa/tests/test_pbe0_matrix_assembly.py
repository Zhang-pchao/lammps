# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Official PBE0 assembly preserves residual, identity and failure contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_excited import dense_fixture, settings
from test_pbe0_tda import install_solver, pbe0_reference

from qmdpa import excited
from qmdpa import pbe0_tda as pbe0
from qmdpa.core import EvaluationError
from qmdpa.providers import provider_contract, validate_provider


def test_supplied_matrix_is_checked_against_original_operator():
    td, calls = dense_fixture([[0.3, 0.04], [0.04, 0.3]])
    matrix = np.array([[0.3, 0.04], [0.04, 0.3]])
    diag = excited._dense_tda(
        td,
        settings(tda_solver="dense"),
        matrix_builder=lambda *args: (matrix, {"tda_matrix_assembly": "test"}),
    )
    assert diag["tda_matrix_assembly"] == "test"
    assert len(calls) == 1
    td, _ = dense_fixture(matrix)
    with pytest.raises(EvaluationError, match="residual"):
        excited._dense_tda(
            td,
            settings(tda_solver="dense"),
            matrix_builder=lambda *args: (matrix + 1e-6 * np.eye(2), {}),
        )


@pytest.mark.parametrize("mode", ["ok", "exception", "missing-points", "too-large"])
def test_bounded_quadrature_covers_points_and_restores_integrator(mode):
    class Integrator:
        def block_loop(self, mol, grids, nao, deriv, max_memory, **kwargs):
            size = kwargs["blksize"]
            for start in range(0, 1000, size):
                if mode == "missing-points" and start:
                    break
                n = 1000 if mode == "too-large" else min(size, 1000 - start)
                yield (None, None, np.ones(n), None)

    original = Integrator()
    mf = SimpleNamespace(
        xc="pbe0", nlc="", grids=SimpleNamespace(level=4, weights=np.ones(1000)), _numint=original
    )

    def get_ab():
        list(mf._numint.block_loop(None, mf.grids))
        if mode == "exception":
            raise RuntimeError("deliberate engine failure")
        return np.eye(2).reshape(1, 2, 1, 2), None

    td = SimpleNamespace(_scf=mf, get_ab=get_ab)
    if mode == "ok":
        matrix, diag = pbe0._official_pbe0_matrix(td, 1, 2)
        np.testing.assert_array_equal(matrix, np.eye(2))
        assert diag["tda_matrix_grid_points"] == 1000 and diag["tda_matrix_grid_blocks"] == 2
    else:
        with pytest.raises((RuntimeError, EvaluationError)):
            pbe0._official_pbe0_matrix(td, 1, 2)
    assert mf._numint is original


@pytest.mark.parametrize("value", [None, True, 896, "automatic", "get-ab"])
def test_invalid_algorithm_rejects_before_runtime_import(monkeypatch, value):
    monkeypatch.setattr(pbe0, "_load_runtime", lambda *a: pytest.fail("runtime imported"))
    r = pbe0_reference()
    with pytest.raises(ValueError):
        pbe0.PySCFPBE0TDABackend(r.settings, r, matrix_assembly=value)


def test_explicit_assembly_is_kept_for_both_backends_and_reuse(monkeypatch):
    install_solver(monkeypatch)
    for embedded in (False, True):
        r = pbe0_reference(embedded)
        cls = pbe0.PySCFPBE0TDAElectrostaticBackend if embedded else pbe0.PySCFPBE0TDABackend
        args = (r.settings, r, r.charges) if embedded else (r.settings, r)
        b = cls(*args, matrix_assembly="official-get-ab-896", reuse_identical_geometry=True)
        assert b.matrix_assembly == "official-get-ab-896"
        b._matrix_assembly = "operator"
        with pytest.raises(EvaluationError, match="changed"):
            b._reuse_key(r.qm_geometry, r.environment_geometry, r.charges, True)


def test_provider_retains_explicit_algorithm_and_preserves_omitted_legacy_keys():
    r = pbe0_reference()
    spec = {
        "engine": "pyscf",
        "device": "cpu",
        "state": {"method": "pbe0-tda", "reference": r.to_record()},
    }
    assert "matrix_assembly" not in validate_provider(spec)
    assert "matrix_assembly" not in provider_contract(spec)
    spec["matrix_assembly"] = "official-get-ab-896"
    assert validate_provider(spec)["matrix_assembly"] == "official-get-ab-896"
    assert provider_contract(spec)["matrix_assembly"] == "official-get-ab-896"


@pytest.mark.parametrize("method", ["ground", "rhf-tda"])
def test_other_methods_cannot_select_pbe0_assembly(method):
    with pytest.raises(ValueError, match="matrix_assembly"):
        validate_provider(
            {
                "engine": "pyscf",
                "device": "cpu",
                "state": {"method": method},
                "matrix_assembly": "official-get-ab-896",
            }
        )


@pytest.mark.parametrize("embedded", [False, True])
def test_native_record_and_restart_bind_the_actual_matrix_assembly(embedded):
    from test_pbe0_provider_state import observer, request
    from test_state_records import frame

    from qmdpa.state_records import validate_frame_state

    pinned, backend, recording = observer(embedded)
    backend.matrix_assembly = "official-get-ab-896"
    backend.diagnostic.pop("tda_dense_batch_size")
    backend.diagnostic.update(
        tda_matrix_assembly="official-get-ab-896",
        tda_matrix_grid_points=100,
        tda_matrix_grid_blocks=1,
    )
    recording.evaluate(pinned.qm_geometry, pinned.environment_geometry)
    snapshot = frame(pinned)
    snapshot["electronic_state"] = recording.accepted_record(snapshot)
    config = {"periodic": False, "provider": request(pinned)}
    with pytest.raises(ValueError, match="matrix assembly"):
        validate_frame_state(snapshot, config)
    config["provider"]["matrix_assembly"] = "official-get-ab-896"
    validate_frame_state(snapshot, config)


@pytest.mark.parametrize("bad", ["algorithm", "coverage", "zero-blocks", "legacy-batch"])
def test_official_record_rejects_missing_or_foreign_assembly_evidence(bad):
    from test_pbe0_provider_state import observer

    pinned, backend, recording = observer()
    backend.matrix_assembly = "official-get-ab-896"
    backend.diagnostic.pop("tda_dense_batch_size")
    backend.diagnostic.update(
        tda_matrix_assembly="official-get-ab-896",
        tda_matrix_grid_points=100,
        tda_matrix_grid_blocks=1,
    )
    if bad == "algorithm":
        backend.diagnostic["tda_matrix_assembly"] = "operator"
    elif bad == "coverage":
        backend.diagnostic["tda_matrix_grid_points"] = 99
    elif bad == "zero-blocks":
        backend.diagnostic["tda_matrix_grid_blocks"] = 0
    else:
        backend.diagnostic["tda_dense_batch_size"] = 16
    with pytest.raises(ValueError, match="matrix"):
        recording.evaluate(pinned.qm_geometry)
    with pytest.raises(EvaluationError, match="No successful"):
        recording.accepted_record({})
