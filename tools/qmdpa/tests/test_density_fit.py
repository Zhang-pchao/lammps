# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit fitting, complete derivative and method-identity contracts."""

import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_cli

# Register shared pytest fixtures consumed through test function arguments.
from test_additive_config import case as additive_case
from test_backends import FakeSCF, pyscf_modules, water
from test_embedding import fake_modules, fixed_charges, geometries
from test_excited_config import provider_request, reference
from test_pbe0_ground_reuse import setup

from qmdpa.backends import PySCFBackend, validate_density_fit
from qmdpa.config import validate_qm
from qmdpa.core import EvaluationError
from qmdpa.dispatch import load_run_config
from qmdpa.embedding import PySCFElectrostaticBackend

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['additive_case']


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "weigend",
        [],
        {},
        {"auxbasis": None},
        {"auxbasis": ""},
        {"auxbasis": " "},
        {"auxbasis": 1},
        {"auxbasis": "../custom.dat"},
        {"auxbasis": "weigend", "auxbasis_response": False},
        {"auxbasis": "weigend", "only_dfj": True},
    ],
)
def test_invalid_options_fail_before_engine_import(value):
    with patch.dict(sys.modules, {"pyscf": None}):
        with pytest.raises(ValueError, match="density_fit"):
            PySCFBackend(density_fit=value)
        with pytest.raises(ValueError, match="density_fit"):
            validate_qm({"density_fit": value})


def test_defaults_preserve_records_and_settings_are_detached():
    assert validate_qm({}) == validate_qm({"density_fit": None})
    assert "density_fit" not in validate_qm({})
    request = {"auxbasis": "def2-universal-jkfit"}
    normalized = validate_density_fit(request)
    request["auxbasis"] = "changed"
    assert normalized == {"auxbasis": "def2-universal-jkfit"}


@pytest.mark.parametrize("multiplicity", [1, 2])
@pytest.mark.parametrize("embedded", [False, True])
def test_official_fit_and_complete_gradient_are_used(multiplicity, embedded):
    observed = {}
    method = FakeSCF()

    def attach(value):
        def fit(*, auxbasis, only_dfj):
            observed["fit"] = (auxbasis, only_dfj)
            value.with_df = SimpleNamespace(auxbasis=auxbasis)
            value.gradient.auxbasis_response = False
            return value

        value.density_fit = fit

    attach(method)
    modules = pyscf_modules(method, observed)
    if embedded:
        observed["mutate"] = attach
        modules = fake_modules(observed)
    with patch.dict(sys.modules, modules):
        options = {"multiplicity": multiplicity, "density_fit": {"auxbasis": "weigend"}}
        if embedded:
            backend = PySCFElectrostaticBackend(fixed_charges(), **options)
            result = backend.evaluate(*geometries())
            method = observed["methods"][0]
            assert result.diagnostics["density_fit"] == {
                "auxbasis": "weigend",
                "only_dfj": False,
                "auxbasis_response": True,
            }
        else:
            backend = PySCFBackend(**options)
            backend.evaluate(water())
    assert observed["fit"] == ("weigend", False)
    assert method.gradient.grid_response is True
    assert method.gradient.auxbasis_response is True


def test_missing_response_and_failed_scf_do_not_return_forces():
    method = FakeSCF()

    def fit(**kwargs):
        method.with_df = SimpleNamespace()
        return method

    method.density_fit = fit
    with patch.dict(sys.modules, pyscf_modules(method, {})):
        backend = PySCFBackend(density_fit={"auxbasis": "weigend"})
        with pytest.raises(EvaluationError, match="auxiliary-basis response"):
            backend.evaluate(water())
        method.converged = False
        method.gradient_called = False
        with pytest.raises(EvaluationError, match="SCF did not converge"):
            backend.evaluate(water())
        assert method.gradient_called is False


@pytest.mark.parametrize("embedded", [False, True])
def test_selected_state_settings_reject_fitting(embedded):
    from dataclasses import asdict

    ref = reference(embedded=embedded)
    with pytest.raises(ValueError, match="unsupported RHF/TDA"):
        validate_qm(
            {**asdict(ref.settings), "density_fit": {"auxbasis": "weigend"}},
            provider=provider_request(ref),
            electrostatic=embedded,
        )


@pytest.mark.parametrize("embedded", [False, True])
def test_matched_pbe0_cache_cannot_hide_a_changed_integral_method(monkeypatch, embedded):
    backend, args = setup(monkeypatch, embedded)
    backend.evaluate(*args)
    backend.density_fit = {"auxbasis": "weigend"}
    with pytest.raises(EvaluationError, match="configuration changed"):
        backend.evaluate(*args)
    assert backend.calls == 1 and backend._last_ground_result is None


def test_mechanical_option_changes_physics_identity():
    fixture = test_cli.ConfigTests()
    fixture.setUp()
    try:
        original = fixture.load()
        fixture.config["qm"]["density_fit"] = {"auxbasis": "weigend"}
        fitted = fixture.load()
        assert original["physics_sha256"] != fitted["physics_sha256"]
        assert fitted["qm"]["density_fit"] == {"auxbasis": "weigend"}
        fixture.config["qm"]["density_fit"]["auxbasis"] = "def2-universal-jkfit"
        assert fixture.load()["physics_sha256"] != fitted["physics_sha256"]
    finally:
        fixture.doCleanups()


def test_additive_selector_preserves_fitting_in_restart_identity(tmp_path, additive_case):
    raw = copy.deepcopy(additive_case)
    raw["coupling"] = {"surface": "ground", "embedding": "electrostatic", "ee_model": "additive"}
    path = tmp_path / "config.json"
    with patch("qmdpa.config.importlib.metadata.version", return_value="test"):
        path.write_text(json.dumps(raw))
        original = load_run_config(path)
        raw["qm"]["density_fit"] = {"auxbasis": "weigend"}
        path.write_text(json.dumps(raw))
        fitted = load_run_config(path)
    assert original["physics_sha256"] != fitted["physics_sha256"]
    assert fitted["qm"]["density_fit"] == {"auxbasis": "weigend"}
