# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Combination rejection, lazy construction, derivative scope and config identity."""

import copy
from unittest.mock import patch

import pytest

from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.providers import make_qm_provider, provider_contract, validate_provider

GROUND = {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}


def test_default_and_explicit_requests_are_equal_detached_and_not_mutated():
    request = copy.deepcopy(GROUND)
    normalized = validate_provider(request)
    assert normalized == validate_provider() == GROUND
    normalized["state"]["method"] = "changed"
    assert request == GROUND
    assert validate_provider() == GROUND


@pytest.mark.parametrize("electrostatic", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"engine": "cp2k"},
        {"device": "gpu"},
        {"state": {"method": "tda"}},
        {"state": {"method": "ground", "root": 1}},
        {"extra": True},
        {"device": None},
    ],
)
def test_unsupported_intersections_fail_before_constructing_provider(change, electrostatic):
    request = {**copy.deepcopy(GROUND), **change}
    charges = (
        FixedEmbeddingCharges([4], [0.0], source="Explicit test zero") if electrostatic else None
    )
    with (
        patch("qmdpa.backends.PySCFBackend") as molecular,
        patch("qmdpa.embedding.PySCFElectrostaticBackend") as embedded,
        pytest.raises(ValueError),
    ):
        make_qm_provider({}, provider=request, charges=charges)
    molecular.assert_not_called()
    embedded.assert_not_called()


@pytest.mark.parametrize("specification", [False, [], "", {"engine": "pyscf"}])
def test_malformed_provider_rejected(specification):
    with pytest.raises(ValueError):
        validate_provider(specification)


def test_contract_distinguishes_qm_gradient_from_both_side_forces():
    isolated = provider_contract()
    embedded = provider_contract(electrostatic=True)
    assert isolated["force_coordinates"] == ["qm"]
    assert embedded["force_coordinates"] == ["qm", "environment"]
    assert embedded["environment"] == "fixed-charge"
    assert isolated["energy"] == embedded["energy"] == "total-electronic-and-nuclear"
    assert isolated["stress"] == embedded["stress"] == "unsupported"
    assert embedded["qualification"] == "requires-matched-runtime-record"
    with pytest.raises(TypeError):
        provider_contract(electrostatic=1)


def test_existing_constructors_receive_unchanged_settings_and_charges():
    settings = {"xc": "pbe", "basis": "def2-svp", "threads": 4}
    with patch("qmdpa.backends.PySCFBackend") as molecular:
        assert make_qm_provider(settings) is molecular.return_value
        molecular.assert_called_once_with(**settings)
    charges = FixedEmbeddingCharges([4], [0.1], source="Explicit diagnostic ion")
    with patch("qmdpa.embedding.PySCFElectrostaticBackend") as embedded:
        assert make_qm_provider(settings, charges=charges) is embedded.return_value
        embedded.assert_called_once_with(charges, **settings)
    assert settings == {"xc": "pbe", "basis": "def2-svp", "threads": 4}
    with pytest.raises(TypeError, match="FixedEmbeddingCharges"):
        make_qm_provider(settings, charges=object())


@pytest.mark.parametrize(
    "state",
    [None, [], {}, {"method": "ground", "reference": {}}, {"method": "rhf-tda"}],
)
def test_state_schema_rejects_ambiguous_or_incomplete_requests(state):
    with pytest.raises(ValueError):
        validate_provider({**GROUND, "state": state})


def test_ground_registry_and_factory_do_not_import_excited_modules():
    import builtins

    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name in ("excited", "state_records") or name.startswith(("pyscf", "gpu4pyscf")):
            raise AssertionError(f"Unexpected optional import: {name}")
        return original(name, *args, **kwargs)

    with (
        patch("qmdpa.backends.PySCFBackend") as backend,
        patch("builtins.__import__", side_effect=checked),
    ):
        assert validate_provider() == GROUND
        assert provider_contract()["state"] == {"method": "ground"}
        assert make_qm_provider({"basis": "sto-3g"}) is backend.return_value
