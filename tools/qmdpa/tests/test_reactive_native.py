#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Compressed reactive EE admission, calibration and owned resource routing."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_execution import request
from test_reactive_additive import document
from test_reactive_config import load, numerical_versions  # noqa: F401
from test_reactive_sampling import sampling
from test_sampling import thermostat

from qmdpa.core import Evaluation
from qmdpa.dispatch import describe_run
from qmdpa.execution import native_arguments, verify_native
from qmdpa.inference import COMPRESSED_PROFILE
from qmdpa.reactive_cli import _reactive_coupling
from qmdpa.reactive_config import build_mixer
from qmdpa.runtime import ProviderResources
from qmdpa.water_reference import prepare_reference


def compressed(tmp_path, style="deepmd/kk"):
    archive = request(tmp_path, style)
    raw = document()
    raw.update(model=archive["model"], model_sha256=archive["model_sha256"],
               inference_profile=COMPRESSED_PROFILE,
               execution={**archive["execution"], "domain": "environment"})
    raw["buffer_reference"].update(
        schema_version=2, model_sha256=raw["model_sha256"],
        inference_profile=COMPRESSED_PROFILE, low_backend=style)
    return raw


@pytest.mark.parametrize("style", ["deepmd", "deepmd/kk"])
def test_compressed_opes_is_an_environment_domain_with_zero_outer_pair(tmp_path, style):
    raw = compressed(tmp_path, style)
    raw.update(steps=8, sample_interval=2, sampling=sampling(tmp_path), thermostat=thermostat())
    cfg = load(tmp_path, raw)
    view = describe_run(cfg)
    assert view["assembly"]["native_pair"] == "zero"
    assert view["assembly"]["environment_pair"] == style
    assert view["assembly"]["additional_native_bias"] == "plumed"
    assert "-k" not in native_arguments(cfg)
    assert cfg["sampling"]["cv"]["atom_ids"] == list(range(1, 10))
    verify_native(SimpleNamespace(extract_global=lambda name: "zero"), cfg)
    with pytest.raises(ValueError, match="actual native pair"):
        verify_native(SimpleNamespace(extract_global=lambda name: style), cfg)


@pytest.mark.parametrize("case", [
    "missing-execution", "full-domain", "missing-domain", "different-backend",
    "raw-reference", "source-reference", "profile", "mechanical", "archive",
])
def test_unqualified_combinations_fail_before_native_construction(tmp_path, case):
    raw = compressed(tmp_path)
    if case == "missing-execution":
        raw.pop("execution")
    elif case == "full-domain":
        raw["execution"]["domain"] = "full"
    elif case == "missing-domain":
        raw["execution"].pop("domain")
    elif case == "different-backend":
        raw["buffer_reference"]["low_backend"] = "deepmd"
    elif case == "raw-reference":
        raw["buffer_reference"]["schema_version"] = 1
    elif case == "source-reference":
        raw["buffer_reference"]["software"]["coupling_sha256"] = "0"*64
    elif case == "profile":
        raw["inference_profile"] = "undeclared"
    elif case == "mechanical":
        raw["hamiltonian"] = "finite-reactive-water-mechanical"
    else:
        archive = request(tmp_path, dtype="float64")
        raw["model_sha256"] = raw["buffer_reference"]["model_sha256"] = archive["model_sha256"]
    with patch("qmdpa.native_deepmd.NativeDeepMDDomainBackend") as native:
        with pytest.raises((TypeError, ValueError)):
            load(tmp_path, raw)
        native.assert_not_called()


def test_low_backend_must_match_calibration(tmp_path):
    cfg = load(tmp_path, compressed(tmp_path))
    with pytest.raises(ValueError, match="calibrated native"):
        build_mixer(cfg, SimpleNamespace(pair_style="deepmd"), Mock())


@pytest.mark.parametrize("failure", [False, True])
def test_resource_is_owned_before_mixer_setup_and_raw_engine_is_unused(tmp_path, failure):
    cfg = load(tmp_path, compressed(tmp_path))
    owner = ProviderResources(tmp_path/"owned")
    low = Mock(pair_style="deepmd/kk", model_sha256=cfg["model_sha256"])
    mixer = object()
    with patch("qmdpa.native_deepmd.NativeDeepMDDomainBackend", return_value=low), \
         patch("qmdpa.inference.prepare_raw_inference") as raw, \
         patch("qmdpa.reactive_config.build_mixer", side_effect=ValueError("setup") if failure else None,
               return_value=mixer), \
         patch("qmdpa.lammps.AdditiveExternalCoupling") as coupling:
        if failure:
            with pytest.raises(RuntimeError, match="ValueError: setup"):
                _reactive_coupling(Mock(), None, cfg, provider_resources=owner)
        else:
            _reactive_coupling(Mock(), None, cfg, provider_resources=owner)
            assert coupling.call_args.args[2] is mixer
        raw.assert_not_called()
        assert owner.close() == []
        low.close_checked.assert_called_once()
        owner.close()
        low.close_checked.assert_called_once()


@pytest.mark.parametrize("failure", [False, True])
def test_compressed_reference_uses_native_energy_and_always_closes(tmp_path, failure):
    raw = compressed(tmp_path)
    ref = raw["buffer_reference"]
    keys = ("model", "model_sha256", "qm", "provider", "inference_profile")
    data = {key: raw[key] for key in keys}
    data.update(low_backend="deepmd/kk", geometry_angstrom=ref["geometry_angstrom"],
                source="native calibration fixture")
    path = tmp_path/"request.json"
    path.write_text(json.dumps(data))
    low = Mock(statistics={"closed": True})
    low.evaluate.side_effect = RuntimeError("native failure") if failure else None
    low.evaluate.return_value = Evaluation(-3., np.zeros((3, 3)))
    high = Mock()
    high.evaluate.return_value = Evaluation(-5., np.zeros((3, 3)))
    output = tmp_path/"reference"
    with patch("qmdpa.native_deepmd.NativeDeepMDDomainBackend", return_value=low), \
         patch("qmdpa.providers.make_qm_provider", return_value=high), \
         patch("qmdpa.inference.prepare_raw_inference") as strict:
        if failure:
            with pytest.raises(RuntimeError, match="native failure"):
                prepare_reference(path, output)
            assert not (output/"reference.json").exists()
            assert (output/"failure.json").exists()
        else:
            result = prepare_reference(path, output)
            assert result["delta_water_ev"] == -2.
            saved = json.loads((output/"reference.json").read_text())
            assert saved["schema_version"] == 2
            assert saved["low_backend"] == "deepmd/kk"
        low.close_checked.assert_called_once()
        strict.assert_not_called()
