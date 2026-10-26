#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""WTMetaD configuration and immutable accepted-history regression contracts."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_sampling import config, frame, request, thermostat

from qmdpa.dynamics import bind_thermostat
from qmdpa.metadynamics import (
    FORM,
    history_record,
    snapshot_history,
    validate_checkpoint_history,
    validate_frame_history,
)
from qmdpa.sampling import bind_sampling, render_plumed


def metad_config(tmp_path):
    cfg = config(tmp_path)
    bind_thermostat(cfg, thermostat())
    raw = request(tmp_path)
    raw["bias"] = {"form": FORM, "sigma_angstrom": 0.1, "height_ev": 0.001,
                   "pace_steps": 2, "bias_factor": 5.0}
    return cfg, raw


def history(tmp_path, cfg, text=None):
    path = tmp_path / "HILLS"
    if text is None:
        time = 2 * cfg["timestep_fs"] / 1000
        text = ("#! FIELDS time qmdpa_cv sigma_qmdpa_cv height biasf\n"
                "#! SET multivariate false\n#! SET kerneltype stretched-gaussian\n"
                f" {time:.17g} -1 0.1 0.00125 5\n")
    path.write_text(text)
    return path


def test_wt_renderer_and_temperature_identity(tmp_path):
    cfg, raw = metad_config(tmp_path)
    bind_sampling(cfg, raw, tmp_path)
    text = render_plumed(cfg["sampling"], tmp_path, restart=True)
    assert text.startswith("RESTART\nUNITS")
    assert "METAD ARG=qmdpa_cv" in text
    assert "BIASFACTOR=5 TEMP=300" in text
    assert "FMT={ %.17g}" in text
    assert "GRID" not in text and "ADAPTIVE" not in text
    cfg2, raw2 = metad_config(tmp_path)
    cfg2["thermostat"]["temperature_K"] = 310
    bind_sampling(cfg2, raw2, tmp_path)
    assert cfg2["physics_sha256"] != cfg["physics_sha256"]


@pytest.mark.parametrize("key,value", [
    ("pace_steps", True), ("pace_steps", 0), ("pace_steps", 1.5),
    ("sigma_angstrom", 0), ("height_ev", -1), ("height_ev", float("nan")),
    ("bias_factor", 1), ("bias_factor", 0.5), ("bias_factor", float("inf")),
    ("grid", True), ("temperature_K", 300),
])
def test_wt_invalid_parameters(tmp_path, key, value):
    cfg, raw = metad_config(tmp_path)
    raw["bias"][key] = value
    with pytest.raises(ValueError):
        bind_sampling(cfg, raw, tmp_path)


def test_wt_requires_declared_nvt(tmp_path):
    cfg, raw = metad_config(tmp_path)
    cfg.pop("thermostat")
    with pytest.raises(ValueError, match="thermostat"):
        bind_sampling(cfg, raw, tmp_path)


def test_accepted_history_is_immutable_and_work_is_scaled(tmp_path):
    cfg, raw = metad_config(tmp_path)
    bind_sampling(cfg, raw, tmp_path)
    path = history(tmp_path, cfg)
    record = history_record(path, cfg, 2)
    assert record["hill_count"] == 1
    assert record["deposition_work_eV"] == pytest.approx(0.001)
    item = frame()
    item["step"] = 2
    item["sampling"]["history"] = record
    validate_frame_history(item, cfg)
    cp = {"frame": item}
    snapshot_history(cp, tmp_path, path, cfg)
    validate_checkpoint_history(cp, tmp_path / "checkpoint.json", cfg)
    frozen = Path(cp["sampling_history_path"])
    original = frozen.read_bytes()
    path.write_text("failed segment must not replace the accepted history\n")
    assert frozen.read_bytes() == original
    validate_checkpoint_history(cp, tmp_path / "checkpoint.json", cfg)
    frozen.write_text(original.decode() + " 0.123 -1 0.1 0.001 5\n")
    with pytest.raises(ValueError, match="hills"):
        validate_checkpoint_history(cp, tmp_path / "checkpoint.json", cfg)


@pytest.mark.parametrize("replacement", [
    " 0.12 -1 0.1 0.00125 5",  # Wrong deposition time.
    " 0.0001 -1 0.2 0.00125 5",  # Wrong width.
    " 0.0001 -1 0.1 -0.001 5",  # Negative hill.
    " 0.0001 -1 0.1 0.00125 6",  # Wrong factor.
    " 0.0001 nan 0.1 0.00125 5",  # Nonfinite CV.
    " 0.0001 -1 0.1 0.00125",  # Missing column.
])
def test_corrupted_history_is_rejected(tmp_path, replacement):
    cfg, raw = metad_config(tmp_path)
    bind_sampling(cfg, raw, tmp_path)
    path = history(tmp_path, cfg)
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n" + replacement + "\n")
    with pytest.raises(ValueError):
        history_record(path, cfg, 2)


def test_missing_duplicate_and_future_hills_are_rejected(tmp_path):
    cfg, raw = metad_config(tmp_path)
    bind_sampling(cfg, raw, tmp_path)
    path = history(tmp_path, cfg)
    for step in [0, 4]:
        with pytest.raises(ValueError, match="hills"):
            history_record(path, cfg, step)
    path.write_text(path.read_text() + path.read_text().splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="hills"):
        history_record(path, cfg, 2)


def test_checkpoint_metadata_and_foreign_history_rejected(tmp_path):
    cfg, raw = metad_config(tmp_path)
    bind_sampling(cfg, raw, tmp_path)
    item = frame()
    item["step"] = 2
    item["sampling"]["history"] = history_record(history(tmp_path, cfg), cfg, 2)
    cp = {"frame": item}
    snapshot_history(cp, tmp_path, tmp_path / "HILLS", cfg)
    altered = deepcopy(cp)
    altered["sampling_history"]["file"] = "../HILLS"
    with pytest.raises(ValueError, match="differs"):
        validate_checkpoint_history(altered, tmp_path / "checkpoint.json", cfg)
    with pytest.raises(ValueError, match="Unexpected"):
        validate_checkpoint_history(cp, tmp_path / "checkpoint.json", {})
    with pytest.raises(FileExistsError):
        snapshot_history(cp, tmp_path, tmp_path / "HILLS", cfg)


def test_environment_cannot_silently_change_wt_kernel(tmp_path, monkeypatch):
    from qmdpa.sampling import verify_loaded_kernel

    cfg, raw = metad_config(tmp_path)
    bind_sampling(cfg, raw, tmp_path)
    monkeypatch.setenv("PLUMED_DP2CUTOFF_NOSTRETCH", "1")
    with pytest.raises(ValueError, match="stretched"):
        verify_loaded_kernel(cfg["sampling"])
