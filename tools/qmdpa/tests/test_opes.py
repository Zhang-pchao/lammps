#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Water CV and exact OPES state admission, including retained failure inputs."""

from copy import deepcopy

import numpy as np
import pytest
from test_sampling import config, request, thermostat

from qmdpa import opes, water_cv
from qmdpa.dynamics import bind_thermostat
from qmdpa.sampling import (
    NativePlumedBias,
    bind_sampling,
    render_plumed,
    validate_checkpoint_history,
)


def raw_cv():
    return {"form": water_cv.FORM, "oxygen_ids": [1, 4], "hydrogen_ids": [2, 3, 5, 6],
            "coordination_kappa_per_angstrom": 7., "distance_kappa_per_angstrom": 8.}


def configured(tmp_path):
    cfg = config(tmp_path)
    cfg.update(steps=8, sample_interval=2)
    cfg["region"] = {"atom_ids": list(range(1, 7)), "symbols": ["O", "H", "H"]*2}
    bind_thermostat(cfg, thermostat())
    raw = request(tmp_path)
    raw["cv"] = raw_cv()
    raw["bias"] = {"form": opes.FORM, "sigma": [0.15, 0.04], "sigma_min": [0.06, 0.01],
                   "barrier_ev": 0.1, "bias_factor": 5., "pace_steps": 2}
    bind_sampling(cfg, raw, tmp_path)
    return cfg, raw


def state(cfg, step=2):
    time = step * cfg["timestep_fs"] / 1000
    return ("#! FIELDS time qmdpa_logdistance qmdpa_ionization "
            "sigma_qmdpa_logdistance sigma_qmdpa_ionization height\n"
            "#! SET action OPES_METAD_state\n#! SET biasfactor 5\n"
            "#! SET epsilon 0.01\n#! SET kernel_cutoff 4\n"
            "#! SET compression_threshold 1\n#! SET zed 0.5\n"
            "#! SET sum_weights 0.3\n#! SET sum_weights2 0.09\n"
            f"#! SET counter {1+step//2}\n{time:.17g} 0.1 0.2 0.15 0.04 0.3\n")


def test_opes_uses_exact_state_and_classical_cv(tmp_path):
    cfg, _ = configured(tmp_path)
    text = render_plumed(cfg["sampling"], tmp_path, restart=True)
    assert "OPES_METAD ARG=qmdpa_logdistance,qmdpa_ionization" in text
    assert "STATE_RFILE=" in text and "PARENT_STATE" in text
    assert "VORONOI_COORDINATION" in text and "VORONOI_DISTANCE" in text
    assert text.count("NOPBC") == 2
    assert "ENSEMBLE" not in text and "NLIST" not in text
    assert "STATE_WSTRIDE=2" in text and "RESTART=YES" in text


@pytest.mark.parametrize("key,value", [
    ("sigma", [0.1]), ("sigma_min", [1., 0.01]), ("pace_steps", True),
    ("pace_steps", 3), ("barrier_ev", float("nan")), ("bias_factor", 1),
])
def test_reject_invalid_opes_or_misaligned_history(tmp_path, key, value):
    cfg, raw = configured(tmp_path)
    raw["bias"][key] = value
    with pytest.raises(ValueError):
        bind_sampling(cfg, raw, tmp_path)


@pytest.mark.parametrize("key,value", [
    ("oxygen_ids", [1]), ("oxygen_ids", [1, 2]), ("oxygen_ids", [1, 99]),
    ("hydrogen_ids", [2, 3, 5, True]), ("hydrogen_ids", [2, 3, 5, 5]),
    ("coordination_kappa_per_angstrom", float("inf")),
])
def test_reject_wrong_water_species_charge_and_identity(tmp_path, key, value):
    cfg, raw = configured(tmp_path)
    raw["cv"][key] = value
    with pytest.raises(ValueError):
        bind_sampling(cfg, raw, tmp_path)


def test_cv_scalar_invariants_and_log_branch(tmp_path):
    cfg, _ = configured(tmp_path)
    cv = cfg["sampling"]["cv"]
    xyz = np.array([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.],
                    [2.8, 0., 0.], [3.76, 0., 0.], [2.56, 0.93, 0.]])
    values = water_cv.values(cv, dict(enumerate(xyz, 1)))
    shifted = water_cv.values(cv, dict(enumerate(xyz + [3., -2., 1.], 1)))
    assert shifted == pytest.approx(values, abs=1e-13)
    reordered = deepcopy(cv)
    reordered["oxygen_ids"].reverse()
    reordered["hydrogen_ids"].reverse()
    assert water_cv.values(reordered, dict(enumerate(xyz, 1))) == pytest.approx(values, abs=1e-13)
    assert water_cv.log_distance(1.) == 0.
    for delta in [1e-6, 1e-7]:
        slope = (water_cv.log_distance(1+delta)-water_cv.log_distance(1-delta))/(2*delta)
        assert slope == pytest.approx(1., abs=3e-7)
    xyz[1] = xyz[0]
    with pytest.raises(ValueError, match="distances"):
        water_cv.values(cv, dict(enumerate(xyz, 1)))


def test_exact_immutable_state_and_corruption(tmp_path):
    cfg, _ = configured(tmp_path)
    live = tmp_path / "STATE"
    live.write_text(state(cfg))
    record = opes.history_record(live, cfg, 2)
    checkpoint = {"frame": {"step": 2, "sampling": {"history": record}}}
    opes.validate_frame_history(checkpoint["frame"], cfg)
    opes.snapshot_history(checkpoint, tmp_path, live, cfg)
    validate_checkpoint_history(checkpoint, tmp_path / "checkpoint.json", cfg)
    accepted = tmp_path / checkpoint["sampling_history"]["file"]
    live.write_text(state(cfg, 4))
    validate_checkpoint_history(checkpoint, tmp_path / "checkpoint.json", cfg)
    accepted.write_text(accepted.read_text().replace("0.15 0.04", "0.16 0.04"))
    with pytest.raises(ValueError, match="checksum"):
        validate_checkpoint_history(checkpoint, tmp_path / "checkpoint.json", cfg)


@pytest.mark.parametrize("old,new", [
    ("counter 2", "counter 3"), ("OPES_METAD_state", "OPES_METAD_kernels"),
    ("height", "logweight"), ("zed 0.5", "zed nan"),
    ("0.15 0.04", "-0.15 0.04"), ("biasfactor 5", "biasfactor 6"),
])
def test_state_rejects_incomplete_or_wrong_history(tmp_path, old, new):
    cfg, _ = configured(tmp_path)
    live = tmp_path / "STATE"
    live.write_text(state(cfg).replace(old, new))
    with pytest.raises(ValueError):
        opes.history_record(live, cfg, 2)


def test_opes_refreshes_force_independently_of_sampling_interval(tmp_path):
    cfg, _ = configured(tmp_path)
    bias = object.__new__(NativePlumedBias)
    bias.config = cfg
    calls = []
    class Coupling:
        def run(self, count):
            calls.append(count)
    bias.advance(Coupling(), 4)
    assert calls == [1, 0]*4
