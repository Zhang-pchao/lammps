#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Failure, acceptance and restart tests independent of a native installation."""

from copy import deepcopy

import numpy as np
import pytest
from test_named_cv import model

from qmdpa.config import file_digest
from qmdpa.external_cv import ExternalCVBias
from qmdpa.named_cv import CVFrame


class Kernel:
    def __init__(self, *args):
        self.updates = 0
        self.closed = False
        self.fail = None

    def cmd(self, key, value=None):
        if self.fail == key:
            raise ValueError("injected kernel failure")
        if key == "getBias":
            value.value = 0.
        if key == "update":
            self.updates += 1

    def close(self):
        self.closed = True


def setup(tmp_path, monkeypatch, *, restart=None, name="bias"):
    monkeypatch.setattr("qmdpa.external_cv._Plumed", Kernel)
    frame = CVFrame(*model(np.eye(3)))
    library = tmp_path/"library.so"
    library.write_bytes(b"test only, never loaded")
    identity = {"path": str(library), "sha256": file_digest(library)}
    params = {s.name: {"center": 0., "kappa": 1., "unit": s.unit} for s in frame.schema.variables}
    bias = ExternalCVBias(frame.schema, params, identity, identity, tmp_path/name,
                          timestep_fs=.05, restart=restart)
    return bias, frame


def test_only_accepted_steps_update_and_restore(tmp_path, monkeypatch):
    bias, frame = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        bias.checkpoint()
    for _ in range(3):
        bias.evaluate(frame, 0)
    assert bias.kernel.updates == 0
    bias.accept(frame, 0)
    bias.accept(frame, 0)
    assert bias.kernel.updates == 1
    record = bias.checkpoint()
    with pytest.raises(ValueError):
        bias.accept(frame, 2)
    with pytest.raises(ValueError):
        bias.accept(CVFrame(*model(np.eye(3)+.01)), 0)
    restored, frame = setup(tmp_path, monkeypatch, restart=record, name="restored")
    restored.accept(frame, 0)
    assert restored.kernel.updates == 0
    restored.accept(frame, 1)
    assert restored.kernel.updates == 1
    assert restored.accepted_count == 2
    bias.close()
    bias.close()
    with pytest.raises(ValueError):
        bias.evaluate(frame, 0)
    restored.close()


@pytest.mark.parametrize("key", ["identity_sha256", "step", "accepted_count", "frame_sha256"])
def test_corrupt_checkpoint_rejected(tmp_path, monkeypatch, key):
    bias, frame = setup(tmp_path, monkeypatch)
    bias.accept(frame, 0)
    record = deepcopy(bias.checkpoint())
    record[key] = None
    with pytest.raises(ValueError):
        setup(tmp_path, monkeypatch, restart=record, name="bad")
    bias.close()


def test_failed_evaluation_never_accepts(tmp_path, monkeypatch):
    bias, frame = setup(tmp_path, monkeypatch)
    bias.accept(frame, 0)
    record = bias.checkpoint()
    bias.kernel.fail = "performCalcNoUpdate"
    with pytest.raises(ValueError):
        bias.accept(frame, 1)
    assert bias.checkpoint() == record
    assert bias.kernel.updates == 1
    bias.close()


def test_dependency_change_rejected_on_restart(tmp_path, monkeypatch):
    from qmdpa.named_cv import digest
    bias, frame = setup(tmp_path, monkeypatch)
    bias.accept(frame, 0)
    record = deepcopy(bias.checkpoint())
    record["identity"]["schema"]["variables"][0]["dependencies"]["states"] = "b"*64
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(ValueError):
        setup(tmp_path, monkeypatch, restart=record, name="changed")
    bias.close()
