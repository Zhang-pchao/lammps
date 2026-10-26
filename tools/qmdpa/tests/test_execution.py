#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit practical precision and native pair admission contracts."""
import hashlib
import json
import zipfile

import pytest
from test_dispatch import mechanical, save
from test_sampling import thermostat

from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.execution import PRACTICAL, native_arguments, restart_limits, verify_native


def request(tmp_path, style="deepmd/kk", dtype="float32", kind="dpa4c_canonical"):
    raw=mechanical(tmp_path)
    path=tmp_path/"model.pt2"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("model/extra/metadata.json",json.dumps({"lower_input_kind":kind}))
        archive.writestr("model/extra/model_def_script.json",json.dumps({
            "descriptor":{"type":"dpa4c","precision":dtype,"use_amp":False},
            "fitting_net":{"precision":dtype}}))
    raw["model"]=path.name
    raw["model_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    raw["execution"]={"pair_style":style,"numerical_profile":PRACTICAL}
    return raw


@pytest.mark.parametrize("style",["deepmd","deepmd/kk"])
def test_explicit_pair_and_numerical_identity(tmp_path,style):
    raw=request(tmp_path,style)
    cfg=load_run_config(save(tmp_path,raw))
    assert describe_run(cfg)["assembly"]["native_pair"]==style
    assert restart_limits(cfg)[:2]==(1e-4,1e-3)
    assert ("-k" in native_arguments(cfg)) == (style=="deepmd/kk")
    raw.pop("execution")
    old=load_run_config(save(tmp_path,raw))
    assert old["physics_sha256"]!=cfg["physics_sha256"]
    assert restart_limits(old)[:2]==(1e-6,1e-5)
    assert "-k" not in native_arguments(old)


@pytest.mark.parametrize("change",[
    {"pair_style":"deepmd/gpu"},{"numerical_profile":"auto"},{"gpus":4},
])
def test_undeclared_execution_fails(tmp_path,change):
    raw=request(tmp_path)
    raw["execution"].update(change)
    with pytest.raises(ValueError):
        load_run_config(save(tmp_path,raw))


@pytest.mark.parametrize("dtype,kind",[("float64","graph"),("mixed","graph"),("float32","nlist")])
def test_only_identified_float32_graphs_are_admitted(tmp_path,dtype,kind):
    with pytest.raises(ValueError):
        load_run_config(save(tmp_path,request(tmp_path,dtype=dtype,kind=kind)))


@pytest.mark.parametrize("key,value",[
    ("periodic",True),("full_charge_spin",[1,2]),("region_charge_spin",[1,2]),
])
def test_scope_and_charge_guards(tmp_path,key,value):
    raw=request(tmp_path)
    raw[key]=value
    with pytest.raises(ValueError):
        load_run_config(save(tmp_path,raw))


def test_sampling_extensions_remain_separate(tmp_path):
    raw=request(tmp_path)
    raw["thermostat"]=thermostat()
    with pytest.raises(ValueError,match="NVE"):
        load_run_config(save(tmp_path,raw))


def test_native_pair_does_not_silently_fallback(tmp_path):
    cfg=load_run_config(save(tmp_path,request(tmp_path)))
    class WrongPair:
        def extract_global(self,name):
            return "deepmd"
    with pytest.raises(ValueError,match="actual native pair"):
        verify_native(WrongPair(),cfg)


def test_kokkos_cleanup_is_deferred_and_registered_once_per_library():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from qmdpa import execution

    finalize = Mock()
    driver = SimpleNamespace(lib=SimpleNamespace(_handle=901, lammps_kokkos_finalize=finalize))
    cfg = {"execution": {"pair_style": "deepmd/kk"}}
    with patch.object(execution, "_CLEANUP_HANDLES", set()), patch.object(execution.atexit, "register") as register:
        execution.register_native_cleanup(driver, cfg)
        execution.register_native_cleanup(driver, cfg)
        register.assert_called_once_with(finalize)
        finalize.assert_not_called()
        assert finalize.argtypes == []
        assert finalize.restype is None


def test_ordinary_execution_does_not_take_runtime_ownership():
    from unittest.mock import Mock, patch

    from qmdpa import execution

    with patch.object(execution.atexit, "register") as register:
        execution.register_native_cleanup(Mock(), {})
        register.assert_not_called()


def test_legacy_bias_cannot_bypass_practical_scope(tmp_path):
    from qmdpa.execution import bind_execution

    cfg = load_run_config(save(tmp_path, request(tmp_path)))
    cfg["bias"] = {}
    with pytest.raises(ValueError, match="NVE"):
        bind_execution(cfg, cfg["execution"])


@pytest.mark.parametrize("field", ["descriptor", "fitting_net"])
def test_archive_structure_rejects_non_objects(tmp_path, field):
    raw = request(tmp_path)
    path = tmp_path / "model.pt2"
    model = {"descriptor": {"type": "dpa4c", "precision": "float32"},
             "fitting_net": {"precision": "float32"}}
    model[field] = []
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/extra/metadata.json", json.dumps({"lower_input_kind": "graph"}))
        archive.writestr("model/extra/model_def_script.json", json.dumps(model))
    raw["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TypeError, match="must contain objects"):
        load_run_config(save(tmp_path, raw))
