#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Water calibration, unified routes, accepted occupancy and fail-fast identity."""
import json
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from test_adaptive_config import document as legacy
from test_water_buffer import geometry, groups, switch

from qmdpa.config import validate_qm
from qmdpa.core import Evaluation, EvaluationError
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.inference import RAW_PROFILE
from qmdpa.water_buffer import HAMILTONIAN, SWITCH, frame_record, validate_buffer_frame
from qmdpa.water_reference import KIND, prepare_reference, reference_identity, reference_software


@pytest.fixture(autouse=True)
def numerical_versions(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "fixture")


def document():
    raw=legacy()
    raw.update(hamiltonian=HAMILTONIAN,core_group_ids=["core-a","core-b"],
               groups=[{"group_id":g.group_id,"atom_ids":list(g.atom_ids),"symbols":list(g.symbols)}
                       for g in groups()],
               full_region={"atom_ids":list(range(1,13)),"symbols":["O","H","H"]*4,
                            "molecules":[list(g.atom_ids) for g in groups()]},
               switch={"method":SWITCH,"center_atom_id":1,"candidate_anchors":[["a",7],["b",10]],
                       "inner_radius_angstrom":3.,"outer_radius_angstrom":5.},
               water_guard={"min_oh_angstrom":.6,"max_oh_angstrom":1.25,"min_foreign_oh_angstrom":1.45})
    p=raw["partitions"][0]
    raw["partitions"]=[dict(p,partition_id=str(i),qm_group_ids=["core-a","core-b"]+list(subset))
                       for i,subset in enumerate(switch().group_subsets)]
    qm=validate_qm(raw["qm"],provider=raw["provider"])
    raw["buffer_reference"]={
        "schema_version":1,"kind":KIND,"model_sha256":raw["model_sha256"],
        "qm":qm,"provider":raw["provider"],"inference_profile":RAW_PROFILE,
        "geometry_angstrom":geometry().positions[:3].tolist(),
        "high_energy_ev":-20.,"low_energy_ev":-18.75,
        "source":"analytic calibration fixture","software":reference_software()}
    return raw


def load(tmp_path,raw=None):
    (tmp_path/"model.pt").write_bytes(b"unevaluated model fixture")
    (tmp_path/"water.data").write_bytes(b"unevaluated data fixture")
    path=tmp_path/"run.json"
    path.write_text(json.dumps(document() if raw is None else raw))
    return load_run_config(path)


def frame(config):
    result={"step":0,"potential_eV":-20.,
            "atoms":[{"id":i+1,"x":list(x),"v":[0.,0.,0.],"f":[0.,0.,0.],"image":[0,0,0]}
                     for i,x in enumerate(geometry().positions)]}
    result["water_buffer"]=frame_record(result,config)
    return result


def test_unified_route_calibrates_every_partition_and_keeps_total_assembly(tmp_path):
    cfg=load(tmp_path)
    view=describe_run(cfg)
    assert view["coupling"]=={"surface":"ground","embedding":"mechanical","me_model":"water-buffer"}
    assert view["assembly"]=={"native_pair":"zero","external_quantity":"complete-total-energy-and-forces"}
    assert [p["energy_offset_ev"] for p in cfg["partitions"]]==[0.,1.25,1.25,2.5]
    assert len({p["offset_source"] for p in cfg["partitions"]})==1
    raw=document()
    del raw["hamiltonian"]
    raw["coupling"]={"surface":"ground","embedding":"mechanical","me_model":"water-buffer"}
    assert load(tmp_path,raw)["physics_sha256"]==cfg["physics_sha256"]


def test_family_order_is_canonical_but_water_reference_is_physical(tmp_path):
    cfg=load(tmp_path)
    raw=document()
    for key in ("groups","partitions","core_group_ids"):
        raw[key].reverse()
    raw["switch"]["candidate_anchors"].reverse()
    assert load(tmp_path,raw)["physics_sha256"]==cfg["physics_sha256"]
    raw["buffer_reference"]["high_energy_ev"]+=.01
    assert load(tmp_path,raw)["physics_sha256"]!=cfg["physics_sha256"]


@pytest.mark.parametrize("failure",["model","qm","source","family","offset","guard","occupancy","periodic","state"])
def test_invalid_contracts_fail_without_constructing_a_backend(tmp_path,failure):
    raw=document()
    if failure=="model":
        raw["buffer_reference"]["model_sha256"]="0"*64
    elif failure=="qm":
        raw["buffer_reference"]["qm"]["basis"]="changed"
    elif failure=="source":
        raw["buffer_reference"]["software"]["coupling_sha256"]="0"*64
    elif failure=="family":
        raw["partitions"].pop(0)
    elif failure=="offset":
        raw["partitions"][1]["energy_offset_ev"]=0
    elif failure=="guard":
        raw["water_guard"]["max_oh_angstrom"]=1.6
    elif failure=="occupancy":
        raw["switch"]["occupancy"]=1
    elif failure=="periodic":
        raw["periodic"]=True
    else:
        raw["provider"]={"engine":"gpu4pyscf","device":"cuda","state":{"method":"ground"}}
    with patch("qmdpa.backends.DeepMDBackend") as backend:
        with pytest.raises((ValueError,TypeError)):
            load(tmp_path,raw)
        backend.assert_not_called()


def test_accepted_occupancy_is_recomputed_from_coordinates_and_reference(tmp_path):
    cfg=load(tmp_path)
    original=frame(cfg)
    validate_buffer_frame(original,cfg)
    record=original["water_buffer"]
    assert record["buffer_water_counts"]==[0,1,1,2]
    assert record["reference_sha256"]==reference_identity(cfg["buffer_reference"])
    assert 0<record["expected_buffer_waters"]<2
    for key in ("weights","reference_sha256","expected_buffer_waters","active_partition_ids"):
        changed=deepcopy(original)
        if key=="weights":
            changed["water_buffer"][key][0]+=.01
        elif key=="reference_sha256":
            changed["water_buffer"][key]="0"*64
        elif key=="expected_buffer_waters":
            changed["water_buffer"][key]+=.01
        else:
            changed["water_buffer"][key]=[]
        with pytest.raises(ValueError):
            validate_buffer_frame(changed,cfg)
    changed=deepcopy(original)
    changed["atoms"][7]["x"][0]+=2.
    with pytest.raises(EvaluationError):
        validate_buffer_frame(changed,cfg)


def reference_request(tmp_path):
    raw=document()
    (tmp_path/"model.pt").write_bytes(b"unevaluated model fixture")
    request={key:raw[key] for key in ("model","model_sha256","qm","provider","inference_profile")}
    request.update(geometry_angstrom=geometry().positions[:3].tolist(),source="frozen unit fixture")
    path=tmp_path/"request.json"
    path.write_text(json.dumps(request))
    return path


@pytest.mark.parametrize("fail",[False,True])
def test_preparation_publishes_only_a_successful_bound_reference(tmp_path,fail):
    path=reference_request(tmp_path)
    output=tmp_path/"prepared"

    class Low:
        def evaluate(self,atoms):
            return Evaluation(-5.,np.zeros_like(atoms.positions))

    class High:
        def evaluate(self,atoms):
            if fail:
                raise EvaluationError("controlled provider failure")
            return Evaluation(-7.,np.zeros_like(atoms.positions))

    with patch("qmdpa.inference.prepare_raw_inference",return_value=object()), \
         patch("qmdpa.inference.strict_runtime_state",return_value={"profile":RAW_PROFILE}), \
         patch("qmdpa.backends.DeepMDBackend",return_value=Low()), \
         patch("qmdpa.providers.make_qm_provider",return_value=High()):
        if fail:
            with pytest.raises(EvaluationError):
                prepare_reference(path,output)
            assert not (output/"reference.json").exists()
            assert (output/"failure.json").is_file()
        else:
            result=prepare_reference(path,output)
            record=json.loads((output/"reference.json").read_text())
            assert result["delta_water_ev"]==-2.
            assert result["identity"]=="water-reference:"+reference_identity(record)
            with pytest.raises(FileExistsError):
                prepare_reference(path,output)
