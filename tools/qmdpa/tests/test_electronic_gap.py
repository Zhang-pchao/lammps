#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Signed gap and native admission tests without electronic engines."""

from copy import deepcopy
import json

import numpy as np
import pytest
from test_excited_config import case

from qmdpa.config import file_digest
from qmdpa.core import Evaluation, Geometry
from qmdpa.dispatch import describe_run, load_run_config
from qmdpa.electronic_gap import FORM, gap_frame
from qmdpa.named_cv import NamedCVSchema

__all__ = ["case"]


def extension(tmp_path, configuration):
    q = configuration.get("region", configuration.get("qm_region"))
    e = configuration.get("environment_region", {"atom_ids": [4, 5, 6], "symbols": ["O", "H", "H"], "molecules": [[4, 5, 6]]})
    path = tmp_path/"plumed.so"
    path.write_bytes(b"not a real shared library")
    library = {"path": str(path), "sha256": file_digest(path)}
    return {"form": FORM, "name": "excitation_gap", "order": ["ground", "selected"],
            "sampled_surface": "ground", "full_region": {k: q[k]+e[k] for k in q},
            "restraint": {"center": 3., "kappa": .2, "unit": "eV"},
            "kernel": library, "wrapper": library}


def write(tmp_path, raw):
    path = tmp_path/"gap.json"
    path.write_text(json.dumps(raw))
    return path


def config_from_case(case, tmp_path):
    # Existing fixture returns (embedded, reference, configuration).
    raw = deepcopy(case[2])
    raw["sample_interval"] = 1
    raw["electronic_gap"] = extension(tmp_path, raw)
    return raw


def test_admission_and_transport(case, tmp_path):
    raw = config_from_case(case, tmp_path)
    cfg = load_run_config(write(tmp_path, raw))
    assert describe_run(cfg)["assembly"]["native_pair"] == "zero"
    changed = deepcopy(raw)
    changed["electronic_gap"]["order"].reverse()
    assert load_run_config(write(tmp_path, changed))["physics_sha256"] != cfg["physics_sha256"]
    changed = deepcopy(raw)
    changed["electronic_gap"]["sampled_surface"] = "selected"
    assert load_run_config(write(tmp_path, changed))["physics_sha256"] != cfg["physics_sha256"]


@pytest.mark.parametrize("change", ["order", "unit", "tensor_name", "sample", "source", "surface", "ground", "extra"])
def test_invalid_gap_rejected(case, tmp_path, change):
    raw = config_from_case(case, tmp_path)
    ext = raw["electronic_gap"]
    if change == "order": ext["order"] = ["ground", "ground"]
    if change == "unit": ext["restraint"]["unit"] = "hartree"
    if change == "tensor_name": ext["name"] = "not.a.scalar"
    if change == "sample": raw["sample_interval"] = 2
    if change == "source": ext["kernel"]["sha256"] = "0"*64
    if change == "surface": ext["sampled_surface"] = "adiabatic"
    if change == "ground": raw["provider"]["state"] = {"method": "ground"}
    if change == "extra": ext["fallback"] = True
    with pytest.raises((ValueError, TypeError)):
        load_run_config(write(tmp_path, raw))


def test_signed_gap_chain_rule_and_mapping():
    schema = NamedCVSchema({"schema_version": 1, "coordinate_unit": "angstrom", "variables": [
        {"name": "gap", "shape": [], "unit": "eV", "dependencies": {"states": "a"*64}}]})
    xyz = np.array([[.2, -.1, .3], [.4, .2, -.7]])

    def evaluate(x, order):
        geo = Geometry([13, 7], ["H", "H"], x)
        states = {"ground": Evaluation(float(np.sum(x*x)), -2*x),
                  "selected": Evaluation(float(np.sum(x**4))-.3, -4*x**3)}
        return gap_frame(schema, geo, states, order)

    forward = evaluate(xyz, ["ground", "selected"])
    reverse = evaluate(xyz, ["selected", "ground"])
    np.testing.assert_array_equal(reverse.values, -forward.values)
    np.testing.assert_array_equal(reverse.jacobian, -forward.jacobian)
    center, kappa = .7, .2
    force = forward.pullback({"gap": kappa*(forward.values[0]-center)})
    np.testing.assert_array_equal(force, reverse.pullback({"gap": kappa*(reverse.values[0]+center)}))
    for i, j in np.ndindex(xyz.shape):
        energies = []
        for sign in (-1, 1):
            x = xyz.copy(); x[i,j] += sign*1e-5
            energies.append(.5*kappa*(evaluate(x, ["ground", "selected"]).values[0]-center)**2)
        assert abs(force[i,j]+(energies[1]-energies[0])/2e-5) < 1e-8


@pytest.mark.parametrize("embedded", [False, True])
def test_same_frame_sharing_and_failure_clears_every_record(monkeypatch, embedded):
    from qmdpa.electronic_gap import EE, ME, GroundTDARestraint
    from qmdpa.embedding import EmbeddedEvaluation

    geometry = Geometry([91, 12, 7], ["H", "H", "O"], [[.2, .3, .4], [1., .2, .3], [3., .7, .9]])
    qids, eids = [91, 12], [7]
    full = {"atom_ids": [91, 12, 7], "symbols": ["H", "H", "O"], "molecules": [[91, 12], [7]]}
    qregion = {"atom_ids": qids, "symbols": ["H", "H"], "molecules": [qids]}
    config = {"hamiltonian": EE if embedded else ME,
        "qm_region" if embedded else "region": qregion,
        "environment_region": {"atom_ids": eids, "symbols": ["O"], "molecules": [eids]},
        "electronic_gap": {"full_region": full, "order": ["ground", "selected"], "sampled_surface": "ground"}}
    schema = NamedCVSchema({"schema_version": 1, "coordinate_unit": "angstrom", "variables": [
        {"name": "gap", "shape": [], "unit": "eV", "dependencies": {"pair": "a"*64}}]})
    monkeypatch.setattr("qmdpa.electronic_gap.schema_for", lambda _: schema)

    class Backend:
        def __init__(self, factor):
            self.factor, self.calls = factor, 0
        def evaluate(self, qm, env=None):
            self.calls += 1
            energy = self.factor*float(np.sum(qm.positions**2))
            if env is None:
                return Evaluation(energy, -2*self.factor*qm.positions)
            return EmbeddedEvaluation(energy+self.factor*float(np.sum(env.positions**2)),
                qm.atom_ids, env.atom_ids, -2*self.factor*qm.positions, -2*self.factor*env.positions, {})

    class Selected(Backend):
        def __init__(self):
            super().__init__(.7)
            self.fail = False
            self.offset = 0.
            self.clear()
        def clear(self):
            self.ground_energy = None
        def evaluate(self, qm, env=None):
            if self.fail: raise ValueError("state lost")
            self.ground_energy = .5*float(np.sum(qm.positions**2))
            if env is not None: self.ground_energy += .5*float(np.sum(env.positions**2))
            return super().evaluate(qm, env)
        def accepted_record(self, frame):
            if self.ground_energy is None: raise ValueError("no state record")
            return {"diagnostics": {"ground_energy_ev": self.ground_energy+self.offset}}

    ground, selected, low, cross, full_low = Backend(.5), Selected(), Backend(.2), Backend(.1), Backend(.3)
    calc = GroundTDARestraint(config, ground, selected, low, full_low=full_low, cross=cross)
    calc.evaluate(geometry)
    assert low.calls == 1 and ground.calls == selected.calls == 1
    assert (cross.calls if embedded else full_low.calls) == 1
    cv = calc.last[2]
    env_rows = [geometry.atom_ids.index(i) for i in eids]
    if not embedded: np.testing.assert_array_equal(cv.jacobian[0, env_rows], 0.)
    selected.fail = True
    with pytest.raises(ValueError, match="state lost"): calc.evaluate(geometry)
    assert calc.last is None and selected.ground_energy is None
    selected.fail = False
    selected.offset = 1e-5
    with pytest.raises(ValueError, match="baselines differ"): calc.evaluate(geometry)
    assert calc.last is None and selected.ground_energy is None
    selected.offset = 0.
    calc.evaluate(geometry.select([7,91,12]))
    np.testing.assert_allclose(calc.last[2].jacobian, cv.jacobian[:,[2,0,1]], atol=0, rtol=0)


def test_native_acceptance_is_pure_until_valid_and_restart_keeps_history(case, tmp_path, monkeypatch):
    import re
    from types import SimpleNamespace
    from test_external_cv import Kernel
    from qmdpa.electronic_gap import schema_for
    from qmdpa.electronic_gap_runtime import capture, validate_frame
    from qmdpa.external_cv import ExternalCVBias
    from qmdpa.named_cv import digest

    class HarmonicKernel(Kernel):
        def cmd(self, key, value=None):
            if key == "readInputLine" and "RESTRAINT" in value:
                self.center=float(re.search(r"AT=(\S+)",value)[1])
                self.kappa=float(re.search(r"KAPPA=(\S+)",value)[1])
            if key.startswith("setExtraCV "): self.values=value
            if key.startswith("setExtraCVForce "): self.forces=value
            if key == "performCalcNoUpdate": self.forces[:]=-self.kappa*(self.values-self.center)
            super().cmd(key,value)
            if key == "getBias": value.value=.5*self.kappa*float(self.values[0]-self.center)**2

    monkeypatch.setattr("qmdpa.external_cv._Plumed",HarmonicKernel)
    raw=config_from_case(case,tmp_path);cfg=load_run_config(write(tmp_path,raw))
    ext=cfg['electronic_gap'];schema=schema_for(cfg);full=ext['full_region']
    xyz=np.arange(len(full['atom_ids'])*3).reshape(-1,3)*.03
    geometry=Geometry(full['atom_ids'],full['symbols'],xyz)
    states={'ground':Evaluation(float(np.sum(xyz**2)),-2*xyz),
            'selected':Evaluation(float(2*np.sum(xyz**2))+.1,-4*xyz)}
    cv=gap_frame(schema,geometry,states,ext['order'])
    library=lambda k:{'path':cfg['electronic_gap_libraries'][k],**ext[k]}
    bias=ExternalCVBias(schema,{ext['name']:ext['restraint']},library('kernel'),library('wrapper'),tmp_path/'bias',timestep_fs=cfg['timestep_fs'])
    try:
        result=bias.evaluate(cv,0)
        total=Evaluation(states['ground'].energy+result['bias_energy_eV'],
                         states['ground'].forces+np.asarray(result['forces_eV_per_angstrom']))
        calculator=SimpleNamespace(bias=bias,last=(geometry,states,cv,result,total,0.))
        coupling=SimpleNamespace(calculator=calculator,failure=None,last_step=0)
        frame={'step':0,'potential_eV':total.energy,'atoms':[
            {'id':i,'x':x.tolist(),'image':[0,0,0],'f':f.tolist()}
            for i,x,f in zip(geometry.atom_ids,xyz,total.forces,strict=True)]}
        wrong=deepcopy(frame);wrong['atoms'][0]['f'][0]+=.1
        with pytest.raises(ValueError,match='native total forces'):capture(wrong,coupling,cfg)
        assert bias.accepted_count==0 and bias.kernel.updates==0
        capture(frame,coupling,cfg)
        capture(frame,coupling,cfg)
        assert bias.accepted_count==1 and bias.kernel.updates==1
        history=deepcopy(frame['electronic_gap']['history'])
        continued=deepcopy(frame)
        capture(continued,coupling,cfg,previous={'frame':frame})
        assert continued['electronic_gap']['history']==history
        assert bias.accepted_count==1 and bias.kernel.updates==1
        for key in ('history','cv','bias_forces_eV_per_angstrom'):
            bad=deepcopy(frame);item=bad['electronic_gap']
            if key=='history':
                item['history']['accepted_count']=2
                item['history']['sha256']=digest({k:v for k,v in item['history'].items() if k!='sha256'})
            elif key=='cv':item['cv']['values'][0]+=.01
            else:item[key][0][0]+=.01
            item['sha256']=digest({k:v for k,v in item.items() if k!='sha256'})
            with pytest.raises(ValueError):validate_frame(bad,cfg)
        continued['step']=1;coupling.last_step=1
        capture(continued,coupling,cfg)
        assert bias.accepted_count==2 and bias.kernel.updates==2
    finally:bias.close()
