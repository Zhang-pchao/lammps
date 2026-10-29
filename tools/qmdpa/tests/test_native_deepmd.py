#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Native transport mapping and checked ownership without a backend fallback."""

import hashlib

import numpy as np
import pytest

from qmdpa.core import Geometry
from qmdpa.native_deepmd import NativeDeepMDDomainBackend
from qmdpa.runtime import ProviderResources


class NativeStub:
    def __init__(self):
        self.commands = []
        self.style = "deepmd/kk"
        self.closed = False
        self.lost = False

    def command(self, text):
        self.commands.append(text)

    def commands_list(self, texts):
        for text in texts:
            self.command(text)

    def extract_global(self, key):
        assert key == "pair_style"
        return self.style

    def create_atoms(self, count, ids, types, xyz):
        self.ids, self.types = np.array(ids), np.array(types)
        self.xyz = np.array(xyz)
        return count-int(self.lost)

    def get_natoms(self):
        return len(self.ids)

    def gather_atoms(self, field, dtype, count):
        return {"id": self.ids, "type": self.types, "f": -self.xyz}[field]

    def get_thermo(self, field):
        assert field == "pe"
        return .5*float(self.xyz @ self.xyz)

    def close(self):
        assert not self.closed
        self.closed = True


def backend(tmp_path, monkeypatch):
    p=tmp_path/"model.pt2";p.write_bytes(b"native archive fixture")
    native=NativeStub()
    monkeypatch.setattr("qmdpa.native_deepmd._create_lammps",lambda style:native)
    obj=NativeDeepMDDomainBackend(p,model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                type_symbols=("O","H"),pair_style="deepmd/kk")
    return obj,native


def test_domain_resize_reordering_and_resource_cleanup(tmp_path,monkeypatch):
    obj,native=backend(tmp_path,monkeypatch)
    owner=ProviderResources(tmp_path/"unused")
    owner.register(obj)
    small=Geometry((30,10,20),("H","O","H"),[[.1,1.,0.],[0.,0.,0.],[1.,0.,0.]])
    large=Geometry((60,10,30,40,20,50),("H","O","H","O","H","H"),
                   [[3.,1.,0.],[0.,0.,0.],[.1,1.,0.],[3.,0.,0.],[1.,0.,0.],[4.,0.,0.]])
    first=obj.evaluate(small)
    for g in (large,small):
        value=obj.evaluate(g)
        np.testing.assert_array_equal(value.forces,-g.positions)
    assert obj.evaluate(small).energy == first.energy
    assert len([x for x in native.commands if x.startswith("pair_style")]) == 1
    assert len([x for x in native.commands if x.startswith("delete_atoms")]) == 4
    assert len(obj.statistics["domains"]) == 2
    assert not owner.close()
    assert not owner.close()
    assert native.closed and obj.statistics["closed"]
    with pytest.raises(RuntimeError,match="cleanup"):
        obj.evaluate(small)


@pytest.mark.parametrize("failure",["style","atom-loss"])
def test_failures_are_reported_and_owned(tmp_path,monkeypatch,failure):
    obj,native=backend(tmp_path,monkeypatch)
    if failure=="style":
        native.style="zero"
    else:
        native.lost=True
    g=Geometry((1,),("O",),[[0.,0.,0.]])
    with pytest.raises((RuntimeError,ValueError),match="pair|lost"):
        obj.evaluate(g)
    assert obj.statistics["successful_calls"] == 0
    assert obj.statistics["failed_calls"] == 1
    obj.close_checked()
    assert native.closed


def test_input_identity_and_geometry_boundaries(tmp_path,monkeypatch):
    obj,native=backend(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="identified"):
        NativeDeepMDDomainBackend(obj.model,model_sha256="0"*64,
                                 type_symbols=("O","H"),pair_style="deepmd/kk")
    with pytest.raises(ValueError,match="finite"):
        obj.evaluate(Geometry((1,),("O",),[[0.,0.,0.]],cell=np.eye(3)*10))
    with pytest.raises(ValueError,match="element"):
        obj.evaluate(Geometry((1,),("C",),[[0.,0.,0.]]))
    assert not native.commands
    obj.close_checked()
