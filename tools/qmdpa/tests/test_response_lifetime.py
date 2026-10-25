# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Response callbacks release workspaces even before cyclic collection runs."""
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import test_excited

from qmdpa import excited, pbe0_tda
from qmdpa.core import EvaluationError


class Workspace:
    pass


class OfficialGradient:
    observed = []

    def __init__(self, td):
        self.base = td
        if td.failure == "constructor":
            raise EvaluationError("Injected gradient constructor failure")

    def solve_cphf(self, fvind, mo_energy, mo_occ, h1):
        return -h1, None

    def extra_force(self, atom, response):
        return np.full(3, 0.25)

    def kernel(self, state):
        if self.base.failure == "before-hook":
            raise EvaluationError("Injected gradient failure before response")
        hook = test_excited.response_hook()
        self.observed.extend(weakref.ref(hook[name]) for name in
                             ("oo0", "dmz1doo", "wvo", "mo_energy"))
        if self.base.refine:
            hook["z1"] = self.solve_cphf(hook["fvind"], hook["mo_energy"],
                                       np.array([2, 0]), hook["wvo"])[0]
        values = [self.extra_force(atom, hook) for atom in range(2)]
        if self.base.failure == "after-hook":
            raise EvaluationError("Injected gradient failure after response")
        return np.asarray(values)


def exercise(method, failure, handles):
    workspace = Workspace()
    workspace.grid = np.ones((1024, 3))
    solved = SimpleNamespace(
        method=workspace, overlap=np.eye(2),
        molecule=SimpleNamespace(natm=2, nelectron=2),
        td=SimpleNamespace(method=workspace, failure=failure, refine=method == "rhf"),
    )
    handles.extend((weakref.ref(workspace), weakref.ref(workspace.grid),
                    weakref.ref(solved.overlap)))
    runtime = SimpleNamespace(tdrhf=SimpleNamespace(Gradients=OfficialGradient),
                              tdrks=SimpleNamespace(Gradients=OfficialGradient))
    try:
        if method == "pbe0":
            return pbe0_tda.PySCFPBE0TDABackend._surface_gradient(
                None, solved, 0, pbe0_tda.PBE0TDASettings(), runtime)
        return excited._total_gradient(solved, 0, test_excited.settings(), runtime)
    except EvaluationError as error:
        # Return no traceback/frame: external exception retention has its own
        # lifetime and is not the callback's successful-result contract.
        if failure is None or "Injected gradient" not in str(error):
            raise
        return None


@pytest.mark.parametrize("method", ["rhf", "pbe0"])
@pytest.mark.parametrize("failure", [None, "constructor", "before-hook", "after-hook"])
def test_workspaces_are_released_without_forcing_global_collection(monkeypatch, method, failure):
    handles = []
    monkeypatch.setattr(OfficialGradient, "observed", handles)
    monkeypatch.setattr(pbe0_tda, "_pbe0_grid_response",
                        lambda *args: (np.full((2, 3), 0.1), {"grid_test": True}))
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = exercise(method, failure, handles)
        assert handles and all(handle() is None for handle in handles)
        if failure is None:
            gradient, density, diagnostics = result
            np.testing.assert_allclose(gradient, 0.35 if method == "pbe0" else 0.25)
            np.testing.assert_allclose(density, [[1.8, 0.2], [0.2, 0.2]])
            assert diagnostics["density_trace"] == 2.0
            assert diagnostics["cphf_residual_max"] == 0.0
        else:
            assert result is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()
