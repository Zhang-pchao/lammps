# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Matched PBE0 ground terms cannot silently change the gap-anchor baseline."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_excited import qm, reference
from test_pbe0_tda import pbe0_reference

from qmdpa import pbe0_tda as api
from qmdpa.core import EvaluationError
from qmdpa.gap_anchor import make_pbe0_gap_anchor


@pytest.mark.parametrize("profile", [("def2-svp", 4), ("sto-3g", 1)])
@pytest.mark.parametrize("embedded", [False, True])
def test_ground_uses_exact_state_settings_with_explicit_prune_nlc_and_response(
    monkeypatch, embedded, profile
):
    ref = pbe0_reference(embedded)
    if profile[0] == "sto-3g":
        from test_pbe0_development import small_reference

        ref = small_reference(ref)
    prune = object()

    def initialize(self, **options):
        self.__dict__.update(options)
        self._dft = SimpleNamespace(gen_grid=SimpleNamespace(nwchem_prune=prune))

    def embedded_initialize(self, charges, **options):
        initialize(self, **options)
        self.charges = charges

    monkeypatch.setattr(api, "_load_runtime", lambda settings: SimpleNamespace(record=ref.runtime))
    monkeypatch.setattr(api.PySCFBackend, "__init__", initialize)
    monkeypatch.setattr(api.PySCFElectrostaticBackend, "__init__", embedded_initialize)
    method = SimpleNamespace(grids=SimpleNamespace(prune=None), nlc="unexpected")
    monkeypatch.setattr(api.PySCFBackend, "_make_method", lambda self, geometry: method)
    backend = (
        api.PySCFPBE0ElectrostaticGroundBackend(ref.settings, ref.charges)
        if embedded
        else api.PySCFPBE0GroundBackend(ref.settings)
    )
    assert backend._runtime.record == ref.runtime
    assert backend.conv_tol == 1e-12 and backend.conv_tol_grad == 1e-8
    assert backend.xc == "pbe0" and backend.basis == profile[0]
    assert backend.grid_level == profile[1] and backend.grid_response is True
    built = backend._make_method(qm())
    assert built.nlc == "" and built.grids.prune is prune
    method.with_df = object()
    with pytest.raises(EvaluationError, match="density fitting"):
        backend._make_method(qm())
    method.with_df = None
    backend.grid_level = 3
    with pytest.raises(EvaluationError, match="settings changed"):
        backend._make_method(qm())


def test_wrong_ground_settings_or_charge_shape_reject_before_engine(monkeypatch):
    monkeypatch.setattr(api, "_load_runtime", lambda settings: pytest.fail("Unexpected runtime"))
    with pytest.raises(TypeError):
        api.PySCFPBE0GroundBackend(reference().settings)
    ref = pbe0_reference(True)
    with pytest.raises(ValueError):
        api.PySCFPBE0ElectrostaticGroundBackend(
            ref.settings, replace(ref.charges, radii_angstrom=[0.1, 0.1])
        )


def test_anchor_constructs_all_terms_from_one_pinned_pbe0_reference():
    ref = pbe0_reference(True)
    with (
        patch.object(api, "PySCFPBE0GroundBackend") as ground,
        patch.object(api, "PySCFPBE0ElectrostaticGroundBackend") as embedded,
        patch.object(api, "PySCFPBE0TDAElectrostaticBackend") as selected,
    ):
        embedded.return_value.charges = selected.return_value.charges = ref.charges
        calc = make_pbe0_gap_anchor(object(), object(), ref, reuse_identical_geometry=True)
        ground.assert_called_once_with(ref.settings)
        embedded.assert_called_once_with(ref.settings, ref.charges)
        selected.assert_called_once_with(
            ref.settings,
            ref,
            ref.charges,
            reuse_identical_geometry=True,
            matrix_assembly="operator",
        )
        assert calc.qm_atom_ids == ref.qm_geometry.atom_ids
        assert calc.environment_atom_ids == ref.environment_geometry.atom_ids


@pytest.mark.parametrize("ref", [reference(), reference(embedded=True), pbe0_reference()])
def test_anchor_rejects_wrong_method_or_missing_field(ref):
    with pytest.raises(ValueError, match="embedded PBE0"):
        make_pbe0_gap_anchor(object(), object(), ref)
