# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Matched small-profile fragment inputs preserve model and state boundaries."""

from dataclasses import asdict
from unittest.mock import patch

import pytest
from test_fragment_ground_native import ground_request
from test_fragment_ground_native import load as load_ground
from test_fragment_response_native import select as select_fragment

# Register shared pytest fixtures consumed through test function arguments.
from test_gap_anchor_native import config, load
from test_interaction_delta_native import select as select_cross
from test_pbe0_development import small_reference
from test_state_inputs import file_provider

from qmdpa import dispatch
from qmdpa.fragment_ground import _ground_settings

# Explicit fixture exports keep registration visible to static tools.
__all__ = ['config']


def small_selected(config, *, fragment):
    refs = select_fragment(config) if fragment else (select_cross(config),)
    small = tuple(small_reference(ref) for ref in refs)
    config["qm"] = asdict(small[0].settings)
    config["provider"]["state"]["reference"] = small[0].to_record()
    if fragment:
        gap = config["fragment_response"]["cluster_gap"]
        for name, ref in zip(("pair", "field"), small[1:]):
            gap[name + "_provider"]["state"]["reference"] = ref.to_record()
        gap["state_correspondence"].update(
            anchor_sha256=small[0].sha256,
            pair_sha256=small[1].sha256,
            field_sha256=small[2].sha256,
        )
    return refs, small


@pytest.mark.parametrize("reuse", [False, True])
def test_small_ground_uses_shared_profile_and_new_restart_identity(tmp_path, config, reuse):
    raw = ground_request(config)
    raw["reuse_ground_terms"] = reuse
    old = load_ground(tmp_path, raw)
    raw["qm"].update(basis="sto-3g", grid_level=1)
    new = load_ground(tmp_path, raw)
    settings = _ground_settings(new["qm"])
    assert settings.basis == "sto-3g" and settings.grid_level == 1
    assert settings.grid_response is True
    assert old["physics_sha256"] != new["physics_sha256"]
    assert old["assembly"] == new["assembly"]
    raw.pop("hamiltonian")
    raw["coupling"] = {
        "surface": "ground",
        "embedding": "mechanical",
        "me_model": "fragment-response",
    }
    assert load_ground(tmp_path, raw) == new


@pytest.mark.parametrize("fragment", [False, True])
def test_selected_small_profile_keeps_total_and_all_reference_files(tmp_path, config, fragment):
    old, small = small_selected(config, fragment=fragment)
    config["reuse_ground_terms"] = True
    if fragment:
        config["reduce_fragment_terms"] = True
    expected = load(tmp_path, config)
    assert expected["assembly"]["pair_style"] == "zero"
    assert expected["assembly"]["ground_method"] == "rks-xc-pbe0"
    assert small[0].sha256 != old[0].sha256
    config.pop("hamiltonian")
    config["coupling"] = {
        "surface": "selected",
        "embedding": "electrostatic",
        "ee_model": "gap-anchor-fragment-response" if fragment else "gap-anchor-cross-delta",
    }
    config["provider"] = file_provider(tmp_path, config["provider"], "anchor")
    if fragment:
        gap = config["fragment_response"]["cluster_gap"]
        for name in ("pair", "field"):
            gap[name + "_provider"] = file_provider(tmp_path, gap[name + "_provider"], name)
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        from test_dispatch import save

        assert dispatch.load_run_config(save(tmp_path, config)) == expected


@pytest.mark.parametrize("side", ["pair", "field"])
def test_mixed_small_legacy_references_reject_before_backend(tmp_path, config, side):
    old, small = small_selected(config, fragment=True)
    gap = config["fragment_response"]["cluster_gap"]
    legacy = old[1 if side == "pair" else 2]
    gap[side + "_provider"]["state"]["reference"] = legacy.to_record()
    gap["state_correspondence"][side + "_sha256"] = legacy.sha256
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        pytest.raises(ValueError, match="settings or runtime"),
    ):
        load(tmp_path, config)
    low.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"basis": "sto-3g"},
        {"grid_level": 1},
        {"basis": "sto-3g", "grid_level": 1, "grid_response": False},
        {"basis": "sto-3g", "grid_level": 1, "conv_tol": 1e-6},
    ],
)
def test_small_ground_does_not_weaken_or_infer_profile(tmp_path, config, change):
    raw = ground_request(config)
    raw["qm"].update(change)
    with pytest.raises(ValueError):
        load_ground(tmp_path, raw)


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("side", ["pair", "field"])
def test_auxiliary_files_match_inline_and_reject_changed_contents(tmp_path, config, small, side):
    if small:
        small_selected(config, fragment=True)
    else:
        select_fragment(config)
    expected = load(tmp_path, config)
    gap = config["fragment_response"]["cluster_gap"]
    gap[side + "_provider"] = file_provider(tmp_path, gap[side + "_provider"], side + ".json")
    assert load(tmp_path, config) == expected
    (tmp_path / (side + ".json")).write_text("{}")
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        pytest.raises(ValueError, match="SHA256 mismatch"),
    ):
        load(tmp_path, config)
    low.assert_not_called()
