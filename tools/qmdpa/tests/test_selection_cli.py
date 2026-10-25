# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Offline report provenance, frame identity, and non-destructive output."""

import hashlib
import json

import pytest

from qmdpa.selection_cli import main


def frame():
    return {
        "units": "angstrom",
        "boundary": "nonperiodic",
        "atom_ids": [1, 2, 3, 4, 5, 6],
        "symbols": ["O", "H", "H", "O", "H", "H"],
        "positions": [
            [0, 0, 0],
            [0.75, 0.55, 0],
            [-0.75, 0.55, 0],
            [3, 0, 0],
            [3.75, 0.55, 0],
            [2.25, 0.55, 0],
        ],
    }


def run(tmp_path, document):
    source, destination = tmp_path / "frames.json", tmp_path / "selection.json"
    source.write_text(json.dumps(document))
    argv = [
        "--input",
        str(source),
        "--output",
        str(destination),
        "--centers",
        "1",
        "--oxygen-count",
        "1",
    ]
    return source, destination, argv


def test_single_frame_hash_and_no_overwrite(tmp_path):
    source, destination, argv = run(tmp_path, frame())
    before = source.read_bytes()
    assert main(argv) == 0
    result = json.loads(destination.read_text())
    assert result["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["selector"]["center_oxygen_ids"] == [1]
    assert result["counts"] == {"frames": 1, "selected": 1, "unresolved": 0}
    assert result["frames"][0]["selected_atom_ids"] == [1, 2, 3]
    saved = destination.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert destination.read_bytes() == saved
    assert source.read_bytes() == before


def test_independent_frames_report_shared_proton_without_reusing_previous_ids(tmp_path):
    first, second = frame(), frame()
    second["positions"][3:] = [[2.4, 0, 0], [3.15, 0.55, 0], [1.65, 0.55, 0]]
    second["positions"][1] = [1.2, 0, 0]
    _, destination, argv = run(tmp_path, {"frames": [first, second, first]})
    assert main(argv) == 0
    result = json.loads(destination.read_text())
    assert result["counts"] == {"frames": 3, "selected": 2, "unresolved": 1}
    assert result["frames"][1]["selected_atom_ids"] == []
    assert result["frames"][0]["selected_atom_ids"] == result["frames"][2]["selected_atom_ids"]


def test_frame_reordering_is_allowed_but_identity_changes_are_not(tmp_path):
    first, second = frame(), frame()
    for key in ("atom_ids", "symbols", "positions"):
        second[key] = second[key][::-1]
    _, destination, argv = run(tmp_path, {"frames": [first, second]})
    assert main(argv) == 0
    result = json.loads(destination.read_text())
    assert result["frames"][0]["selected_atom_ids"] == result["frames"][1]["selected_atom_ids"]
    destination.unlink()
    second["atom_ids"][0] = 100
    argv_source = tmp_path / "frames.json"
    argv_source.write_text(json.dumps({"frames": [first, second]}))
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize(
    "document",
    [
        {"frames": []},
        {"frames": {}},
        [],
        {**frame(), "units": "bohr"},
        {**frame(), "boundary": "periodic"},
        {**frame(), "periodic": True},
        {**frame(), "cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]},
        {"frames": [frame(), {"units": "angstrom"}]},
    ],
)
def test_invalid_input_writes_no_report(tmp_path, document):
    _, destination, argv = run(tmp_path, document)
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert not destination.exists()


def settings_args(tmp_path, document, settings):
    source, destination, _ = run(tmp_path, document)
    config = tmp_path / "settings.json"
    config.write_text(json.dumps(settings))
    return (
        source,
        destination,
        config,
        ["--input", str(source), "--output", str(destination), "--settings", str(config)],
    )


def water_config():
    return {
        "schema_version": 1,
        "mode": "offline",
        "method": "water-nearest",
        "centers": [1],
        "oxygen_count": 1,
    }


def test_settings_water_matches_legacy_and_records_both_hashes(tmp_path):
    source, legacy_output, legacy_args = run(tmp_path, frame())
    assert main(legacy_args) == 0
    legacy = json.loads(legacy_output.read_text())
    legacy_output.unlink()
    _, destination, config, argv = settings_args(tmp_path, frame(), water_config())
    config_before, input_before = config.read_bytes(), source.read_bytes()
    assert main(argv) == 0
    report = json.loads(destination.read_text())
    assert report["frames"] == legacy["frames"]
    assert report["selector"] == legacy["selector"]
    assert report["settings"]["resolved"] == legacy["settings"]["resolved"]
    assert report["settings"]["sha256"] == hashlib.sha256(config_before).hexdigest()
    assert report["settings"]["hash_basis"] == "file-bytes"
    assert report["source"]["sha256"] == hashlib.sha256(input_before).hexdigest()
    canonical = json.dumps(
        legacy["settings"]["resolved"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert legacy["settings"]["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert legacy["settings"]["hash_basis"] == "canonical-json"
    saved = destination.read_bytes()
    with pytest.raises(SystemExit):
        main(argv)
    assert destination.read_bytes() == saved
    assert config.read_bytes() == config_before
    assert source.read_bytes() == input_before


def test_settings_arbitrary_groups_and_reordered_frames(tmp_path):
    first = {
        "units": "angstrom",
        "boundary": "nonperiodic",
        "atom_ids": [5, 9, 12],
        "symbols": ["Si", "O", "Ar"],
        "positions": [[0, 0, 0], [1.6, 0, 0], [5, 0, 0]],
    }
    second = {**first, **{key: first[key][::-1] for key in ("atom_ids", "symbols", "positions")}}
    config = {
        "schema_version": 1,
        "mode": "offline",
        "method": "fixed-groups",
        "groups": [
            {"group_id": "oxide", "atom_ids": [9, 5], "symbols": ["O", "Si"]},
            {"group_id": "argon", "atom_ids": [12], "symbols": ["Ar"]},
        ],
        "selected_group_ids": ["oxide"],
    }
    _, destination, _, argv = settings_args(tmp_path, {"frames": [first, second]}, config)
    assert main(argv) == 0
    report = json.loads(destination.read_text())
    assert report["counts"] == {"frames": 2, "selected": 2, "unresolved": 0}
    for result in report["frames"]:
        assert result["selected_atom_ids"] == [5, 9]
        assert result["selected_group_ids"] == ["oxide"]
    destination.unlink()
    second["symbols"][0] = "He"
    (tmp_path / "frames.json").write_text(json.dumps({"frames": [first, second]}))
    with pytest.raises(SystemExit):
        main(argv)
    assert not destination.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--centers", "1"],
        ["--oxygen-count", "1"],
        ["--max-oh-distance", "1.35"],
        ["--shared-distance-gap", "0.1"],
        ["--score-tolerance", "1e-10"],
    ],
)
def test_settings_rejects_every_selector_flag_without_silent_precedence(tmp_path, extra):
    _, destination, _, argv = settings_args(tmp_path, frame(), water_config())
    with pytest.raises(SystemExit) as error:
        main(argv + extra)
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize(
    "settings",
    [
        {**water_config(), "mode": "adaptive"},
        {**water_config(), "dynamic": True},
        {**water_config(), "max_oh_distance": float("nan")},
        {**water_config(), "schema_version": True},
    ],
)
def test_invalid_settings_leave_no_report(tmp_path, settings):
    _, destination, _, argv = settings_args(tmp_path, frame(), settings)
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("extra", [[], ["--centers", "1"], ["--oxygen-count", "1"]])
def test_legacy_requires_both_center_and_budget(tmp_path, extra):
    source, destination, _ = run(tmp_path, frame())
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--output", str(destination), *extra])
    assert not destination.exists()
