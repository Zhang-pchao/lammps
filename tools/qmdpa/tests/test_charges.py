# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Explicit fixed-charge sources, coverage, overrides and content identity."""

import copy
import hashlib
import json

import pytest

from qmdpa.charges import resolve_fixed_charges

IDS = [11, 12, 13]
SYMBOLS = ["O", "H", "H"]


def preset(name="TIP3P-original-1983"):
    return {
        "kind": "water-preset",
        "units": "e",
        "source": "Explicit diagnostic choice; not a qualified hybrid potential",
        "expected_total_charge_e": 0,
        "preset": name,
        "groups": [[11, 12, 13]],
    }


def per_atom(values=(-0.834, 0.417, 0.417), ids=IDS):
    return {
        "kind": "per-atom",
        "units": "e",
        "source": "Declared charges for the test atoms",
        "expected_total_charge_e": sum(values),
        "charges": [{"atom_id": atom, "charge_e": value} for atom, value in zip(ids, values)],
    }


def external(tmp_path, raw=None):
    if raw is None:
        raw = json.dumps(
            {
                "schema_version": 1,
                "units": "e",
                "source": "Independent per-atom file",
                "charges": per_atom()["charges"],
            }
        ).encode()
    path = tmp_path / "charges.json"
    path.write_bytes(raw)
    spec = {
        "kind": "file",
        "units": "e",
        "source": "Explicit file choice",
        "expected_total_charge_e": 0,
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return spec, path


@pytest.mark.parametrize(
    "name,expected",
    [
        ("TIP3P-original-1983", [-0.834, 0.417, 0.417]),
        ("SPC", [-0.820, 0.410, 0.410]),
        ("SPC/E", [-0.8476, 0.4238, 0.4238]),
    ],
)
def test_named_water_templates_have_exact_declared_parameters(name, expected):
    spec = preset(name)
    original = copy.deepcopy(spec)
    result = resolve_fixed_charges(spec, IDS, SYMBOLS)
    assert result["embedding"]["charges_e"] == expected
    assert result["embedding"]["atom_ids"] == IDS
    assert result["provenance"]["details"]["reference_url"].startswith("https://docs.lammps.org/")
    assert result["provenance"]["details"]["preset"] == name
    assert result["provenance"]["customized"] is False
    assert spec == original
    json.dumps(result, allow_nan=False)


def test_stable_large_ids_and_input_order_are_preserved():
    ids = [2**60 + index for index in (1, 2, 3)]
    spec = per_atom(ids=ids)
    expected = dict(zip(ids, [-0.834, 0.417, 0.417]))
    spec["charges"].reverse()
    order = ids[::-1]
    result = resolve_fixed_charges(spec, order, SYMBOLS[::-1])
    assert result["embedding"]["atom_ids"] == order
    assert result["embedding"]["charges_e"] == [expected[atom] for atom in order]


def test_types_are_explicit_labels_and_never_derived_from_elements():
    spec = {
        "kind": "atom-types",
        "units": "e",
        "source": "Two independently specified oxygen types",
        "expected_total_charge_e": 0,
        "atom_types": [{"atom_id": 8, "type": "oxide-O"}, {"atom_id": 3, "type": "water-O"}],
        "type_charges": [{"type": "water-O", "charge_e": -1}, {"type": "oxide-O", "charge_e": 1}],
    }
    result = resolve_fixed_charges(spec, [3, 8], ["O", "O"])
    assert result["embedding"]["charges_e"] == [-1.0, 1.0]
    reordered = copy.deepcopy(spec)
    reordered["atom_types"].reverse()
    reordered["type_charges"].reverse()
    assert resolve_fixed_charges(reordered, [3, 8], ["O", "O"]) == result


def test_overrides_have_precedence_provenance_and_post_override_sum():
    spec = preset()
    spec["expected_total_charge_e"] = 1
    spec["overrides"] = [{"atom_id": 11, "charge_e": 0.166, "source": "Explicit sensitivity case"}]
    result = resolve_fixed_charges(spec, IDS, SYMBOLS)
    assert result["embedding"]["charges_e"] == [0.166, 0.417, 0.417]
    provenance = result["provenance"]
    assert provenance["base_charges_e"] == [-0.834, 0.417, 0.417]
    assert provenance["total_charge_e"] == pytest.approx(1)
    assert provenance["customized"] is True
    assert provenance["overrides"][0]["previous_charge_e"] == -0.834
    assert "custom charge template" in result["embedding"]["source"]
    spec["expected_total_charge_e"] = 0
    with pytest.raises(ValueError, match="total charge"):
        resolve_fixed_charges(spec, IDS, SYMBOLS)


def test_ionic_and_explicit_zero_charge_inputs_are_allowed():
    ionic = per_atom((1.0,), [7])
    assert resolve_fixed_charges(ionic, [7], ["Na"])["embedding"]["charges_e"] == [1.0]
    zeros = per_atom((0, 0, 0))
    assert resolve_fixed_charges(zeros, IDS, SYMBOLS)["embedding"]["charges_e"] == [0.0] * 3
    with pytest.raises(ValueError):
        resolve_fixed_charges({}, IDS, SYMBOLS)


def test_per_atom_order_and_override_order_do_not_change_resolved_record():
    spec = per_atom()
    spec["overrides"] = [
        {"atom_id": 12, "charge_e": 0.3, "source": "First adjustment"},
        {"atom_id": 13, "charge_e": 0.534, "source": "Compensating adjustment"},
    ]
    expected = resolve_fixed_charges(spec, IDS, SYMBOLS)
    spec["charges"].reverse()
    spec["overrides"].reverse()
    assert resolve_fixed_charges(spec, IDS, SYMBOLS) == expected


def test_file_digest_is_required_checked_and_independent_of_location(tmp_path):
    spec, path = external(tmp_path)
    result = resolve_fixed_charges(spec, IDS, SYMBOLS, base_directory=tmp_path)
    moved = tmp_path / "other" / "renamed.json"
    moved.parent.mkdir()
    moved.write_bytes(path.read_bytes())
    same = {**spec, "path": str(moved)}
    assert resolve_fixed_charges(same, IDS, SYMBOLS) == result
    assert result["provenance"]["details"]["sha256"] == spec["sha256"]
    assert str(tmp_path) not in json.dumps(result)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 differs"):
        resolve_fixed_charges(spec, IDS, SYMBOLS, base_directory=tmp_path)
    for digest in ("", "0" * 63, "g" * 64, True):
        with pytest.raises(ValueError, match="SHA256"):
            resolve_fixed_charges({**spec, "sha256": digest}, IDS, SYMBOLS, base_directory=tmp_path)


def test_file_overrides_are_recorded_after_exact_coverage(tmp_path):
    spec, _ = external(tmp_path)
    spec["overrides"] = [{"atom_id": 11, "charge_e": 0.166, "source": "File sensitivity"}]
    spec["expected_total_charge_e"] = 1
    result = resolve_fixed_charges(spec, IDS, SYMBOLS, base_directory=tmp_path)
    assert result["provenance"]["details"]["file_source"] == "Independent per-atom file"
    assert result["provenance"]["customized"] is True
    assert result["embedding"]["charges_e"][0] == 0.166


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1,"units":"e","source":"x","charges":[]}',
        b'{"schema_version":1,"units":"e","source":"x","charges":[{"atom_id":11,"charge_e":NaN}]}',
        b'{"schema_version":true,"units":"e","source":"x","charges":[]}',
        b'{"schema_version":1,"units":"coulomb","source":"x","charges":[]}',
        b'{"schema_version":1,"units":"e","source":"x","charges":[],"path":"nested.json"}',
    ],
)
def test_invalid_external_documents_are_not_silently_accepted(tmp_path, raw):
    spec, _ = external(tmp_path, raw)
    with pytest.raises(ValueError):
        resolve_fixed_charges(spec, IDS, SYMBOLS, base_directory=tmp_path)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0.1", None, 10**1000])
def test_nonfinite_nonreal_and_boolean_charge_values_are_rejected(bad):
    spec = per_atom()
    spec["charges"][0]["charge_e"] = bad
    with pytest.raises(ValueError):
        resolve_fixed_charges(spec, IDS, SYMBOLS)
    spec = preset()
    spec["expected_total_charge_e"] = bad
    with pytest.raises(ValueError):
        resolve_fixed_charges(spec, IDS, SYMBOLS)


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "element-defaults"},
        {"units": "coulomb"},
        {"source": " "},
        {"expected_total_charge_e": 1},
        {"charges": []},
        {"charges": [{"atom_id": 11, "charge_e": 0}]},
        {"charges": per_atom()["charges"] + [{"atom_id": 99, "charge_e": 0}]},
        {"charges": per_atom()["charges"] + [{"atom_id": 11, "charge_e": 0}]},
        {"charges": [{"atom_id": True, "charge_e": 0}]},
        {"charges": [{"atom_id": 11.0, "charge_e": 0}]},
        {"preset": "SPC"},
        {"overrides": [{"atom_id": 99, "charge_e": 0, "source": "Unknown atom"}]},
        {"overrides": [{"atom_id": 11, "charge_e": 0}]},
        {"overrides": [{"atom_id": 11, "charge_e": False, "source": "Invalid value"}]},
        {"overrides": [{"atom_id": 11, "charge_e": -0.834, "source": "same"}] * 2},
        {"overrides": {}},
    ],
)
def test_missing_conflicting_or_mixed_explicit_inputs_are_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        resolve_fixed_charges({**per_atom(), **change}, IDS, SYMBOLS)


@pytest.mark.parametrize(
    "groups,symbols",
    [
        ([[11, 12]], SYMBOLS),
        ([[11, 12, 12]], SYMBOLS),
        ([[11, 12, 13], [11, 12, 13]], SYMBOLS),
        ([[12, 11, 13]], SYMBOLS),
        ([[11, 12, 99]], SYMBOLS),
        ([[11, 12, 13]], ["O", "H", "O"]),
    ],
)
def test_water_templates_require_explicit_complete_disjoint_ohh_groups(groups, symbols):
    with pytest.raises(ValueError):
        resolve_fixed_charges({**preset(), "groups": groups}, IDS, symbols)


def test_water_preset_does_not_assign_undeclared_oxide_oxygen():
    with pytest.raises(ValueError, match="cover.*exactly"):
        resolve_fixed_charges(preset(), [*IDS, 20], [*SYMBOLS, "O"])


@pytest.mark.parametrize("name", ["TIP4P", "TIP4P/2005", "OPC"])
def test_unsupported_water_names_give_actionable_rejection(name):
    with pytest.raises(ValueError, match="virtual charge sites"):
        resolve_fixed_charges(preset(name), IDS, SYMBOLS)


@pytest.mark.parametrize(
    "mutation", ["duplicate-atom", "duplicate-type", "missing-type", "unused-type"]
)
def test_type_conflicts_and_incomplete_tables_are_rejected(mutation):
    spec = {
        "kind": "atom-types",
        "units": "e",
        "source": "Explicit hydrogen type",
        "expected_total_charge_e": 0,
        "atom_types": [{"atom_id": 1, "type": "site"}],
        "type_charges": [{"type": "site", "charge_e": 0}],
    }
    if mutation == "duplicate-atom":
        spec["atom_types"].append({"atom_id": 1, "type": "other"})
    elif mutation == "duplicate-type":
        spec["type_charges"].append({"type": "site", "charge_e": 1})
    elif mutation == "missing-type":
        spec["type_charges"] = [{"type": "other", "charge_e": 0}]
    else:
        spec["type_charges"].append({"type": "other", "charge_e": 0})
    with pytest.raises(ValueError):
        resolve_fixed_charges(spec, [1], ["H"])


@pytest.mark.parametrize(
    "ids,symbols",
    [([1, 1], ["H", "H"]), ([True], ["H"]), ([0], ["H"]), ([], []), ([1], []), ([1], [""])],
)
def test_invalid_target_identity_is_rejected(ids, symbols):
    with pytest.raises(ValueError):
        resolve_fixed_charges(per_atom((0,), [1]), ids, symbols)


def test_unknown_entry_fields_and_total_overflow_are_rejected():
    spec = per_atom()
    spec["charges"][0]["element"] = "O"
    with pytest.raises(ValueError, match="unsupported"):
        resolve_fixed_charges(spec, IDS, SYMBOLS)
    spec = per_atom((0.0, 0.0), [1, 2])
    spec["charges"] = [{"atom_id": 1, "charge_e": 1e308}, {"atom_id": 2, "charge_e": 1e308}]
    with pytest.raises(ValueError, match="finite"):
        resolve_fixed_charges(spec, [1, 2], ["H", "H"])


def test_unlisted_three_site_model_is_not_misreported_as_a_virtual_site_model():
    with pytest.raises(ValueError, match="Unknown water charge preset"):
        resolve_fixed_charges(preset("OPC3"), IDS, SYMBOLS)
