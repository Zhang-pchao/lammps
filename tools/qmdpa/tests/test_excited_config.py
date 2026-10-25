# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Native RHF/TDA identity and preflight checks without a quantum engine."""

import hashlib
import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest

from qmdpa import excited
from qmdpa.additive_config import load_additive_config
from qmdpa.config import load_config, validate_qm
from qmdpa.core import EvaluationError, Geometry
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.inference import RAW_PROFILE
from qmdpa.providers import make_qm_provider, provider_contract, validate_provider


def reference(*, embedded=False, solver="dense"):
    settings = excited.TDASettings(tda_solver=solver)
    qm = Geometry([1, 2], ["H", "H"], [[0, 0, 0], [0, 0, 0.8]])
    environment = Geometry([4, 5, 6], ["O", "H", "H"], [[3, 0, 0], [4, 0, 0], [3, 1, 0]])
    charges = FixedEmbeddingCharges(
        [4, 5, 6], [-0.834, 0.417, 0.417], "Native configuration test charges"
    )
    return excited.TDAReference(
        qm,
        settings,
        [[0.0, 1.0], [0.0, 0.0]],
        1,
        2.0,
        1.0,
        "Synthetic state identity only",
        json.dumps(
            {"pyscf_version": excited._PYSCF_VERSION, "source_sha256": excited._SOURCE_SHA256}
        ),
        environment if embedded else None,
        charges if embedded else None,
    )


def provider_request(pinned):
    return {
        "engine": "pyscf",
        "device": "cpu",
        "state": {"method": "rhf-tda", "reference": pinned.to_record()},
    }


def region(geometry):
    ids = list(geometry.atom_ids)
    return {"atom_ids": ids, "symbols": list(geometry.symbols), "molecules": [ids]}


@pytest.fixture(params=[False, True], ids=["mechanical", "additive"])
def case(request, tmp_path):
    embedded = request.param
    pinned = reference(embedded=embedded)
    model = b"Configuration identity fixture; never evaluated"
    (tmp_path / "model.pt").write_bytes(model)
    (tmp_path / "atoms.data").write_text("Input identity fixture; never executed")
    configuration = {
        "schema_version": 1,
        "data_file": "atoms.data",
        "model": "model.pt",
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "type_symbols": ["H", "O"],
        "periodic": False,
        "provider": provider_request(pinned),
        "qm": asdict(pinned.settings),
        "timestep_fs": 0.1,
        "steps": 0,
    }
    if embedded:
        configuration.update(
            hamiltonian="finite-electrostatic-additive",
            qm_region=region(pinned.qm_geometry),
            environment_region=region(pinned.environment_geometry),
            environment_charge_spin=[0, 1],
            embedding=pinned.to_record()["charges"],
            inference_profile=RAW_PROFILE,
            cross={
                "form": "lj12-6",
                "source": "Explicit configuration test",
                "mixing": "none",
                "cutoff": None,
                "pairs": [
                    {
                        "qm_symbol": "H",
                        "environment_symbol": symbol,
                        "epsilon_ev": 0.0,
                        "sigma_angstrom": 1.0,
                    }
                    for symbol in ("O", "H")
                ],
            },
        )
    else:
        configuration.update(
            region=region(pinned.qm_geometry),
            full_charge_spin=[0, 1],
            region_charge_spin=[0, 1],
        )
    return embedded, pinned, configuration


def load(tmp_path, case):
    embedded, _, configuration = case
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(configuration))
    with patch("qmdpa.config.importlib.metadata.version", return_value="test-version"):
        return (load_additive_config if embedded else load_config)(path)


def test_dense_reference_is_detached_and_contract_contains_only_summary(case):
    embedded, pinned, configuration = case
    provider = configuration["provider"]
    normalized = validate_provider(provider, electrostatic=embedded)
    assert normalized == provider
    normalized["state"]["reference"]["transition_density"][0][1] = 0.0
    assert provider["state"]["reference"] == pinned.to_record()
    contract = provider_contract(provider, electrostatic=embedded)
    assert contract["state"] == {"method": "rhf-tda", "reference_sha256": pinned.sha256}
    assert contract["energy"] == "total-rhf-plus-tda-excitation"
    assert contract["force_coordinates"] == (["qm", "environment"] if embedded else ["qm"])
    assert "transition_density" not in json.dumps(contract)


def test_both_loaders_preserve_complete_reference_and_tda_settings(tmp_path, case):
    _, pinned, _ = case
    loaded = load(tmp_path, case)
    assert loaded["provider"]["state"]["reference"] == pinned.to_record()
    assert loaded["qm"] == asdict(pinned.settings)
    assert "grid_level" not in loaded["qm"]
    assert "nlc_grid_level" not in loaded["qm"]
    assert "include_qm_dipole" not in loaded["qm"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("grid_level", 3),
        ("grid_response", True),
        ("nlc_grid_level", None),
        ("include_qm_dipole", False),
        ("unrecognized", 1),
    ],
)
def test_ground_fields_are_not_silently_discarded_for_tda(tmp_path, case, field, value):
    case[2]["qm"][field] = value
    with pytest.raises(ValueError, match="unsupported RHF/TDA"):
        load(tmp_path, case)


@pytest.mark.parametrize("field,value", [("threads", 2), ("conv_tol", 1e-10), ("basis", "sto-3g")])
def test_settings_must_exactly_match_reference(tmp_path, case, field, value):
    case[2]["qm"][field] = value
    with pytest.raises(ValueError, match="settings differ"):
        load(tmp_path, case)


def test_dense_solver_is_explicit_not_an_inferred_default(tmp_path, case):
    del case[2]["qm"]["tda_solver"]
    with pytest.raises(ValueError, match="settings differ"):
        load(tmp_path, case)
    embedded, _, configuration = case
    configuration["provider"] = provider_request(reference(embedded=embedded, solver="iterative"))
    with pytest.raises(ValueError, match="explicitly dense"):
        load(tmp_path, case)


@pytest.mark.parametrize("change", ["missing", "extra", "path", "tamper", "source"])
def test_reference_rejection_precedes_engine_construction(tmp_path, case, change):
    state = case[2]["provider"]["state"]
    if change == "missing":
        del state["reference"]
    elif change == "extra":
        state["seed_root"] = 1
    elif change == "path":
        state["reference"] = "reference.json"
    elif change == "tamper":
        state["reference"]["transition_density"][0][1] = 2.0
    else:
        record = state["reference"]
        record["runtime"]["source_sha256"]["grad/tdrhf.py"] = "0" * 64
        payload = {key: value for key, value in record.items() if key != "sha256"}
        record["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    with (
        patch("qmdpa.excited._load_runtime") as runtime,
        pytest.raises((ValueError, EvaluationError)),
    ):
        load(tmp_path, case)
    runtime.assert_not_called()


@pytest.mark.parametrize("change", ["ids", "symbols", "order", "periodic"])
def test_qm_membership_and_periodicity_must_match(tmp_path, case, change):
    embedded, _, configuration = case
    qm = configuration["qm_region" if embedded else "region"]
    if change == "ids":
        qm["atom_ids"] = [2, 3]
        qm["molecules"] = [[2, 3]]
    elif change == "symbols":
        qm["symbols"][0] = "O"
        if embedded:
            configuration["cross"]["pairs"].extend(
                {
                    "qm_symbol": "O",
                    "environment_symbol": symbol,
                    "epsilon_ev": 0.0,
                    "sigma_angstrom": 1.0,
                }
                for symbol in ("O", "H")
            )
    elif change == "order":
        qm["atom_ids"].reverse()
    else:
        configuration["periodic"] = True
    with pytest.raises(ValueError, match="region IDs|nonperiodic"):
        load(tmp_path, case)


def test_environment_presence_cannot_be_inferred_from_state_alone(tmp_path, case):
    embedded, _, configuration = case
    configuration["provider"] = provider_request(reference(embedded=not embedded))
    with pytest.raises(ValueError, match="reference environment"):
        load(tmp_path, case)


def test_changed_reference_coordinates_are_allowed_and_change_identity(tmp_path, case):
    _, pinned, configuration = case
    baseline = load(tmp_path, case)
    displaced = replace(
        pinned,
        qm_geometry=Geometry(
            pinned.qm_geometry.atom_ids,
            pinned.qm_geometry.symbols,
            pinned.qm_geometry.positions + 0.1,
        ),
    )
    configuration["provider"] = provider_request(displaced)
    changed = load(tmp_path, case)
    assert changed["physics_sha256"] != baseline["physics_sha256"]
    assert changed["provider"]["state"]["reference"]["sha256"] == displaced.sha256
    # The pinned coordinates are an overlap reference, not the current data.
    (tmp_path / "atoms.data").write_text("A displaced or restarted geometry")
    evolved = load(tmp_path, case)
    assert evolved["provider"] == changed["provider"]
    assert evolved["physics_sha256"] != changed["physics_sha256"]
    configuration["steps"] = 4
    configuration["sample_interval"] = 2
    assert load(tmp_path, case)["physics_sha256"] == evolved["physics_sha256"]


def test_complete_reference_changes_restart_identity_even_with_same_contract_sha(tmp_path, case):
    embedded, pinned, configuration = case
    baseline = load(tmp_path, case)
    contract = baseline["provider_contract"]
    configuration["provider"] = provider_request(
        replace(pinned, label="Different explicit state label")
    )
    target = (
        "qmdpa.additive_config.provider_contract" if embedded else "qmdpa.config.provider_contract"
    )
    with patch(target, return_value=contract):
        changed = load(tmp_path, case)
    assert changed["provider_contract"] == baseline["provider_contract"]
    assert changed["physics_sha256"] != baseline["physics_sha256"]


@pytest.mark.parametrize("change", ["ids", "symbols", "order", "values", "source", "radii"])
def test_embedded_reference_checks_complete_environment_identity(tmp_path, case, change):
    embedded, _, configuration = case
    if not embedded:
        pytest.skip("An isolated reference has no charge environment")
    env = configuration["environment_region"]
    if change == "ids":
        env["atom_ids"] = [7, 8, 9]
        env["molecules"] = [[7, 8, 9]]
        configuration["embedding"]["atom_ids"] = [7, 8, 9]
    elif change == "symbols":
        env["symbols"] = ["H", "O", "H"]
    elif change == "order":
        env["atom_ids"] = [4, 6, 5]
        configuration["embedding"]["atom_ids"] = [4, 6, 5]
    elif change == "values":
        configuration["embedding"]["charges_e"] = [-0.8, 0.4, 0.4]
    elif change == "source":
        configuration["embedding"]["source"] = "Another charge provenance"
    else:
        configuration["embedding"]["radii_angstrom"] = [0.5, 0.4, 0.4]
    with pytest.raises(ValueError, match="environment region|fixed charges"):
        load(tmp_path, case)


def test_schema2_charge_resolution_must_match_complete_tda_reference(tmp_path, case):
    embedded, pinned, configuration = case
    if not embedded:
        pytest.skip("Charge requests belong to the additive route")
    configuration["schema_version"] = 2
    configuration["embedding"] = {
        "charge_request": {
            "kind": "water-preset",
            "preset": "TIP3P-original-1983",
            "units": "e",
            "source": pinned.charges.source,
            "expected_total_charge_e": 0,
            "groups": [[4, 5, 6]],
        }
    }
    # Resolved provenance is part of the fixed source identity, so pin the
    # actual resolution before testing the provider boundary.
    from qmdpa.charges import resolve_fixed_charges

    resolved = resolve_fixed_charges(
        configuration["embedding"]["charge_request"], [4, 5, 6], ["O", "H", "H"]
    )
    pinned = replace(pinned, charges=FixedEmbeddingCharges(**resolved["embedding"]))
    configuration["provider"] = provider_request(pinned)
    loaded = load(tmp_path, case)
    assert loaded["embedding"]["source"] == pinned.charges.source
    configuration["embedding"]["charge_request"]["source"] = "Changed source"
    with pytest.raises(ValueError, match="fixed charges"):
        load(tmp_path, case)


def test_factory_wraps_exact_backend_without_extra_evaluation(case):
    embedded, pinned, configuration = case
    target = "PySCFTDAElectrostaticBackend" if embedded else "PySCFTDABackend"
    charges = pinned.charges if embedded else None
    with (
        patch(f"qmdpa.excited.{target}") as backend,
        patch("qmdpa.state_records.RecordingTDAProvider") as recording,
    ):
        result = make_qm_provider(
            configuration["qm"], provider=configuration["provider"], charges=charges
        )
    assert result is recording.return_value
    recording.assert_called_once_with(backend.return_value)
    arguments = backend.call_args.args
    assert arguments[0] == pinned.settings
    assert arguments[1].to_record() == pinned.to_record()
    assert arguments[2:] == ((charges,) if embedded else ())
    backend.return_value.evaluate.assert_not_called()
    backend.return_value.evaluate_surface.assert_not_called()


def test_factory_rechecks_settings_before_any_quantum_runtime(case):
    embedded, pinned, configuration = case
    configuration["qm"]["threads"] = 8
    with (
        patch("qmdpa.excited._load_runtime") as runtime,
        pytest.raises(ValueError, match="settings differ"),
    ):
        make_qm_provider(
            configuration["qm"],
            provider=configuration["provider"],
            charges=pinned.charges if embedded else None,
        )
    runtime.assert_not_called()


def test_ground_normalization_and_defaults_remain_independent_of_tda():
    isolated = validate_qm({})
    embedded = validate_qm({}, electrostatic=True)
    assert isolated["xc"] == embedded["xc"] == "pbe"
    assert isolated["grid_level"] == embedded["grid_level"] == 3
    assert embedded["nlc_grid_level"] is None
    assert embedded["include_qm_dipole"] is False
    assert "tda_solver" not in isolated


@pytest.mark.parametrize("field,value", [("source", "Changed"), ("charges_e", [-0.8, 0.4, 0.4])])
def test_factory_rechecks_fixed_charge_identity_before_runtime(field, value):
    pinned = reference(embedded=True)
    changed = pinned.to_record()["charges"]
    changed[field] = value
    with (
        patch("qmdpa.excited._load_runtime") as runtime,
        pytest.raises(ValueError, match="fixed charges"),
    ):
        make_qm_provider(
            asdict(pinned.settings),
            provider=provider_request(pinned),
            charges=FixedEmbeddingCharges(**changed),
        )
    runtime.assert_not_called()


def test_native_gaussian_reference_is_rejected_before_runtime(tmp_path, case):
    embedded, pinned, configuration = case
    if not embedded:
        pytest.skip("An isolated reference has no Gaussian charges")
    gaussian = replace(pinned.charges, radii_angstrom=[0.5, 0.4, 0.4])
    pinned = replace(pinned, charges=gaussian)
    configuration["provider"] = provider_request(pinned)
    configuration["embedding"] = pinned.to_record()["charges"]
    with (
        patch("qmdpa.excited._load_runtime") as runtime,
        pytest.raises(ValueError, match="point charges only"),
    ):
        load(tmp_path, case)
    runtime.assert_not_called()
    with (
        patch("qmdpa.excited._load_runtime") as runtime,
        pytest.raises(ValueError, match="point charges only"),
    ):
        make_qm_provider(configuration["qm"], provider=configuration["provider"], charges=gaussian)
    runtime.assert_not_called()
    # This restriction belongs to the native registry, not the standalone API.
    assert excited.TDAReference.from_record(pinned.to_record()).charges.radii_angstrom.tolist() == [
        0.5,
        0.4,
        0.4,
    ]


@pytest.mark.parametrize(
    "location", ["provider", "reference", "reference-setting", "reference-label"]
)
def test_inline_tda_raw_json_rejects_duplicate_keys(tmp_path, case, location):
    embedded, _, configuration = case
    raw = json.dumps(configuration)
    if location == "provider":
        raw = '{"provider": {"state": {"method": "ground"}},' + raw[1:]
    elif location == "reference":
        raw = raw.replace('"reference": {', '"reference": {}, "reference": {', 1)
    elif location == "reference-setting":
        raw = raw.replace('"nstates": 3', '"nstates": 99, "nstates": 3', 1)
    else:
        raw = raw.replace('"label": ', '"label": "discarded", "label": ', 1)
    # Ordinary JSON parsing has already lost the duplicated declaration.
    assert json.loads(raw) == configuration
    path = tmp_path / "ambiguous.json"
    path.write_text(raw)
    with (
        patch("qmdpa.excited._load_runtime") as runtime,
        pytest.raises(ValueError, match="Duplicate JSON key"),
    ):
        (load_additive_config if embedded else load_config)(path)
    runtime.assert_not_called()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_inline_tda_raw_json_rejects_shadowed_nonfinite_values(tmp_path, case, constant):
    embedded, _, configuration = case
    raw = json.dumps(configuration)
    raw = raw.replace('"nstates": 3', '"nstates": ' + constant + ', "nstates": 3', 1)
    assert json.loads(raw) == configuration
    path = tmp_path / "nonfinite.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="Nonfinite JSON constant"):
        (load_additive_config if embedded else load_config)(path)


@pytest.mark.parametrize("enabled", [False, True])
def test_exact_reuse_option_is_explicit_and_changes_restart_identity(case, tmp_path, enabled):
    embedded, pinned, configuration = case
    original = load(tmp_path, case)
    configuration["provider"]["reuse_identical_geometry"] = enabled
    resolved = load(tmp_path, case)
    assert resolved["provider"]["reuse_identical_geometry"] is enabled
    assert resolved["physics_sha256"] != original["physics_sha256"]
    contract = provider_contract(configuration["provider"], electrostatic=embedded)
    assert contract.get("result_reuse") == ("last-successful-exact-input" if enabled else None)
    assert contract["history"] == "none"
    target = "PySCFTDAElectrostaticBackend" if embedded else "PySCFTDABackend"
    with patch(f"qmdpa.excited.{target}") as backend, patch("qmdpa.state_records.RecordingTDAProvider"):
        make_qm_provider(configuration["qm"], provider=configuration["provider"],
                         charges=pinned.charges if embedded else None)
    assert backend.call_args.kwargs == {"reuse_identical_geometry": enabled}


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_exact_reuse_rejects_non_boolean_requests(case, flag):
    embedded, pinned, configuration = case
    configuration["provider"]["reuse_identical_geometry"] = flag
    with pytest.raises(ValueError, match="true or false"):
        validate_provider(configuration["provider"], electrostatic=embedded)


@pytest.mark.parametrize("enabled", [False, True])
def test_ground_state_does_not_silently_accept_reuse_option(enabled):
    with pytest.raises(ValueError, match="only for RHF/TDA"):
        validate_provider({"engine": "pyscf", "device": "cpu", "state": {"method": "ground"},
                           "reuse_identical_geometry": enabled})
