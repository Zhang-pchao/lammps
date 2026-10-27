"""Periodic embedding reuses bounded cold workers and rejects identity drift."""
import dataclasses
import hashlib
import sys

import numpy as np
import pytest

from qmdpa.abacus import (
    AbacusBackend,
    validate_abacus_settings,
    validate_mechanical_abacus_settings,
)
from qmdpa.core import EvaluationError, Geometry
from qmdpa.periodic_field import PeriodicGaussianField, native_field_text, parse_native_field_result


def inputs(tmp_path):
    exe = tmp_path / "engine"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    pp = tmp_path / "H.upf"
    pp.write_text("test identity only")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    field = PeriodicGaussianField((11, 12), [0.3, -0.3], [0.8, 0.8],
                                  [6, 6, 6], 6, "declared fixture")
    settings = {"executable": str(exe), "executable_sha256": digest(exe),
                    "pseudopotentials": {"H": {"path": str(pp), "sha256": digest(pp)}},
                    "supercell_angstrom": [6, 6, 6], "coordinate_offset_angstrom": [0, 0, 0],
                    "coordinate_gauge": "periodic-wrap", "execution_mode": "fileio",
                    "boundary": "periodic-same-cell", "periodic_field": field.record()}
    qm = Geometry((1, 2), ("H", "H"), [[2.63, 3, 3], [3.37, 3, 3]], np.eye(3) * 6)
    env = Geometry((11, 12), ("H", "H"), [[1, 1.8, 2.4], [4.8, 4.1, 3.6]], qm.cell)
    return settings, field, qm, env


def test_explicit_field_admission_and_mechanical_rejection(tmp_path):
    settings, field, _, _ = inputs(tmp_path)
    normalized = validate_abacus_settings(settings, tmp_path)
    assert normalized["periodic_field"] == field.record()
    with pytest.raises(ValueError, match="Mechanical"):
        validate_mechanical_abacus_settings(settings, tmp_path)
    for change in ({"boundary": "finite-supercell"}, {"execution_mode": "socket"},
                   {"supercell_angstrom": [6, 6, 7]}):
        with pytest.raises(ValueError):
            validate_abacus_settings({**settings, **change}, tmp_path)


def test_native_field_units_and_parser_guards(tmp_path):
    _, field, _, env = inputs(tmp_path)
    rows = native_field_text(field, env).splitlines()
    assert rows[:2] == ["QMDPA_GAUSSIAN_FIELD_V1", "2"]
    assert float(rows[2].split()[2]) == pytest.approx(0.8 / 0.5291770)
    log = """E_gaussian_field 0.05 0.6802849
QMDPA_GAUSSIAN_FIELD_V1_RESULT
reciprocal_kmax 6
ionic_energy_Ry 0.03
electronic_energy_Ry 0.02
environment_count 2
environment_forces_Ry_per_Bohr
11 0.001 0 0
12 -0.001 0 0
END_QMDPA_GAUSSIAN_FIELD_V1_RESULT
"""
    force, diagnostics = parse_native_field_result(log, field)
    assert force[0, 0] == pytest.approx(0.001 * 13.605698 / 0.5291770)
    assert diagnostics["field"] == field.record()
    for bad in (log + log, log.replace("11 0.001", "13 0.001"),
                log.replace("ionic_energy_Ry 0.03", "ionic_energy_Ry nan"),
                log.replace("reciprocal_kmax 6", "reciprocal_kmax 5"),
                log.replace("0.6802849", "0.8802849")):
        with pytest.raises(ValueError):
            parse_native_field_result(bad, field)


WORKER = r"""
import hashlib,json,sys
from pathlib import Path
mode=sys.argv[1]
for line in sys.stdin:
    req=json.loads(line)
    reply={"id":req["id"]}
    if req["op"]=="initialize":
        field=req["settings"]["periodic_field"]
        source=Path(__file__).resolve()
        reply.update(status="ready",execution_mode="fileio",
                     interface={"path":str(source),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
    elif req["op"]=="evaluate":
        q=req["geometry"]; e=req["environment"]
        reply.update(status="ok",scf_converged=True,atom_ids=q["atom_ids"],symbols=q["symbols"],
                     energy_ev=2.0,forces_ev_per_angstrom=[[0,0,0] for _ in q["atom_ids"]],
                     environment_atom_ids=e["atom_ids"],environment_symbols=e["symbols"],
                     environment_forces_ev_per_angstrom=[[0,0,0] for _ in e["atom_ids"]],
                     periodic_field_diagnostics={"field":field})
        if mode=="wrong-environment":reply["environment_atom_ids"]=[12,11]
        if mode=="missing-forces":del reply["environment_forces_ev_per_angstrom"]
    elif req["op"]=="close":
        reply["status"]="closed"
        print(json.dumps(reply),flush=True)
        break
    print(json.dumps(reply),flush=True)
"""


def test_cold_worker_complete_environment_and_permanent_failure(tmp_path, monkeypatch):
    settings, field, qm, env = inputs(tmp_path)
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    monkeypatch.setattr("qmdpa.abacus._worker_command",
                        lambda *args: [sys.executable, str(script), "ok"])
    provider = AbacusBackend(**settings, directory=tmp_path / "runtime")
    result = provider.evaluate(qm, env)
    assert result.energy_ev == 2
    assert result.environment_atom_ids == field.atom_ids
    assert result.environment_forces_ev_per_angstrom.shape == (2, 3)
    result = provider.evaluate(qm, dataclasses.replace(env, positions=env.positions + 0.1))
    assert provider.statistics["successful_evaluations"] == 2
    with pytest.raises(EvaluationError, match="order"):
        provider.evaluate(qm, Geometry((12, 11), env.symbols, env.positions, env.cell))
    with pytest.raises(EvaluationError):
        provider.evaluate(qm, env)
    provider.close_checked()
    assert len(list((tmp_path / "runtime/evaluations").iterdir())) == 2


@pytest.mark.parametrize("mode", ["wrong-environment", "missing-forces"])
def test_bad_response_cannot_return_qm_only_success(tmp_path, monkeypatch, mode):
    settings, _, qm, env = inputs(tmp_path)
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    monkeypatch.setattr("qmdpa.abacus._worker_command",
                        lambda *args: [sys.executable, str(script), mode])
    provider = AbacusBackend(**settings, directory=tmp_path / "runtime")
    with pytest.raises(EvaluationError):
        provider.evaluate(qm, env)
    assert provider.statistics["successful_evaluations"] == 0
    assert provider.statistics["failed_evaluations"] == 1
    assert (tmp_path / "runtime/failure.json").is_file()
    provider.close()


def test_missing_or_inconsistent_environment_fails_before_spawn(tmp_path, monkeypatch):
    settings, _, qm, _ = inputs(tmp_path)
    provider = AbacusBackend(**settings, directory=tmp_path / "runtime")
    calls = []
    monkeypatch.setattr(provider, "_start", lambda *args: calls.append(args))
    with pytest.raises(EvaluationError, match="requires environment"):
        provider.evaluate(qm)
    assert not calls
    provider.close()


def test_explicit_density_cutoff_is_validated_and_retained(tmp_path):
    settings, _, _, _ = inputs(tmp_path)
    legacy = validate_abacus_settings(settings, tmp_path)
    assert "ecutrho" not in legacy
    chosen = validate_abacus_settings({**settings, "ecutwfc": 80, "ecutrho": 640}, tmp_path)
    assert chosen["ecutrho"] == 640
    for bad in (True, 0, float("nan"), 319.9):
        with pytest.raises(ValueError):
            validate_abacus_settings({**settings, "ecutwfc": 80, "ecutrho": bad}, tmp_path)
