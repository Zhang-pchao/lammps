# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""State evidence belongs to one successful callback and accepted geometry."""

import copy
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from test_excited import reference, settings
from test_lammps_bridge import Driver, RemoteWriteFailureComm, WorkerComm

from qmdpa import cli, excited
from qmdpa.config import file_digest
from qmdpa.core import Evaluation, EvaluationError, Geometry
from qmdpa.embedding import EmbeddedEvaluation
from qmdpa.lammps import ExternalCoupling
from qmdpa.region import FixedRegion
from qmdpa.runtime import _attach_state, _compare_restart, _validate_restart
from qmdpa.state_records import (
    RecordingTDAProvider,
    geometry_digest,
    validate_frame_state,
    validate_state_record,
)


def diagnostics(pinned, root=1):
    index, chosen = excited.select_tda_state(
        [1.0, 0.0] if root == 1 else [0.0, 1.0], [2.0, 3.0], pinned.settings
    )
    return {
        **chosen,
        "reference_sha256": pinned.sha256,
        "tracking": "fixed-reference-ao-transition-density",
        "scf_converged": True,
        "candidate_roots_converged": True,
        "forces_evaluated": True,
        "method": "cpu-rhf-tda-singlet",
        "energy_definition": "E_RHF+omega",
        "ground_energy_ev": -10.0,
        "excitation_energy_ev": [2.0, 3.0][index],
        "tda_solver": "dense",
        "tda_dimension": 4,
        "tda_dense_max_dimension": 256,
        "tda_dense_batch_size": 16,
        "tda_hermiticity_max_error_hartree": 1e-15,
        "tda_hermiticity_tolerance_hartree": 1e-12,
        "tda_positive_eigenvalue_threshold_hartree": 0.001,
        "tda_matrix_residuals_hartree": [1e-14, 2e-14],
        "tda_eigensystem_residuals_hartree": [1e-14, 2e-14],
        "tda_eigensystem_residual_max_hartree": 2e-14,
        "tda_eigensystem_residual_tolerance_hartree": 1e-10,
        "cphf_residual_history": [2e-7, 1e-14],
        "cphf_refinement_iterations": 1,
        "cphf_residual_before_refinement": 2e-7,
        "cphf_residual_after_refinement": 1e-14,
        "cphf_residual_max": 2e-14,
        "cphf_residual_tolerance": 1e-7,
        "cphf_refinement_method": "normalized-defect-correction",
        "density_trace": 2.0,
    }


class Backend:
    def __init__(self, pinned):
        self.reference = pinned
        self.calls = []
        self.failure = False
        self.diagnostic = diagnostics(pinned)

    def evaluate_surface(self, qm, environment=None, *, forces):
        self.calls.append((qm, environment, forces))
        if self.failure:
            raise EvaluationError("injected quantum failure")
        return excited.TDASurfaceEvaluation(
            -10.0 + self.diagnostic["excitation_energy_ev"],
            qm.atom_ids,
            () if environment is None else environment.atom_ids,
            np.ones_like(qm.positions),
            None if environment is None else np.ones_like(environment.positions) * 2,
            self.diagnostic,
        )


def fixture(embedded=False):
    pinned = reference(embedded=embedded, configuration=settings(tda_solver="dense"))
    backend = Backend(pinned)
    observer = RecordingTDAProvider(backend)
    config = {
        "periodic": False,
        "provider": {
            "engine": "pyscf",
            "device": "cpu",
            "state": {"method": "rhf-tda", "reference": pinned.to_record()},
        },
        "physics_sha256": "synthetic-same-physics",
    }
    return pinned, backend, observer, config


def frame(pinned, step=4):
    atoms = []
    for geometry in (pinned.qm_geometry, pinned.environment_geometry):
        if geometry is not None:
            atoms.extend(
                {
                    "id": atom,
                    "x": position.tolist(),
                    "v": [0.0] * 3,
                    "f": [1.0] * 3,
                    "image": [0] * 3,
                }
                for atom, position in zip(geometry.atom_ids, geometry.positions)
            )
    return {"step": step, "potential_eV": -8.0, "atoms": sorted(atoms, key=lambda a: a["id"])}


def evaluate(observer):
    pinned = observer.reference
    return observer.evaluate(pinned.qm_geometry, pinned.environment_geometry)


def accepted(pinned, observer, config):
    result = frame(pinned)
    coupling = SimpleNamespace(failure=None, last_step=4, state_observer=observer)
    _attach_state(result, coupling, config)
    return result


@pytest.mark.parametrize("embedded", [False, True])
def test_one_force_evaluation_retains_detached_state_and_existing_result_contract(embedded):
    pinned, backend, observer, config = fixture(embedded)
    result = evaluate(observer)
    assert isinstance(result, EmbeddedEvaluation if embedded else Evaluation)
    assert len(backend.calls) == 1 and backend.calls[0][2] is True
    snapshot = accepted(pinned, observer, config)
    validate_frame_state(snapshot, config)
    assert snapshot["electronic_state"]["reference_sha256"] == pinned.sha256
    backend.diagnostic["candidate_overlaps"] = [0.1, 0.0]
    snapshot["electronic_state"]["diagnostics"]["candidate_overlaps"][0] = 0.2
    assert observer.accepted_record(frame(pinned))["diagnostics"]["candidate_overlaps"][0] == 1.0
    assert len(backend.calls) == 1  # Reporting never triggers another SCF.


@pytest.mark.parametrize("embedded", [False, True])
def test_new_failure_cannot_expose_last_successful_state(embedded):
    pinned, backend, observer, _ = fixture(embedded)
    evaluate(observer)
    backend.failure = True
    with pytest.raises(EvaluationError, match="injected"):
        evaluate(observer)
    with pytest.raises(EvaluationError, match="No successful"):
        observer.accepted_record(frame(pinned))


@pytest.mark.parametrize("embedded", [False, True])
def test_geometry_identity_covers_every_evaluated_coordinate_and_rejects_images(embedded):
    pinned, _, observer, _ = fixture(embedded)
    evaluate(observer)
    original = frame(pinned)
    for index in range(len(original["atoms"])):
        changed = copy.deepcopy(original)
        changed["atoms"][index]["x"][0] += 1e-12
        with pytest.raises(EvaluationError, match="sampled geometry"):
            observer.accepted_record(changed)
    changed = copy.deepcopy(original)
    changed["atoms"][0]["image"][0] = 1
    with pytest.raises(ValueError, match="zero image"):
        observer.accepted_record(changed)
    observer.clear()
    with pytest.raises(EvaluationError, match="No successful"):
        observer.accepted_record(original)


@pytest.mark.parametrize(
    "name,value",
    [
        ("selected_root", True),
        ("forces_evaluated", False),
        ("scf_converged", False),
        ("reference_sha256", "0" * 64),
        ("overlap_margin", 0.9),
        ("candidate_gap_ev", 0.5),
        ("tda_dimension", 257),
        ("tda_solver", "iterative"),
        ("tda_hermiticity_max_error_hartree", 2e-12),
        ("tda_eigensystem_residuals_hartree", [1e-10, 1e-14]),
        ("tda_eigensystem_residual_max_hartree", 0.0),
        ("cphf_residual_max", 2e-7),
        ("cphf_refinement_iterations", 3),
        ("cphf_residual_history", [1e-14, 2e-14]),
        ("density_trace", float("nan")),
    ],
)
def test_invalid_diagnostics_cannot_be_accepted_or_written(name, value):
    pinned, _, observer, _ = fixture()
    evaluate(observer)
    record = observer.accepted_record(frame(pinned))
    record["diagnostics"][name] = value
    with pytest.raises((ValueError, EvaluationError)):
        validate_state_record(record, pinned)


def test_record_root_energy_and_geometry_sha_are_checked():
    pinned, _, observer, config = fixture()
    evaluate(observer)
    original = accepted(pinned, observer, config)
    for key, value in (
        ("selected_root", 2),
        ("provider_energy_ev", -9),
        ("geometry_sha256", "bad"),
    ):
        changed = copy.deepcopy(original)
        changed["electronic_state"][key] = value
        with pytest.raises(ValueError):
            validate_frame_state(changed, config)
    with pytest.raises(ValueError, match="Ground-state"):
        validate_frame_state(original, {})


@pytest.mark.parametrize("failure,last_step", [("failed callback", 4), (None, 3)])
def test_accepted_frame_requires_successful_current_callback(failure, last_step):
    pinned, _, observer, config = fixture()
    evaluate(observer)
    result = frame(pinned)
    coupling = SimpleNamespace(failure=failure, last_step=last_step, state_observer=observer)
    with pytest.raises(ValueError, match="successful current"):
        _attach_state(result, coupling, config)
    assert "electronic_state" not in result


def test_restart_preflight_rejects_missing_state_before_opening_native_binary(tmp_path):
    pinned, _, observer, config = fixture()
    evaluate(observer)
    checkpoint = {
        "physics_sha256": config["physics_sha256"],
        "restart_file": "missing.bin",
        "restart_sha256": "0" * 64,
        "frame": frame(pinned),
    }
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="Incomplete electronic-state"):
        _validate_restart(path, config)
    checkpoint["frame"] = accepted(pinned, observer, config)
    binary = tmp_path / "restart.bin"
    binary.write_bytes(b"synthetic restart")
    checkpoint.update(restart_file=binary.name, restart_sha256=file_digest(binary))
    path.write_text(json.dumps(checkpoint))
    assert _validate_restart(path, config)["restart_path"] == str(binary)


def test_restart_recomputes_state_but_does_not_require_bitwise_equal_diagnostics():
    pinned, backend, observer, config = fixture()
    evaluate(observer)
    previous = accepted(pinned, observer, config)
    backend.diagnostic["cphf_residual_max"] = 3e-14
    evaluate(observer)
    resumed = accepted(pinned, observer, config)
    _compare_restart(resumed, {"frame": previous}, config)
    backend.diagnostic = diagnostics(pinned, root=2)
    evaluate(observer)
    changed = accepted(pinned, observer, config)
    with pytest.raises(ValueError, match="different electronic state"):
        _compare_restart(changed, {"frame": previous}, config)


def test_callback_failure_before_provider_and_remote_force_write_clear_state():
    pinned, _, observer, _ = fixture()
    evaluate(observer)

    def fail_before_provider(geometry):
        raise EvaluationError("low-level calculation failed")

    region = FixedRegion(pinned.qm_geometry.atom_ids, pinned.qm_geometry.symbols)
    lmp = Driver(tags=(1, 2))
    coupling = ExternalCoupling(
        lmp, region, SimpleNamespace(correction=fail_before_provider), state_observer=observer
    )
    with pytest.raises(EvaluationError, match="low-level"):
        coupling.run(0)
    with pytest.raises(EvaluationError, match="No successful"):
        observer.accepted_record(frame(pinned))
    lmp = Driver(tags=(1, 2), world_size=2)
    coupling = ExternalCoupling(
        lmp,
        region,
        SimpleNamespace(correction=observer.evaluate),
        state_observer=observer,
        comm=RemoteWriteFailureComm(),
    )
    with pytest.raises(EvaluationError, match="Remote force buffer"):
        coupling.run(0)
    assert not lmp.swallowed
    with pytest.raises(EvaluationError, match="No successful"):
        observer.accepted_record(frame(pinned))
    with pytest.raises(ValueError, match="only to rank zero"):
        ExternalCoupling(
            Driver(world_size=2), region, None, state_observer=observer, comm=WorkerComm()
        )


def test_mechanical_factory_passes_exact_recording_provider_to_callback():
    pinned, _, observer, _ = fixture()
    config = {
        "model": "unused.pt",
        "qm": {},
        "provider": {"engine": "pyscf"},
        "region": {
            "atom_ids": list(pinned.qm_geometry.atom_ids),
            "symbols": list(pinned.qm_geometry.symbols),
        },
        "type_symbols": ["H"],
    }
    with (
        patch("qmdpa.cli.make_qm_provider", return_value=observer),
        patch("qmdpa.cli.DeepMDBackend"),
        patch("qmdpa.cli.ExternalCoupling") as coupling,
    ):
        cli._mechanical_coupling(SimpleNamespace(), None, config)
    assert coupling.call_args.kwargs["state_observer"] is observer
    assert coupling.call_args.args[2].high is observer


@pytest.mark.parametrize("bad_frame", [False, True])
def test_driver_records_same_callback_before_writing_any_checkpoint(tmp_path, bad_frame):
    pinned, _, observer, config = fixture()
    config.update(data_file="unused.data", timestep_fs=0.1, steps=1, sample_interval=1)
    drivers = []

    class Native:
        def __init__(self, **kwargs):
            self.step = 0
            self.commands = []
            drivers.append(self)

        def get_mpi_comm(self):
            return None

        def extract_setting(self, name):
            return 1

        def extract_global(self, name):
            return self.step

        def commands_list(self, commands):
            self.commands.extend(commands)

        def command(self, command):
            self.commands.append(command)
            words = shlex.split(command)
            if words[0] == "write_restart":
                Path(words[1]).write_bytes(b"native checkpoint fixture")

        def close(self):
            pass

    class Coupling:
        failure = None

        def __init__(self, lmp):
            self.lmp = lmp
            self.state_observer = observer
            self.last_step = None

        def run(self, steps):
            evaluate(observer)
            self.lmp.step += steps
            self.last_step = self.lmp.step

        def statistics(self):
            return {}

    def capture(lmp, comm, *, periodic=False):
        result = frame(pinned, lmp.step)
        if bad_frame and lmp.step:
            result["atoms"][0]["x"][0] += 0.001
        return result

    output = tmp_path / "output"
    with (
        patch.dict(sys.modules, {"lammps": SimpleNamespace(lammps=Native)}),
        patch("qmdpa.cli._frame", side_effect=capture),
    ):

        def run():
            cli._run(
                "unused.json",
                output,
                restart=None,
                config_loader=lambda p: config,
                setup_pair=lambda lmp, cfg: None,
                make_coupling=lambda lmp, comm, cfg: Coupling(lmp),
            )

        if bad_frame:
            with pytest.raises(RuntimeError, match="sampled geometry"):
                run()
        else:
            run()
    frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert all("electronic_state" in item for item in frames)
    writes = [command for command in drivers[0].commands if command.startswith("write_restart ")]
    assert len(writes) == (0 if bad_frame else 1)
    assert len(frames) == (1 if bad_frame else 2)
    checkpoints = list(output.glob("checkpoint-*.json"))
    assert len(checkpoints) == (0 if bad_frame else 1)
    if checkpoints:
        saved = json.loads(checkpoints[0].read_text())["frame"]
        assert saved["electronic_state"] == frames[-1]["electronic_state"]


def test_geometry_digest_keeps_environment_and_input_order_explicit():
    pinned, _, _, _ = fixture(True)
    q, e = pinned.qm_geometry, pinned.environment_geometry
    assert geometry_digest(q) != geometry_digest(q, e)
    reversed_qm = Geometry(q.atom_ids[::-1], q.symbols[::-1], q.positions[::-1])
    assert geometry_digest(q, e) != geometry_digest(reversed_qm, e)


def test_mechanical_callback_signed_zero_matches_raw_accepted_coordinates():
    pinned, backend, observer, config = fixture()
    driver = Driver(tags=pinned.qm_geometry.atom_ids)
    driver.positions[:] = pinned.qm_geometry.positions
    driver.positions[driver.positions == 0.0] = -0.0
    driver.box = ([0, 0, 0], [8, 8, 8], 0.0, 0.0, 0.0, [0, 0, 0], 0)
    coupling = ExternalCoupling(
        driver,
        FixedRegion(pinned.qm_geometry.atom_ids, pinned.qm_geometry.symbols),
        SimpleNamespace(correction=observer.evaluate),
        state_observer=observer,
    )
    coupling.run(0)
    assert np.signbit(driver.positions[0, 0])
    assert not np.signbit(backend.calls[0][0].positions[0, 0])
    sampled = frame(pinned, step=0)
    for atom, position in zip(sampled["atoms"], driver.positions):
        atom["x"] = position.tolist()
    _attach_state(sampled, coupling, config)
    validate_frame_state(sampled, config)
    assert sampled["electronic_state"]["geometry_sha256"] == geometry_digest(pinned.qm_geometry)

    # Exact-zero normalization must not round even the smallest nonzero value.
    shifted = pinned.qm_geometry.positions.copy()
    shifted[0, 0] = np.nextafter(0.0, 1.0)
    tiny = Geometry(pinned.qm_geometry.atom_ids, pinned.qm_geometry.symbols, shifted)
    assert geometry_digest(tiny) != geometry_digest(pinned.qm_geometry)


def test_record_validation_failure_invalidates_backend_reuse():
    pinned, backend, observer, _ = fixture()
    cleared = []
    backend.clear_reuse = lambda: cleared.append(True)
    observer.evaluate(pinned.qm_geometry)
    backend.diagnostic["cphf_residual_max"] = 2.0
    with pytest.raises(ValueError):
        observer.evaluate(pinned.qm_geometry)
    assert cleared == [True]
    with pytest.raises(EvaluationError, match="No successful"):
        observer.accepted_record(frame(pinned))
