# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# SPDX-License-Identifier: GPL-2.0-only
"""Early rejection must be collective and must never initialize an engine."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from qmdpa.runtime import preflight_restart


def test_fresh_run_requires_no_restart_loader():
    loader = Mock(side_effect=AssertionError("must not read"))
    assert preflight_restart("unused", None, config_loader=loader) is None
    loader.assert_not_called()


def test_root_error_is_broadcast_before_raise():
    comm = Mock(rank=0)
    comm.bcast.side_effect = lambda value, root: value
    with pytest.raises(RuntimeError, match="broken checkpoint"):
        preflight_restart("input", "checkpoint", comm=comm,
                          config_loader=Mock(side_effect=ValueError("broken checkpoint")))
    assert comm.bcast.call_count == 1
    assert comm.bcast.call_args.args[0][0] == "ValueError: broken checkpoint"


def test_peer_rejects_without_loading_or_native_work():
    comm = Mock(rank=1)
    comm.bcast.return_value = ("ValueError: missing selected-state", None)
    loader = Mock(side_effect=AssertionError("non-owner read"))
    with pytest.raises(RuntimeError, match="missing selected-state"):
        preflight_restart("input", "checkpoint", comm=comm, config_loader=loader)
    loader.assert_not_called()


def test_valid_preflight_reuses_actual_checkpoint_validator():
    loader = Mock(return_value={"physics_sha256": "test"})
    result = {"restart_path": "binary"}
    with patch("qmdpa.runtime._validate_restart", return_value=result) as validate:
        assert preflight_restart("input", "checkpoint", config_loader=loader) is result
    validate.assert_called_once_with("checkpoint", loader.return_value)


def test_native_run_rejects_before_lammps_constructor(tmp_path):
    from qmdpa.cli import _run
    constructor = Mock(side_effect=AssertionError("native startup reached"))
    comm = Mock(rank=0)
    comm.bcast.side_effect = lambda value, root: value
    with patch.dict(sys.modules, {"mpi4py": SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=comm)),
                                  "lammps": SimpleNamespace(lammps=constructor)}):
        with pytest.raises(RuntimeError, match="invalid restart"):
            _run("input", tmp_path/"fresh", restart="checkpoint",
                 config_loader=Mock(side_effect=ValueError("invalid restart")),
                 setup_pair=Mock(), make_coupling=Mock())
    constructor.assert_not_called()
    failure = json.loads((tmp_path/"fresh/failure.json").read_text())
    assert failure["native_initialized"] is False
    assert "invalid restart" in failure["error"]
    assert list((tmp_path/"fresh").iterdir()) == [tmp_path/"fresh/failure.json"]


def test_validation_cli_checks_restart_without_running(tmp_path, capsys):
    from qmdpa.cli import main
    path = tmp_path/"checkpoint.json"
    with (patch("qmdpa.dispatch.load_run_config", return_value={"test": True}),
          patch("qmdpa.dispatch.describe_run", return_value={"status": "configuration-valid"}),
          patch("qmdpa.cli._validate_restart") as validate,
          patch("qmdpa.dispatch.run") as run):
        assert main(["validate", "input.json", "--restart", str(path)]) == 0
    validate.assert_called_once()
    run.assert_not_called()
    assert json.loads(capsys.readouterr().out)["restart"]["native_recomputed_state"] == "not evaluated"


def test_preflight_never_overwrites_existing_evidence(tmp_path):
    output = tmp_path/"evidence"
    output.mkdir()
    failure = output/"failure.json"
    failure.write_text("old evidence\n")
    with pytest.raises(RuntimeError, match="invalid restart"):
        preflight_restart("input", "checkpoint", output=output,
                          config_loader=Mock(side_effect=ValueError("invalid restart")))
    assert failure.read_text() == "old evidence\n"
