# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Native adapter assembly and collective failure guards with analytic providers."""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from test_adaptive import PairPolynomial, geometry
from test_adaptive_config import load
from test_additive_lammps_bridge import AdditiveDriver, AdditiveWorkerComm

from qmdpa import adaptive_cli
from qmdpa.adaptive_config import build_mixer, load_adaptive_config
from qmdpa.core import EvaluationError
from qmdpa.inference import RAW_ENV
from qmdpa.lammps import AdditiveExternalCoupling


class TrimerDriver(AdditiveDriver):
    def __init__(self, *, world_size=1):
        ids = (9, 4, 2, 8, 1, 5, 3, 7, 6)
        super().__init__(tags=ids, world_size=world_size)
        self.natoms = 9
        self.types[:] = [1 if atom in (1, 4, 7) else 2 for atom in ids]
        atoms = geometry()
        self.positions[:] = atoms.select(ids).positions


def torch_state():
    return SimpleNamespace(
        __version__="control",
        version=SimpleNamespace(cuda="control"),
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
    )


def providers(config, torch):
    lows, highs = [], []

    def low(model, *, charge_spin, no_jit):
        assert model == config["model"] and charge_spin == [0, 1] and no_jit is True
        value = PairPolynomial(0.04, -10.0)
        value.no_jit = True
        value._potential = SimpleNamespace(deep_eval=SimpleNamespace())
        lows.append(value)
        return value

    def high(qm, *, provider):
        assert qm == config["qm"] and provider == config["provider"]
        value = PairPolynomial(0.07, -5.0)
        value.xc, value.basis = qm["xc"], qm["basis"]
        highs.append(value)
        return value

    return lows, highs, low, high


@pytest.fixture
def attached(tmp_path):
    config = load(tmp_path)
    lmp, torch = TrimerDriver(), torch_state()
    lows, highs, low, high = providers(config, torch)
    with (
        patch.dict(os.environ, RAW_ENV),
        patch("qmdpa.inference.prepare_raw_inference", return_value=torch) as prepare,
        patch("qmdpa.backends.DeepMDBackend", side_effect=low) as low_factory,
        patch("qmdpa.providers.make_qm_provider", side_effect=high) as high_factory,
    ):
        coupling = adaptive_cli._adaptive_coupling(lmp, None, config)
        yield SimpleNamespace(
            config=config,
            lmp=lmp,
            coupling=coupling,
            torch=torch,
            lows=lows,
            highs=highs,
            prepare=prepare,
            low_factory=low_factory,
            high_factory=high_factory,
        )


def test_total_potential_and_all_atom_forces_are_published_once(attached):
    a = attached
    assert isinstance(a.coupling, AdditiveExternalCoupling)
    assert a.low_factory.call_count == 1 and a.high_factory.call_count == 2
    a.prepare.assert_called_once_with(a.config["model"])
    assert a.coupling.region.atom_ids == tuple(range(1, 10))
    assert a.config["inference_runtime"]["shared_full_and_region_model"] is True
    assert a.config["inference_runtime"]["no_jit"] is True
    original_hash = a.config["physics_sha256"]
    energy = a.coupling.run(0)
    assert a.lows[0].calls == 3
    assert [provider.calls for provider in a.highs] == [1, 1]
    expected = build_mixer(
        a.config,
        PairPolynomial(0.04, -10.0),
        (PairPolynomial(0.07, -5.0), PairPolynomial(0.07, -5.0)),
    ).evaluate(geometry())
    assert energy == pytest.approx(expected.energy, abs=1e-12)
    order = [atom - 1 for atom in a.lmp.tags]
    np.testing.assert_allclose(a.lmp.forces, expected.forces[order], atol=1e-12, rtol=0)
    assert a.lmp.energy_calls[-1] == ("qmdpa_adaptive", energy)
    assert a.coupling.statistics()["calculator_successes"] == 1
    assert a.config["physics_sha256"] == original_hash
    assert a.lmp.commands == [
        "fix qmdpa_adaptive all external pf/callback 1 1",
        "fix_modify qmdpa_adaptive energy yes virial no",
        "run 0 post no",
    ]


def test_pair_setup_installs_only_zero_native_potential():
    lmp = SimpleNamespace(commands_list=Mock())
    adaptive_cli._zero_pair(lmp, {})
    lmp.commands_list.assert_called_once_with(["pair_style zero 1.0", "pair_coeff * *"])


def test_native_assembly_is_rejected_before_backend_allocation(tmp_path):
    config = load(tmp_path)
    for field, value in (("pair_style", "deepmd"), ("bond_style", "harmonic"), ("natoms", 8)):
        lmp = TrimerDriver()
        if field == "natoms":
            lmp.natoms = value
        else:
            lmp.styles[field] = value
        with patch("qmdpa.backends.DeepMDBackend") as factory, pytest.raises(RuntimeError):
            adaptive_cli._adaptive_coupling(lmp, None, config)
        factory.assert_not_called()
        assert lmp.commands == []


def test_provider_failure_clears_total_force_without_publishing_a_new_state(attached):
    a = attached
    a.coupling.run(0)
    a.highs[1].failed = True
    with pytest.raises(EvaluationError, match="Deliberate provider"):
        a.coupling.run(1)
    assert a.coupling.last_step == 0 and a.lmp.timeout
    assert a.lmp.energy_calls[-1][1] == 0
    np.testing.assert_array_equal(a.lmp.forces, np.zeros((9, 3)))
    assert a.coupling.statistics()["calculator_failures"] == 1
    before = a.lows[0].calls
    with pytest.raises(EvaluationError):
        a.coupling.run(1)
    assert a.lows[0].calls == before


def test_runtime_policy_mutation_fails_closed_after_force_calculation(attached):
    a = attached
    evaluate = a.highs[1].evaluate

    def changed(frame):
        value = evaluate(frame)
        a.torch.backends.cuda.matmul.allow_tf32 = True
        return value

    a.highs[1].evaluate = changed
    with pytest.raises(EvaluationError, match="policy changed"):
        a.coupling.run(0)
    assert a.lmp.timeout and a.coupling.last_step is None
    np.testing.assert_array_equal(a.lmp.forces, np.zeros((9, 3)))
    assert a.lmp.energy_calls[-1][1] == 0


def test_atom_type_and_force_assembly_guards_remain_active(attached):
    a = attached
    a.coupling.run(0)
    a.lmp.native_forces[0, 0] = 1e-30
    with pytest.raises(EvaluationError, match="exactly zero"):
        a.coupling.run(1)
    assert a.coupling.last_step == 0
    assert a.lmp.energy_calls[-1][1] == 0


def test_worker_rank_constructs_no_scientific_provider(tmp_path):
    config = load(tmp_path)
    lmp = TrimerDriver(world_size=2)
    comm = AdditiveWorkerComm([(None, None), (None, None)])
    with (
        patch("qmdpa.backends.DeepMDBackend") as low,
        patch("qmdpa.providers.make_qm_provider") as high,
        patch("qmdpa.inference.prepare_raw_inference") as prepare,
    ):
        coupling = adaptive_cli._adaptive_coupling(lmp, comm, config)
    low.assert_not_called()
    high.assert_not_called()
    prepare.assert_not_called()
    assert coupling.calculator is None and coupling.comm.rank == 1


def test_run_dispatches_the_existing_accepted_checkpoint_lifecycle(tmp_path):
    output = tmp_path / "new-run"
    restart = tmp_path / "checkpoint.json"
    config = tmp_path / "config.json"
    with patch("qmdpa.cli._run", return_value="delegated") as run:
        assert adaptive_cli.run(config, output, restart=restart) == "delegated"
    run.assert_called_once_with(
        config,
        output,
        restart=restart,
        config_loader=load_adaptive_config,
        setup_pair=adaptive_cli._zero_pair,
        make_coupling=adaptive_cli._adaptive_coupling,
    )


def test_main_preserves_error_exit_and_restart_arguments(tmp_path, capsys):
    config = tmp_path / "config.json"
    output = tmp_path / "out"
    restart = tmp_path / "checkpoint.json"
    with patch.object(adaptive_cli, "run") as run:
        assert (
            adaptive_cli.main([str(config), "--output", str(output), "--restart", str(restart)])
            == 0
        )
    run.assert_called_once_with(config, output, restart=restart)
    with patch.object(adaptive_cli, "run", side_effect=RuntimeError("unsupported state")):
        assert adaptive_cli.main([str(config), "--output", str(output)]) == 1
    assert "unsupported state" in capsys.readouterr().err


def test_no_jit_loss_is_rejected_before_qm_construction(tmp_path):
    config = load(tmp_path)
    torch = torch_state()
    value = SimpleNamespace(no_jit=False)
    with (
        patch.dict(os.environ, RAW_ENV),
        patch("qmdpa.inference.prepare_raw_inference", return_value=torch),
        patch("qmdpa.backends.DeepMDBackend", return_value=value),
        patch("qmdpa.providers.make_qm_provider") as high,
        pytest.raises(RuntimeError, match="no_jit=True"),
    ):
        adaptive_cli._adaptive_coupling(TrimerDriver(), None, config)
    high.assert_not_called()


def test_changed_provider_state_cannot_silently_change_partition_charge(attached):
    a = attached
    a.highs[0].charge = 1
    with pytest.raises(EvaluationError, match="High-provider"):
        a.coupling.run(0)
    assert a.coupling.last_step is None and a.lmp.energy_calls[-1][1] == 0


def test_help_does_not_construct_native_or_scientific_backends(capsys):
    with patch("qmdpa.cli._run") as run, pytest.raises(SystemExit) as error:
        adaptive_cli.main(["--help"])
    assert error.value.code == 0
    run.assert_not_called()
    assert "--restart" in capsys.readouterr().out
