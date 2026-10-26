"""Complete-surface accounting and immutable electronic-history transactions."""
import copy
import dataclasses

import numpy as np
import pytest

from qmdpa.core import Evaluation, EvaluationError, Geometry
from qmdpa.ehrenfest import SurfacePair
from qmdpa.ehrenfest_mechanical import EhrenfestStepper, MechanicalPair, _digest


class PairProvider:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.offset = 0.

    def evaluate_pair(self, geometry):
        self.calls += 1
        if self.fail:
            raise EvaluationError("Real-provider stand-in failed")
        return SurfacePair(np.array([-0.2, 0.8]) + self.offset,
                           np.zeros((2, len(geometry.atom_ids), 3)),
                           np.full(geometry.positions.shape, 0.3))


def geometry(x=0.):
    return Geometry((1, 2), ("Li", "H"), [[x, 0, 0], [1.5, 0, 0]])


def stepper(provider=None, checkpoint=None, **kwargs):
    args = {"timestep_fs": 0.01, "initial_amplitudes": [0.6, 0.8j],
            "identity_sha256": "a" * 64, "checkpoint": checkpoint, **kwargs}
    return EhrenfestStepper(provider or PairProvider(), **args)


def test_duplicate_step_and_exact_serialized_continuation():
    provider = PairProvider()
    original = stepper(provider)
    original.evaluate(geometry(), 0)
    original.evaluate(geometry(), 0)
    assert provider.calls == 1
    original.evaluate(geometry(0.002), 1)
    snapshot = original.record()
    original.evaluate(geometry(0.002), 1)
    assert original.record() == snapshot and provider.calls == 2
    restored_provider = PairProvider()
    resumed = stepper(restored_provider, checkpoint=copy.deepcopy(snapshot))
    assert restored_provider.calls == 0
    resumed.evaluate(geometry(0.002), 1)
    assert restored_provider.calls == 1
    assert resumed.record() == snapshot
    expected = original.evaluate(geometry(0.004), 2)
    actual = resumed.evaluate(geometry(0.004), 2)
    np.testing.assert_array_equal(actual.forces, expected.forces)
    assert actual.energy == expected.energy and resumed.record() == original.record()


def test_failure_preserves_previous_snapshot_and_requires_fresh_instance():
    provider = PairProvider()
    active = stepper(provider)
    active.evaluate(geometry(), 0)
    saved = active.record()
    provider.fail = True
    with pytest.raises(EvaluationError, match="stand-in failed"):
        active.evaluate(geometry(0.002), 1)
    assert active.record() == saved
    provider.fail = False
    with pytest.raises(EvaluationError, match="history failed"):
        active.evaluate(geometry(0.002), 1)
    recovered = stepper(PairProvider(), checkpoint=saved)
    recovered.evaluate(geometry(), 0)
    recovered.evaluate(geometry(0.002), 1)
    assert recovered.state.step == 1


@pytest.mark.parametrize("step,atoms", [(0, geometry(0.002)), (2, geometry()),
                                        (1, geometry().select((2, 1)))])
def test_illegal_callback_transition_invalidates_instance(step, atoms):
    active = stepper()
    active.evaluate(geometry(), 0)
    before = active.record()
    with pytest.raises(EvaluationError):
        active.evaluate(atoms, step)
    assert active.record() == before and active.failure is not None


@pytest.mark.parametrize("key", ["amplitudes", "energies_ev", "forces_ev_angstrom",
                                  "d01_per_angstrom", "positions_angstrom"])
def test_modified_history_is_rejected(key):
    active = stepper()
    active.evaluate(geometry(), 0)
    record = active.record()
    record[key] = []
    with pytest.raises(ValueError, match="checksum"):
        stepper(checkpoint=record)


def test_forged_force_history_fails_fresh_provider_check():
    active = stepper()
    active.evaluate(geometry(), 0)
    record = active.record()
    record["forces_ev_angstrom"][0][0][0] = 0.1
    record["sha256"] = _digest({k: v for k, v in record.items() if k != "sha256"})
    resumed = stepper(checkpoint=record)
    with pytest.raises(EvaluationError, match="Restored electronic forces"):
        resumed.evaluate(geometry(), 0)
    assert resumed.failure is not None


@pytest.mark.parametrize("changes", [{"timestep_fs": 0.02}, {"initial_amplitudes": [1, 0]},
                                      {"identity_sha256": "b" * 64}])
def test_changed_physics_cannot_restore(changes):
    active = stepper()
    active.evaluate(geometry(), 0)
    with pytest.raises(ValueError, match="physics or initialization"):
        stepper(checkpoint=active.record(), **changes)


def test_unvalidated_restart_cannot_skip_to_next_step():
    active = stepper()
    active.evaluate(geometry(), 0)
    resumed = stepper(checkpoint=active.record())
    with pytest.raises(EvaluationError, match="not restored"):
        resumed.evaluate(geometry(0.002), 1)


class HarmonicLow:
    def evaluate(self, atoms):
        return Evaluation(0.1 * np.sum(atoms.positions**2), -0.2 * atoms.positions)


class HarmonicHigh:
    def evaluate_pair(self, atoms):
        value = 0.3 * np.sum(atoms.positions**2)
        return SurfacePair([value, value + 1.4], np.repeat((-0.6 * atoms.positions)[None], 2, 0),
                           np.full(atoms.positions.shape, 0.17))


def test_full_hybrid_derivatives_mapping_and_all_qm_cancellation():
    full = Geometry((8, 2, 7), ("H", "Li", "H"), [[-1.4, 0, 0], [0, 0.2, 0], [1.5, 0, 0]])
    high, low = HarmonicHigh(), HarmonicLow()
    pair = MechanicalPair(low, low, high, qm_atom_ids=(2, 7))
    baseline = pair.evaluate_pair(full)
    np.testing.assert_array_equal(baseline.nac01[0], np.zeros(3))
    np.testing.assert_array_equal(baseline.nac01[1:], np.full((2, 3), 0.17))
    for row in range(3):
        for axis in range(3):
            values = []
            for sign in (-1, 1):
                xyz = full.positions.copy()
                xyz[row, axis] += sign * 1e-4
                values.append(pair.evaluate_pair(dataclasses.replace(full, positions=xyz)).energies)
            np.testing.assert_allclose(-(values[1]-values[0])/2e-4, baseline.forces[:, row, axis], atol=2e-11)
    reverse = pair.evaluate_pair(full.select((7, 2, 8)))
    np.testing.assert_allclose(reverse.energies, baseline.energies, atol=1e-15)
    np.testing.assert_array_equal(reverse.forces[:, ::-1], baseline.forces)
    all_qm = MechanicalPair(low, low, high, qm_atom_ids=full.atom_ids).evaluate_pair(full)
    qm_reference = high.evaluate_pair(full)
    np.testing.assert_allclose(all_qm.energies, qm_reference.energies, atol=1e-15)
    np.testing.assert_allclose(all_qm.forces, qm_reference.forces, atol=1e-15)


def test_explicit_restore_requires_fresh_instance_and_rechecks_provider():
    original = stepper()
    original.evaluate(geometry(), 0)
    record = original.record()
    provider = PairProvider()
    restored = stepper(provider)
    restored.restore(record)
    assert provider.calls == 0
    with pytest.raises(EvaluationError, match="fresh instance"):
        restored.restore(record)
    restored.evaluate(geometry(), 0)
    assert provider.calls == 1 and restored.record() == record
    with pytest.raises(EvaluationError, match="fresh instance"):
        original.restore(record)
