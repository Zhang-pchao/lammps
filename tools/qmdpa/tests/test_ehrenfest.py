"""Independent two-level identities for the Ehrenfest force and propagator."""
import itertools

import numpy as np
import pytest

from qmdpa.core import EvaluationError
from qmdpa.ehrenfest import (
    HBAR_EV_FS,
    SurfacePair,
    amplitudes,
    correspondence,
    electronic_step,
    mapped_pair,
    mean_field,
)


def pair():
    return SurfacePair([-0.3, 0.7], np.zeros((2, 1, 3)), [[0.4, -0.2, 0.1]])


def test_exact_fixed_generator_and_time_reversal():
    p = pair()
    c = amplitudes([np.sqrt(0.3), 1j * np.sqrt(0.7)])
    delta = np.array([[0.07, -0.02, 0.03]])
    dt = 0.11
    result = electronic_step(c, p, p, delta, dt)
    tau = float(np.sum(delta * p.nac01) / dt)
    h = np.array([[-0.5, -1j * HBAR_EV_FS * tau],
                  [1j * HBAR_EV_FS * tau, 0.5]])
    omega = np.sqrt(0.5**2 + (HBAR_EV_FS * tau)**2)
    expected = np.exp(-1j * 0.2 * dt / HBAR_EV_FS) * (
        np.cos(omega * dt / HBAR_EV_FS) * np.eye(2)
        - 1j * np.sin(omega * dt / HBAR_EV_FS) * h / omega) @ c
    np.testing.assert_allclose(result, expected, atol=2e-15)
    reverse = electronic_step(result.conj(), p, p, -delta, dt).conj()
    np.testing.assert_allclose(reverse, c, atol=2e-15)
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_zero_coupling_and_pure_derivative_rotation():
    p = SurfacePair([0.2, 1.2], np.zeros((2, 1, 3)), np.zeros((1, 3)))
    c = amplitudes([0.6, 0.8j])
    np.testing.assert_allclose(electronic_step(c, p, p, [[1., 0, 0]], 0.4),
                               c * np.exp(-1j * p.energies * 0.4 / HBAR_EV_FS), atol=1e-15)
    rotation = SurfacePair([0, 0], np.zeros((2, 1, 3)), [[1., 0, 0]])
    value = electronic_step([1, 0], rotation, rotation, [[0.3, 0, 0]], 0.1)
    np.testing.assert_allclose(value, [np.cos(0.3), np.sin(0.3)], atol=1e-15)


def test_complete_force_against_fixed_diabatic_wavefunction_derivative():
    x = 0.37
    theta_prime = 0.6
    c = amplitudes([np.sqrt(0.4), np.sqrt(0.6) * np.exp(0.3j)])

    def rotation(position):
        theta = theta_prime * position
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def energies(position):
        return np.array([0.2 * position, 0.8 - 0.3 * position])

    psi = rotation(x) @ c
    p = SurfacePair(energies(x), [[[-0.2, 0, 0]], [[0.3, 0, 0]]], [[-theta_prime, 0, 0]])
    result = mean_field(p, c)
    for step in (1e-4, 5e-5):
        values = []
        for sign in (-1, 1):
            pos = x + sign * step
            h = rotation(pos) @ np.diag(energies(pos)) @ rotation(pos).T
            values.append(np.vdot(psi, h @ psi).real)
        assert result.forces[0, 0] == pytest.approx(-(values[1]-values[0])/(2*step), abs=3e-9)
    population_force = np.sum(np.abs(c)**2 * p.forces[:, 0, 0])
    assert abs(population_force - result.forces[0, 0]) > 0.1


@pytest.mark.parametrize("roots", [(0, 1), (1, 0)])
@pytest.mark.parametrize("signs", list(itertools.product([-1, 1], repeat=2)))
def test_signed_permutation_preserves_force_and_evolution(roots, signs):
    p = pair()
    c = amplitudes([np.sqrt(0.35), np.sqrt(0.65) * np.exp(0.42j)])
    matrix = np.zeros((2, 2))
    for i in range(2):
        matrix[i, roots[i]] = signs[i]
    assert correspondence(matrix) == (roots, signs)
    mapped = mapped_pair(p, roots, signs)
    new_c = np.array(signs) * c[list(roots)]
    np.testing.assert_allclose(mean_field(mapped, new_c).forces, mean_field(p, c).forces, atol=1e-15)
    delta = [[0.04, -0.03, 0.02]]
    a = electronic_step(new_c, mapped, mapped, delta, 0.2)
    b = electronic_step(c, p, p, delta, 0.2)
    np.testing.assert_allclose(a, np.array(signs) * b[list(roots)], atol=1e-15)


@pytest.mark.parametrize("bad", [[1, 1], [0, 0], [float('nan'), 0], [True, False], ['1', '0'], [[1, 0]]])
def test_amplitude_errors_are_not_repaired(bad):
    with pytest.raises((ValueError, EvaluationError)):
        amplitudes(bad)


@pytest.mark.parametrize("matrix", [np.full((2, 2), 0.5), np.eye(2)*0.3, np.eye(2)*1.1])
def test_ambiguous_or_lost_states_rejected(matrix):
    with pytest.raises(EvaluationError):
        correspondence(matrix)


@pytest.mark.parametrize("dt", [0, -1, True, float('inf'), [0.1]])
def test_invalid_step_rejected(dt):
    with pytest.raises(ValueError):
        electronic_step([1, 0], pair(), pair(), [[0, 0, 0]], dt)


def test_pair_is_immutable_and_rejects_incomplete_forces():
    p = pair()
    for a in (p.energies, p.forces, p.nac01):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(ValueError):
        SurfacePair([0, 1], np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        SurfacePair([0, 1], np.zeros((2, 1, 3)), [[float('nan'), 0, 0]])
