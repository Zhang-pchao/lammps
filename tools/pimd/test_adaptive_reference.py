#!/usr/bin/env python3
"""Independent endpoint, force, measure and input checks for the reference model."""

import unittest

import numpy as np
from adaptive_reference import (
    AdaptivePathMeasure,
    SphereResolution,
    UniformResolution,
    harmonic_potential,
)


class AdaptiveReferenceTests(unittest.TestCase):
    def test_normal_modes_and_uniform_quantum_energy(self):
        rng = np.random.default_rng(71)
        for p in (1, 2, 3, 4, 8, 16):
            model = AdaptivePathMeasure(p, [1.0, 16.0], 1.7, 0.8, UniformResolution(1))
            c, z = rng.normal(size=(2, 3)), rng.normal(size=(p - 1, 2, 3))
            q = model.positions(c, z)
            np.testing.assert_allclose(q.mean(axis=0), c, atol=2e-14)
            np.testing.assert_allclose(
                model.modes.T @ model.modes, np.eye(p - 1), atol=2e-14
            )
            energy = model.evaluate(c, z, harmonic_potential(0.7))[0]
            explicit = (
                0.35 * (q * q).sum() / p
                + p
                / (2 * (1.7 * 0.8) ** 2)
                * (
                    (q - np.roll(q, 1, axis=0)) ** 2 * np.array([1, 16])[None, :, None]
                ).sum()
            )
            self.assertAlmostEqual(float(energy), float(explicit), places=9)

    def test_exact_classical_endpoint_with_auxiliary_modes(self):
        rng = np.random.default_rng(72)
        for p in (1, 2, 4, 16):
            model = AdaptivePathMeasure(p, [1, 16], 1.3, 1, UniformResolution(0))
            c, z = rng.normal(size=(3, 2, 2)), rng.normal(size=(3, p - 1, 2, 2))
            q = model.positions(c, z)
            np.testing.assert_array_equal(q, np.broadcast_to(c[:, None], q.shape))
            _, fc, fz, fa = model.evaluate(c, z, harmonic_potential(1.2))
            np.testing.assert_allclose(fc, -1.2 * c, atol=1e-14)
            np.testing.assert_allclose(fz, -model.stiffness[..., None] * z)
            np.testing.assert_array_equal(fa, 0)

    def test_cyclic_hessian_statistics(self):
        # Independent bead-coordinate precision matrix, without the map's modes.
        for p in (1, 2, 4, 8, 16):
            for amplitude in (0.0, 0.15, 0.5, 1.0):
                beta, mass, hbar, k = 1.9, 1.3, 0.7, 2.1
                model = AdaptivePathMeasure(
                    p, [mass], beta, hbar, UniformResolution(amplitude)
                )
                variance = 1 / (beta * k) + np.sum(
                    amplitude**2
                    / (beta * p * (model.stiffness[:, 0] + amplitude**2 * k / p))
                )
                if amplitude == 0:
                    expected = 1 / (beta * k)
                else:
                    incidence = np.eye(p) - np.roll(np.eye(p), 1, axis=0)
                    precision = (
                        k * np.eye(p) / p
                        + mass
                        * p
                        / (beta * hbar * amplitude) ** 2
                        * incidence.T
                        @ incidence
                    )
                    expected = np.trace(np.linalg.inv(beta * precision)) / p
                self.assertAlmostEqual(float(variance), float(expected), places=11)

    def test_nonlinear_coordinate_mode_anchor_forces(self):
        rng = np.random.default_rng(73)
        c = np.array([[0.9, 0.1], [1.1, -0.2], [0.1, 0.1]])
        z = rng.normal(scale=0.35, size=(3, 3, 2))
        region = SphereResolution((0.0, 0.0), 0.4, 1.5)
        model = AdaptivePathMeasure(4, [1.0, 2.0, 16.0], 1.6, 1, region)

        def potential(q):
            total = q.sum(axis=(-2, -1))
            energy = (
                0.2 * (q * q).sum(axis=(-2, -1))
                + 0.03 * (q**4).sum(axis=(-2, -1))
                + 0.17 * np.sin(total) ** 2
            )
            force = (
                -0.4 * q
                - 0.12 * q**3
                - (0.34 * np.sin(total) * np.cos(total))[..., None, None]
            )
            return energy, force

        _, fc, fz, fa = model.evaluate(c, z, potential)
        h = 2e-6
        for value, force, which in ((c, fc, "centroid"), (z, fz, "mode")):
            for index in np.ndindex(value.shape):
                plus, minus = value.copy(), value.copy()
                plus[index] += h
                minus[index] -= h
                args_plus = (plus, z) if which == "centroid" else (c, plus)
                args_minus = (minus, z) if which == "centroid" else (c, minus)
                numerical = -(
                    model.evaluate(*args_plus, potential)[0]
                    - model.evaluate(*args_minus, potential)[0]
                ) / (2 * h)
                self.assertAlmostEqual(float(force[index]), float(numerical), places=7)
        for axis in range(2):
            plus, minus = np.zeros(2), np.zeros(2)
            plus[axis] = h
            minus[axis] = -h
            ep = AdaptivePathMeasure(
                4, [1, 2, 16], 1.6, 1, SphereResolution(tuple(plus), 0.4, 1.5)
            ).evaluate(c, z, potential)[0]
            em = AdaptivePathMeasure(
                4, [1, 2, 16], 1.6, 1, SphereResolution(tuple(minus), 0.4, 1.5)
            ).evaluate(c, z, potential)[0]
            self.assertAlmostEqual(
                float(fa[axis]), float(-(ep - em) / (2 * h)), places=7
            )
        physical = potential(model.positions(c, z))[1].mean(axis=0)
        np.testing.assert_allclose(
            fc.sum(axis=0) + fa, physical.sum(axis=0), atol=2e-14
        )
        # Deliberately omit transition forces. The independent derivative must reject it.
        self.assertGreater(float(np.max(np.abs(fc - physical))), 0.01)

    def test_conditional_free_energy_and_normalization(self):
        model = AdaptivePathMeasure(4, [1.0], 2.0, 1, UniformResolution(1))
        kappa = model.stiffness[:, 0]
        for a in (0.0, 0.2, 0.7, 1.0):
            oracle = np.prod(np.sqrt(kappa / (kappa + 1.4 * a * a / 4)))
            log_ratio = -0.5 * np.log1p(1.4 * a * a / (4 * kappa)).sum()
            self.assertAlmostEqual(float(np.exp(log_ratio)), float(oracle), places=14)
        # In physical internal coordinates the normalization a^(-K) is essential.
        a = 0.2
        wrong_free_integral_ratio = a**3
        self.assertLess(wrong_free_integral_ratio, 0.01)
        self.assertNotAlmostEqual(wrong_free_integral_ratio, 1.0)

    def test_free_particle_has_no_resolution_entropy_force(self):
        region = SphereResolution((0.0,), 0.4, 1.2)
        model = AdaptivePathMeasure(4, [1], 2.0, 2.0, region)
        z = np.array([[[0.2]], [[-0.3]], [[0.7]]])
        energies = []
        for position in (0.0, 0.4, 0.7, 1.2, 1.8):
            c = np.array([[position]])
            energy, fc, fz, fa = model.evaluate(
                c, z, lambda q: (np.zeros(q.shape[:-2]), np.zeros_like(q))
            )
            energies.append(energy)
            np.testing.assert_array_equal(fc, 0)
            np.testing.assert_array_equal(fa, 0)
            np.testing.assert_allclose(fz, -model.stiffness[..., None] * z)
        np.testing.assert_array_equal(energies, np.full(5, energies[0]))

    def test_native_auxiliary_bead_force_normalization(self):
        from adaptive_reference import auxiliary_bead_pullback

        rng = np.random.default_rng(74)
        region = SphereResolution((0.0, 0.0), 0.4, 1.5)
        c = np.array([[0.9, 0.1], [0.1, 0.1]])
        model = AdaptivePathMeasure(4, [1, 16], 1.7, 0.8, region)
        z = rng.normal(scale=0.3, size=(3, 2, 2))
        y = c[None] + np.einsum("bk,kid->bid", model.modes, z)
        potential = harmonic_potential(1.3)
        q, _, dynamic = auxiliary_bead_pullback(y, region, potential)
        np.testing.assert_allclose(q, model.positions(c, z), atol=1e-14)
        _, fc, fz, _ = model.evaluate(c, z, potential)
        np.testing.assert_allclose(dynamic.mean(axis=0), fc, atol=1e-13)
        fz_potential = fz + model.stiffness[..., None] * z
        np.testing.assert_allclose(
            np.einsum("bk,bid->kid", model.modes, dynamic) / 4, fz_potential, atol=1e-13
        )
        h = 2e-6
        for index in np.ndindex(y.shape):
            plus, minus = y.copy(), y.copy()
            plus[index] += h
            minus[index] -= h
            ep = auxiliary_bead_pullback(plus, region, potential)[1]
            em = auxiliary_bead_pullback(minus, region, potential)[1]
            self.assertAlmostEqual(
                float(dynamic[index] / 4), float(-(ep - em) / (2 * h)), places=8
            )

    def test_smooth_endpoints_and_validation(self):
        region = SphereResolution((0.0,), 0.5, 1.5)
        a, g = region.evaluate(np.array([[0.0], [0.5], [1.5], [2.0]]))
        np.testing.assert_array_equal(a, [1, 1, 0, 0])
        np.testing.assert_array_equal(g, 0)
        for params in (
            (4, [0], 1, 1),
            (0, [1], 1, 1),
            (True, [1], 1, 1),
            (4, [1], 0, 1),
            (4, [np.nan], 1, 1),
        ):
            with self.assertRaises(ValueError):
                AdaptivePathMeasure(*params, region)
        with self.assertRaises(ValueError):
            SphereResolution((0.0,), 1, 1)
        model = AdaptivePathMeasure(4, [1], 1, 1, region)
        with self.assertRaises(ValueError):
            model.positions(np.zeros((1, 1)), np.zeros((4, 1, 1)))
        with self.assertRaises(ValueError):
            model.evaluate(
                np.zeros((1, 1)),
                np.zeros((3, 1, 1)),
                lambda q: (np.zeros(4), np.full_like(q, np.nan)),
            )


if __name__ == "__main__":
    unittest.main()
