#!/usr/bin/env python3
"""Reference adaptive path measure; not a LAMMPS integrator or real-time model.

Coordinates are path centroids and independent internal normal modes. The
physical positions use a smooth spatial amplitude. Keeping the internal Gaussian
measure fixed avoids a singular change of dimension at the classical endpoint.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphereResolution:
    """C2 amplitude: one in the core, zero outside, with a fixed anchor."""

    center: tuple
    inner: float
    outer: float

    def __post_init__(self):
        if (
            not np.isfinite([*self.center, self.inner, self.outer]).all()
            or not self.center
            or self.inner < 0
            or self.outer <= self.inner
        ):
            raise ValueError("Require a finite center and 0 <= inner < outer")

    def evaluate(self, centroids):
        centroids = np.asarray(centroids, dtype=float)
        if centroids.shape[-1] != len(self.center) or not np.isfinite(centroids).all():
            raise ValueError("Centroid dimension or values are invalid")
        delta = centroids - self.center
        radius = np.linalg.norm(delta, axis=-1)
        t = np.clip((radius - self.inner) / (self.outer - self.inner), 0, 1)
        amplitude = (1 - t) ** 3 * (1 + 3 * t + 6 * t**2)
        derivative = -30 * t**2 * (1 - t) ** 2 / (self.outer - self.inner)
        direction = np.divide(
            delta,
            radius[..., None],
            out=np.zeros_like(delta),
            where=radius[..., None] > 0,
        )
        return amplitude, derivative[..., None] * direction


@dataclass(frozen=True)
class UniformResolution:
    amplitude: float

    def __post_init__(self):
        if not np.isfinite(self.amplitude) or not 0 <= self.amplitude <= 1:
            raise ValueError("Amplitude must lie in [0, 1]")

    def evaluate(self, centroids):
        return (np.full(centroids.shape[:-1], self.amplitude), np.zeros_like(centroids))


class AdaptivePathMeasure:
    """Normalized internal-coordinate Hamiltonian with full PES evaluations.

    Potential callbacks receive (..., beads, atoms, dimensions) and return
    per-bead energies (..., beads) and Cartesian forces with the input shape.
    Force results are conjugate to centroids and internal modes, not the physical
    bead force expected by an existing PIMD fix. beta and hbar define the units.
    No PBC, moving-region dynamics, constraint, bias, or NPT adapter is provided.
    """

    def __init__(self, beads, masses, beta, hbar, resolution):
        if (
            isinstance(beads, bool)
            or not isinstance(beads, (int, np.integer))
            or beads < 1
        ):
            raise ValueError("beads must be a positive integer")
        masses = np.asarray(masses, dtype=float)
        if (
            masses.ndim != 1
            or masses.size == 0
            or not np.isfinite(masses).all()
            or np.any(masses <= 0)
            or not np.isfinite([beta, hbar]).all()
            or beta <= 0
            or hbar <= 0
        ):
            raise ValueError("Require positive finite masses, beta and hbar")
        self.beads, self.masses = beads, masses
        self.beta, self.hbar, self.resolution = float(beta), float(hbar), resolution
        # Assemble the cyclic incidence matrix, including the P=1 and P=2 cases.
        incidence = np.eye(beads) - np.roll(np.eye(beads), 1, axis=1)
        values, vectors = np.linalg.eigh(incidence.T @ incidence)
        self.eigenvalues = values[1:]
        self.modes = vectors[:, 1:]
        self.stiffness = (
            beads * self.eigenvalues[:, None] * masses[None, :] / (beta * hbar) ** 2
        )

    def _coordinates(self, centroids, modes):
        c, z = np.asarray(centroids, dtype=float), np.asarray(modes, dtype=float)
        expected = c.shape[:-2] + (self.beads - 1,) + c.shape[-2:]
        if (
            c.ndim < 2
            or c.shape[-2] != self.masses.size
            or c.shape[-1] < 1
            or z.shape != expected
            or not np.isfinite(c).all()
            or not np.isfinite(z).all()
        ):
            raise ValueError("Invalid centroid or internal-mode shape/values")
        amplitude, gradient = self.resolution.evaluate(c)
        u = np.einsum("bk,...kid->...bid", self.modes, z)
        q = c[..., None, :, :] + amplitude[..., None, :, None] * u
        return c, z, amplitude, gradient, u, q

    def positions(self, centroids, modes):
        return self._coordinates(centroids, modes)[-1]

    def evaluate(self, centroids, modes, potential: Callable):
        _c, z, amplitude, gradient, u, q = self._coordinates(centroids, modes)
        bead_energy, force = potential(q)
        bead_energy, force = np.asarray(bead_energy), np.asarray(force)
        if (
            bead_energy.shape != q.shape[:-2]
            or force.shape != q.shape
            or not np.isfinite(bead_energy).all()
            or not np.isfinite(force).all()
        ):
            raise ValueError("Potential returned invalid energy or force")
        spring = self.stiffness[..., None] * z
        energy = bead_energy.mean(axis=-1) + 0.5 * np.sum(spring * z, axis=(-3, -2, -1))
        coupling = np.sum(force * u, axis=-1).mean(axis=-2)
        transition = gradient * coupling[..., None]
        centroid_force = force.mean(axis=-3) + transition
        mode_force = (
            np.einsum("bk,...bid->...kid", self.modes, force)
            * amplitude[..., None, :, None]
            / self.beads
            - spring
        )
        # Derivative with respect to a translated fixed anchor; optional diagnostic.
        anchor_force = -transition.sum(axis=-2)
        return energy, centroid_force, mode_force, anchor_force


def harmonic_potential(stiffness=1.0):
    """Independent Cartesian harmonic wells centered at the origin."""
    if not np.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("Harmonic stiffness must be positive")

    def evaluate(q):
        return 0.5 * stiffness * np.sum(q * q, axis=(-2, -1)), -stiffness * q

    return evaluate


def auxiliary_bead_pullback(auxiliary_positions, resolution, potential):
    """Return physical positions and P-scaled forces conjugate to auxiliary beads.

    This bridges the reference centroid/mode measure to the existing native
    P-times-Hamiltonian convention. Springs are deliberately excluded. It must
    not be applied to an already transformed/normalized integrator force.
    """
    y = np.asarray(auxiliary_positions, dtype=float)
    if y.ndim < 3 or y.shape[-3] < 1 or not np.isfinite(y).all():
        raise ValueError("Require finite (..., beads, atoms, dimensions) coordinates")
    centroid = y.mean(axis=-3)
    displacement = y - centroid[..., None, :, :]
    amplitude, gradient = resolution.evaluate(centroid)
    physical = centroid[..., None, :, :] + amplitude[..., None, :, None] * displacement
    energy, force = potential(physical)
    energy, force = np.asarray(energy), np.asarray(force)
    if (
        force.shape != y.shape
        or energy.shape != y.shape[:-2]
        or not np.isfinite(force).all()
        or not np.isfinite(energy).all()
    ):
        raise ValueError("Potential returned invalid physical forces")
    mean_force = force.mean(axis=-3)
    coupling = (force * displacement).sum(axis=-1).mean(axis=-2)
    pulled = (
        amplitude[..., None, :, None] * force
        + (1 - amplitude[..., None, :, None]) * mean_force[..., None, :, :]
        + gradient[..., None, :, :] * coupling[..., None, :, None]
    )
    return physical, energy.mean(axis=-1), pulled
