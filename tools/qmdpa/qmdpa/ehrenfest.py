# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Bounded two-state Ehrenfest algebra in eV, Angstrom and femtoseconds.

The coupling is d01 = <0|grad 1>. This module does not solve an electronic
structure problem or integrate nuclei. Native LAMMPS owns nuclear integration.
"""

from dataclasses import dataclass

import numpy as np

from .core import Evaluation, EvaluationError, _readonly_array

# hbar = h/(2*pi), using exact SI h and elementary charge, converted to eV fs.
HBAR_EV_FS = 6.62607015e-34 / (2 * np.pi * 1.602176634e-19) * 1e15
NORM_TOLERANCE = 1e-10


def _real(values, name):
    if np.asarray(values).dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers, not booleans or strings")
    return _readonly_array(values, name)


def amplitudes(values):
    """Validate a normalized pair; never normalize or otherwise repair it."""
    array = np.asarray(values)
    if array.shape != (2,) or array.dtype.kind not in "fciu":
        raise ValueError("Two numeric complex amplitudes are required")
    array = np.asarray(array, dtype=np.complex128)
    if not np.isfinite(array).all():
        raise ValueError("Electronic amplitudes must be finite")
    norm = float(np.vdot(array, array).real)
    if not np.isfinite(norm) or abs(norm - 1) > NORM_TOLERANCE:
        raise EvaluationError("Electronic amplitudes lost unit norm")
    return np.frombuffer(array.tobytes(), dtype=np.complex128)


@dataclass(frozen=True, eq=False)
class SurfacePair:
    """Complete hybrid state energies/forces and antisymmetric d01 on all atoms.

    In fixed-region ME the environment rows of d01 are zero. State forces must
    already include the full DPA baseline and fragment correction exactly once.
    """

    energies: np.ndarray
    forces: np.ndarray
    nac01: np.ndarray

    def __post_init__(self):
        energies = _real(self.energies, "State energies")
        forces = _real(self.forces, "State forces")
        nac = _real(self.nac01, "Derivative coupling")
        if (energies.shape != (2,) or forces.ndim != 3 or forces.shape[0] != 2
                or forces.shape[1] < 1 or forces.shape[2] != 3
                or nac.shape != forces.shape[1:]):
            raise ValueError("Require two energies, forces (2,N,3) and d01 (N,3)")
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "nac01", nac)


def mean_field(pair, coefficients):
    """Expectation potential and full Ehrenfest force, including coherence."""
    if not isinstance(pair, SurfacePair):
        raise TypeError("Ehrenfest forces require a SurfacePair")
    c = amplitudes(coefficients)
    populations = np.abs(c)**2
    energy = float(populations @ pair.energies)
    forces = np.einsum("i,iaj->aj", populations, pair.forces)
    gap = pair.energies[1] - pair.energies[0]
    forces -= 2 * np.real(c[0].conjugate() * c[1]) * gap * pair.nac01
    return Evaluation(energy, forces)


def electronic_step(coefficients, previous, current, displacement, timestep_fs):
    """Unitary endpoint-averaged propagation over one native Verlet step.

    Nuclear secant velocity contracts the endpoint-averaged derivative coupling.
    The generator is diag(U) - i*hbar*(v.d). A repeated callback must be detected
    by the caller and must not invoke this function a second time for that step.
    """
    if not isinstance(previous, SurfacePair) or not isinstance(current, SurfacePair):
        raise TypeError("Electronic propagation requires two SurfacePair endpoints")
    dt_array = _real(timestep_fs, "Electronic timestep")
    if dt_array.shape != () or float(dt_array) <= 0:
        raise ValueError("Electronic timestep must be a positive scalar")
    dt = float(dt_array)
    delta = _real(displacement, "Nuclear displacement")
    if delta.shape != previous.nac01.shape or delta.shape != current.nac01.shape:
        raise ValueError("Both endpoint atom orders and the displacement must agree")
    c = amplitudes(coefficients)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rate = float(np.sum(delta * (previous.nac01 + current.nac01) / (2 * dt)))
        generator = np.diag((previous.energies + current.energies) / 2).astype(complex)
        generator[0, 1] = -1j * HBAR_EV_FS * rate
        generator[1, 0] = 1j * HBAR_EV_FS * rate
        eigenvalues, eigenvectors = np.linalg.eigh(generator)
        phases = np.exp(-1j * eigenvalues * dt / HBAR_EV_FS)
        result = eigenvectors @ (phases * (eigenvectors.conj().T @ c))
    return amplitudes(result)


def correspondence(overlap, *, minimum_overlap=0.8, minimum_margin=0.2):
    """Match two real wavefunctions to fixed references with signed phases.

    Reject weak/ambiguous assignments instead of silently selecting a new pair.
    Returns current root indexes and signs, both in fixed-reference order.
    """
    matrix = _real(overlap, "Signed wavefunction overlap")
    limits = _real([minimum_overlap, minimum_margin], "State matching limits")
    if matrix.shape != (2, 2) or not 0 < limits[0] <= 1 or not 0 < limits[1] <= 2:
        raise ValueError("Require a 2 by 2 overlap and positive bounded matching limits")
    if np.max(np.abs(matrix)) > 1 + NORM_TOLERANCE:
        raise EvaluationError("Wavefunction overlap exceeds unit norm")
    diagonal = abs(matrix[0, 0]) + abs(matrix[1, 1])
    crossed = abs(matrix[0, 1]) + abs(matrix[1, 0])
    if abs(diagonal - crossed) < limits[1]:
        raise EvaluationError("Two-state correspondence is ambiguous")
    roots = (0, 1) if diagonal > crossed else (1, 0)
    selected = np.array([matrix[i, roots[i]] for i in range(2)])
    if np.min(np.abs(selected)) < limits[0]:
        raise EvaluationError("Two-state reference overlap is too small")
    return roots, tuple(int(x) for x in np.sign(selected))


def mapped_pair(pair, roots, signs):
    """Apply the same permutation and signed gauge to surfaces and d01."""
    if (not isinstance(pair, SurfacePair) or len(roots) != 2
            or any(type(x) is not int for x in roots) or tuple(roots) not in ((0, 1), (1, 0))
            or len(signs) != 2 or any(type(x) is not int or x not in (-1, 1) for x in signs)):
        raise ValueError("A pair requires a two-root permutation and two signs")
    orientation = 1 if tuple(roots) == (0, 1) else -1
    return SurfacePair(pair.energies[list(roots)], pair.forces[list(roots)],
                       orientation * signs[0] * signs[1] * pair.nac01)
