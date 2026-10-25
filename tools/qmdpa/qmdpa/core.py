# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Fixed-region subtractive energies and forces, in eV and Angstrom.

Cell vectors are rows. A missing cell means an isolated system; a supplied
cell means three-dimensional periodicity. Partial periodicity is not implied.
No electronic-structure or machine-learning package is imported here.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EvaluationError(RuntimeError):
    """A backend did not produce a usable energy and force evaluation."""


def _atom_ids(values):
    ids = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Atom IDs must be positive integers, not booleans")
        try:
            atom_id = operator.index(value)
        except TypeError as exc:
            raise ValueError("Atom IDs must be integers") from exc
        if atom_id <= 0:
            raise ValueError("Atom IDs must be positive")
        ids.append(atom_id)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Atom IDs must be nonempty and distinct")
    return tuple(ids)


def _readonly_array(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    # Back the snapshot with immutable bytes, so callers cannot re-enable writes.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class Geometry:
    """A geometry snapshot with stable IDs and coordinates in Angstrom."""

    atom_ids: tuple[int, ...]
    symbols: tuple[str, ...]
    positions: np.ndarray
    cell: np.ndarray | None = None

    def __post_init__(self):
        ids = _atom_ids(self.atom_ids)
        symbols = tuple(self.symbols)
        if len(symbols) != len(ids) or any(
            not isinstance(symbol, str) or not symbol for symbol in symbols
        ):
            raise ValueError("Each atom ID needs a nonempty element symbol")
        positions = _readonly_array(self.positions, "Positions")
        if positions.shape != (len(ids), 3):
            raise ValueError("Positions must have shape (number of atoms, 3)")
        cell = None
        if self.cell is not None:
            cell = _readonly_array(self.cell, "Cell")
            if cell.shape != (3, 3):
                raise ValueError("Cell must have shape (3, 3)")
            determinant = np.linalg.det(cell)
            if not np.isfinite(determinant) or determinant <= 0.0:
                raise ValueError("Cell must be a right-handed, nonsingular 3 by 3 matrix")
        object.__setattr__(self, "atom_ids", ids)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cell", cell)

    @property
    def periodic(self):
        """Whether the geometry has three-dimensional periodic boundaries."""
        return self.cell is not None

    def select(self, atom_ids, *, isolated=False):
        """Select atoms in the requested ID order, optionally removing the cell."""
        ids = _atom_ids(atom_ids)
        lookup = {atom_id: index for index, atom_id in enumerate(self.atom_ids)}
        try:
            indices = [lookup[atom_id] for atom_id in ids]
        except KeyError as exc:
            raise ValueError(f"Selected atom ID {exc.args[0]} is absent") from exc
        return Geometry(
            ids,
            tuple(self.symbols[index] for index in indices),
            self.positions[indices],
            None if isolated else self.cell,
        )


@dataclass(frozen=True, eq=False)
class Evaluation:
    """A finite energy in eV and forces in eV/Angstrom, in input atom order."""

    energy: float
    forces: np.ndarray

    def __post_init__(self):
        value = np.asarray(self.energy)
        if value.ndim != 0 or value.dtype.kind not in "fiu":
            raise ValueError("Energy must be a real scalar")
        energy = float(value)
        if not np.isfinite(energy):
            raise ValueError("Energy must be finite")
        forces = _readonly_array(self.forces, "Forces")
        if forces.ndim != 2 or forces.shape[1] != 3 or forces.shape[0] == 0:
            raise ValueError("Forces must have shape (number of atoms, 3)")
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "forces", forces)


class Backend(Protocol):
    """Minimal interface implemented by each energy and force provider."""

    def evaluate(self, geometry: Geometry) -> Evaluation:
        """Return an evaluation in exactly the geometry's atom order."""
        ...


def _evaluate(backend, geometry):
    result = backend.evaluate(geometry)
    if not isinstance(result, Evaluation):
        raise EvaluationError("Backend must return an Evaluation")
    if result.forces.shape != geometry.positions.shape:
        raise EvaluationError("Backend force count does not match the input atoms")
    return result


class SubtractiveCalculator:
    """Combine full low-level energy with an explicitly bounded region correction.

    ``low`` evaluates the isolated QM region. ``full_low`` may be a separate
    instance with different full-system charge/spin or atomic parameters.
    It defaults to ``low`` when those settings are identical.
    """

    def __init__(self, low: Backend, high: Backend, *, full_low: Backend | None = None,
                 region_boundary="isolated"):
        if region_boundary not in ("isolated", "periodic-same-cell"):
            raise ValueError("region_boundary must be isolated or periodic-same-cell")
        self.region_boundary = region_boundary
        self.low = low
        self.high = high
        self.full_low = low if full_low is None else full_low

    def correction(self, qm_geometry: Geometry) -> Evaluation:
        """Return high minus low, in the supplied QM atom order."""
        if self.region_boundary == "isolated" and qm_geometry.periodic:
            raise ValueError("The QM correction requires an explicitly isolated geometry")
        if self.region_boundary == "periodic-same-cell" and not qm_geometry.periodic:
            raise ValueError("The periodic QM correction requires an explicit cell")
        low = _evaluate(self.low, qm_geometry)
        high = _evaluate(self.high, qm_geometry)
        return Evaluation(high.energy - low.energy, high.forces - low.forces)

    def evaluate(self, full_geometry: Geometry, qm_geometry: Geometry) -> Evaluation:
        """Return the mixed energy and forces in the full-system atom order.

        QM coordinates may differ from the selected full-system coordinates
        only by periodic image shifts. Their mapping has unit Cartesian
        derivative; adaptive membership and link atoms are not supported.
        """
        if self.region_boundary == "periodic-same-cell" and (
            not full_geometry.periodic or not qm_geometry.periodic
            or not np.array_equal(full_geometry.cell, qm_geometry.cell)
        ):
            raise ValueError("Periodic full and QM geometries must use the same cell")
        selected = full_geometry.select(qm_geometry.atom_ids)
        if selected.symbols != qm_geometry.symbols:
            raise ValueError("QM and full-system element symbols differ for the same IDs")
        displacement = qm_geometry.positions - selected.positions
        if full_geometry.periodic:
            fractional = np.linalg.solve(full_geometry.cell.T, displacement.T).T
            consistent = np.allclose(fractional, np.rint(fractional), atol=1e-9, rtol=0.0)
        else:
            consistent = np.allclose(displacement, 0.0, atol=1e-10, rtol=0.0)
        if not consistent:
            raise ValueError("QM coordinates must match the selected atoms and their images")
        correction = self.correction(qm_geometry)
        full = _evaluate(self.full_low, full_geometry)
        lookup = {atom_id: index for index, atom_id in enumerate(full_geometry.atom_ids)}
        indices = [lookup[atom_id] for atom_id in qm_geometry.atom_ids]
        forces = full.forces.copy()
        forces[indices] += correction.forces
        return Evaluation(full.energy + correction.energy, forces)
