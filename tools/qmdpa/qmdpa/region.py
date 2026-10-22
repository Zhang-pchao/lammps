# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Stable atom identity and explicit periodic image handling."""

from dataclasses import dataclass

import numpy as np

from .core import Geometry


def box_matrix(box):
    """Return lattice vectors as rows for a restricted triclinic LAMMPS box."""
    lo, hi, xy, yz, xz, _, _ = box
    lx, ly, lz = np.asarray(hi) - np.asarray(lo)
    return np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)


def unwrap(positions, images, cell):
    """Use recorded integer images; never select a fresh nearest image per step."""
    positions = np.asarray(positions, dtype=float)
    images = np.asarray(images)
    cell = np.asarray(cell, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or images.shape != positions.shape:
        raise ValueError("positions and images must both have shape (N, 3)")
    if not np.issubdtype(images.dtype, np.integer):
        raise ValueError("image flags must be integers")
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("cell must be a finite (3, 3) matrix")
    return positions + images @ cell


@dataclass(frozen=True)
class FixedRegion:
    """Fixed complete-molecule atom IDs, in the declared backend order.

    The caller supplies complete molecules and initially consistent image flags.
    No topology inference, dynamic reassignment, or covalent boundary is implied.
    """

    atom_ids: tuple[int, ...]
    symbols: tuple[str, ...]

    def __post_init__(self):
        validated = Geometry(self.atom_ids, self.symbols, np.zeros((len(self.atom_ids), 3)))
        object.__setattr__(self, "atom_ids", validated.atom_ids)
        object.__setattr__(self, "symbols", validated.symbols)

    def assemble(self, packets):
        """Gather owned atoms into one isolated geometry, detecting loss/duplication."""
        by_id = {}
        expected = set(self.atom_ids)
        for ids, positions in packets:
            if len(ids) != len(positions):
                raise ValueError("atom IDs and positions have different lengths")
            for atom_id, position in zip(ids, positions):
                if atom_id not in expected:
                    raise ValueError(f"unexpected region atom ID {atom_id}")
                if atom_id in by_id:
                    raise ValueError(f"region atom ID {atom_id} has multiple owners")
                by_id[atom_id] = position
        missing = expected.difference(by_id)
        if missing:
            raise ValueError(f"missing region atom IDs: {sorted(missing)}")
        return Geometry(self.atom_ids, self.symbols, [by_id[i] for i in self.atom_ids])

    def local_forces(self, local_ids, correction):
        """Map by current IDs, including empty ranks and reordered local arrays."""
        forces = np.asarray(correction, dtype=float)
        if forces.shape != (len(self.atom_ids), 3) or not np.isfinite(forces).all():
            raise ValueError("invalid region correction forces")
        rows = {atom_id: row for row, atom_id in enumerate(self.atom_ids)}
        result = np.zeros((len(local_ids), 3))
        for local, atom_id in enumerate(local_ids):
            if atom_id in rows:
                result[local] = forces[rows[atom_id]]
        return result
