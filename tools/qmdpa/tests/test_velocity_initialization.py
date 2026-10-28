#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Actual native thermal initialization for consecutive and sparse atom IDs."""

import numpy as np
import pytest

from qmdpa.cli import _initialize_velocities


def test_native_velocity_initialization_preserves_ids_and_legacy_sequence(tmp_path):
    native = pytest.importorskip("lammps")
    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    config = {"temperature_K": 30.0, "velocity_seed": 76123}
    rows = [(0., 0., 0.), (1.2, 0., 0.), (-.6, .9, .02), (-.6, -.9, -.02),
            (3.9, .2, .1), (3., .05, .12), (4.2, 1.1, 0.)]
    for ids in [list(range(1, 8)), [101, 102, 103, 104, 201, 202, 203]]:
        path = tmp_path / "atoms.data"
        path.write_text("Velocity test\n\n7 atoms\n1 atom types\n\n"
                        "-8 12 xlo xhi\n-8 12 ylo yhi\n-8 12 zlo zhi\n\n"
                        "Masses\n\n1 1.0\n\nAtoms # atomic\n\n" +
                        "\n".join(f"{a} 1 {x} {y} {z}" for a, (x, y, z) in zip(ids, rows)) + "\n")
        shared = comm.bcast(str(path) if comm.rank == 0 else None, root=0)
        lmp = native.lammps(comm=comm, cmdargs=["-log", "none", "-screen", "none"])
        try:
            lmp.commands_list(["units metal", "atom_style atomic", "atom_modify map yes",
                               "boundary f f f", f'read_data "{shared}"',
                               "pair_style zero 6.0", "pair_coeff * *"])
            _initialize_velocities(lmp, config)
            count = lmp.extract_setting("nlocal")
            tags = lmp.numpy.extract_atom("id")[:count].copy()
            first = lmp.numpy.extract_atom("v")[:count].copy()
            assert np.isfinite(first).all()
            all_tags = [a for part in comm.allgather(tags.tolist()) for a in part]
            assert sorted(all_tags) == ids
            momentum = np.sum([np.sum(v, axis=0) for v in comm.allgather(first)], axis=0)
            assert np.max(np.abs(momentum)) < 1e-12
            assert sum(np.sum(v * v) for v in comm.allgather(first)) > 0
            suffix = "" if ids[0] == 1 else " loop geom"
            lmp.command("velocity all create 30.0 76123 mom yes rot yes dist gaussian" + suffix)
            np.testing.assert_array_equal(lmp.numpy.extract_atom("v")[:count], first)
        finally:
            lmp.close()
