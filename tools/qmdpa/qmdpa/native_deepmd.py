#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Owned native DeePMD evaluations of finite atom-ID domains on COMM_SELF."""

from pathlib import Path

import numpy as np

from .config import file_digest
from .core import Evaluation
from .runtime import _quoted


def _create_lammps(style):
    from lammps import lammps
    from mpi4py import MPI

    from .execution import native_arguments, register_native_cleanup

    config = {"execution": {"pair_style": style}}
    lmp = lammps(cmdargs=native_arguments(config), comm=MPI.COMM_SELF)
    register_native_cleanup(lmp, config)
    if lmp.extract_setting("world_size") != 1:
        lmp.close()
        raise RuntimeError("The native domain evaluator must own a one-rank communicator")
    return lmp


class NativeDeepMDDomainBackend:
    """Reuse one native model while rebuilding each exact finite environment.

    Dense local IDs are only a transport mapping; returned forces follow the
    requested global atom IDs. Rebuild all atoms and neighbors on each call,
    retaining the pair/model object. This also covers changing domain sizes.
    The owner must call close_checked(), including after a failed evaluation.
    Kokkos finalization belongs to process exit, never an individual domain.
    """

    def __init__(self, model, *, model_sha256, type_symbols, pair_style, charge_spin=(0, 1)):
        self.model = str(Path(model).resolve())
        if Path(self.model).suffix != ".pt2" or file_digest(self.model) != model_sha256:
            raise ValueError("Native domain inference requires the identified .pt2 archive")
        if pair_style not in ("deepmd", "deepmd/kk"):
            raise ValueError("Native domains require deepmd or deepmd/kk")
        symbols = tuple(type_symbols)
        if not symbols or len(set(symbols)) != len(symbols) or any(
                not isinstance(s, str) or not s.isalpha() for s in symbols):
            raise ValueError("Native domain types must be distinct element symbols")
        if tuple(charge_spin) != (0, 1) or any(type(x) is not int for x in charge_spin):
            raise ValueError("Native domain inference requires explicit neutral-singlet conditioning")
        _quoted(self.model)
        self.model_sha256, self.type_symbols = model_sha256, symbols
        self.pair_style, self.charge_spin = pair_style, (0, 1)
        self._lmp = None
        self._closed = False
        self._calls = self._failures = 0
        self._domains = set()

    def _open(self, bounds):
        lmp = _create_lammps(self.pair_style)
        self._lmp = lmp  # Retain ownership even when setup fails.
        lmp.commands_list([
            "units metal", "atom_style atomic", "atom_modify map array sort 0 0",
            "boundary f f f", "region qmdpa_domain block " + bounds + " units box",
            f"create_box {len(self.type_symbols)} qmdpa_domain", "mass * 1.0",
            f"pair_style {self.pair_style} {_quoted(self.model)} charge_spin 0 1",
            "pair_coeff * * " + " ".join(self.type_symbols),
            "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check no",
            "thermo_style custom step pe", "thermo_modify norm no", "thermo 1",
        ])

    def evaluate(self, geometry):
        if self._closed:
            raise RuntimeError("Native domain evaluation cannot follow checked cleanup")
        if geometry.periodic:
            raise ValueError("Native environment domains must be finite")
        ordered = geometry.select(tuple(sorted(geometry.atom_ids)))
        try:
            types = [self.type_symbols.index(s)+1 for s in ordered.symbols]
        except ValueError as exc:
            raise ValueError("A domain element is absent from the declared type map") from exc
        low, high = ordered.positions.min(axis=0)-6., ordered.positions.max(axis=0)+6.
        box = " ".join(f"{x:.17g}" for pair in zip(low, high, strict=True) for x in pair)
        try:
            if self._lmp is None:
                self._open(box)
            lmp = self._lmp
            if lmp.extract_global("pair_style") != self.pair_style:
                raise ValueError("The actual native domain pair differs from its declaration")
            lmp.command("delete_atoms group all compress yes")
            lmp.command("change_box all " + " ".join(
                f"{axis} final {lo:.17g} {hi:.17g}" for axis, lo, hi in
                zip("xyz", low, high, strict=True)) + " units box")
            count = len(ordered.atom_ids)
            created = lmp.create_atoms(count, list(range(1, count+1)), types,
                                       ordered.positions.reshape(-1).tolist())
            if created != count:
                raise RuntimeError("Native domain atom creation lost atoms")
            lmp.command("run 0 post no")
            if lmp.get_natoms() != count:
                raise RuntimeError("Native domain evaluation changed the atom count")
            native_ids = np.ctypeslib.as_array(lmp.gather_atoms("id", 0, 1), shape=(count,))
            native_types = np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1), shape=(count,))
            if not np.array_equal(native_ids, np.arange(1, count+1)) or not np.array_equal(native_types, types):
                raise RuntimeError("Native domain atom mapping differs")
            forces = np.ctypeslib.as_array(lmp.gather_atoms("f", 1, 3), shape=(3*count,)).copy().reshape(-1, 3)
            rows = {a:i for i,a in enumerate(ordered.atom_ids)}
            value = Evaluation(float(lmp.get_thermo("pe")), forces[[rows[a] for a in geometry.atom_ids]])
            self._calls += 1
            self._domains.add((ordered.atom_ids, ordered.symbols))
            return value
        except Exception:
            self._failures += 1
            raise

    def close_checked(self):
        if not self._closed:
            self._closed = True
            if self._lmp is not None:
                self._lmp.close()
                self._lmp = None

    @property
    def statistics(self):
        return {
            "kind": "native-deepmd-domain", "model_sha256": self.model_sha256,
            "pair_style": self.pair_style, "communicator": "MPI.COMM_SELF",
            "successful_calls": self._calls, "failed_calls": self._failures,
            "closed": self._closed,
            "domains": [{"atom_ids": list(ids), "symbols": list(symbols)}
                        for ids, symbols in sorted(self._domains)],
        }
