# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite fixed-region additive coupling, with explicit cross-region parameters.

The energy is E_DPA(M) + E_QM_emb(Q; qM) + E_LJ(Q, M). The embedded provider
includes the QM internal energy and fixed-charge interaction. No environment
Coulomb term or full-system DPA evaluation is added. This energy definition
does not establish that a caller's charges or cross parameters are physical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .core import Evaluation, EvaluationError, Geometry, _atom_ids, _evaluate
from .embedding import EmbeddedEvaluation


@dataclass(frozen=True, eq=False)
class CrossLennardJones:
    """Untruncated 12-6 LJ interactions between Q and M, with no mixing rules.

    Each directional (QM symbol, environment symbol) key maps to
    (epsilon in eV, sigma in Angstrom). Every encountered pair needs an explicit
    value; epsilon=0 disables that pair. Parameters and source are immutable.
    No within-region interactions or periodic images are evaluated.
    """

    parameters: Mapping[tuple[str, str], tuple[float, float]]
    source: str = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("An explicit cross-parameter source is required")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Cross parameters must map symbol pairs to epsilon and sigma")
        parameters = {}
        for key, values in self.parameters.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(not isinstance(symbol, str) or not symbol.strip() for symbol in key)
            ):
                raise ValueError("Each cross-parameter key must be (QM symbol, environment symbol)")
            try:
                epsilon, sigma = values
            except (TypeError, ValueError) as exc:
                raise ValueError("Each cross pair requires epsilon and sigma") from exc
            converted = []
            for value in (epsilon, sigma):
                scalar = np.asarray(value)
                if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
                    raise ValueError("Cross parameters must be finite real scalars")
                converted.append(float(scalar))
            epsilon, sigma = converted
            if not np.isfinite(converted).all() or epsilon < 0 or sigma <= 0:
                raise ValueError("Cross parameters require finite epsilon >= 0 and sigma > 0")
            parameters[key] = (epsilon, sigma)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def evaluate(self, qm_geometry: Geometry, environment_geometry: Geometry) -> EmbeddedEvaluation:
        """Return cross energy and equal/opposite forces in the supplied orders."""
        if qm_geometry.periodic or environment_geometry.periodic:
            raise ValueError("Cross interactions require nonperiodic geometries")
        if set(qm_geometry.atom_ids).intersection(environment_geometry.atom_ids):
            raise ValueError("QM and environment atom IDs must be disjoint")
        required = {
            (qm_symbol, environment_symbol)
            for qm_symbol in qm_geometry.symbols
            for environment_symbol in environment_geometry.symbols
        }
        missing = required.difference(self.parameters)
        if missing:
            raise ValueError(f"Missing explicit cross parameters for {sorted(missing)}")
        qm_forces = np.zeros_like(qm_geometry.positions)
        environment_forces = np.zeros_like(environment_geometry.positions)
        energy = 0.0
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                for i, qm_symbol in enumerate(qm_geometry.symbols):
                    for j, environment_symbol in enumerate(environment_geometry.symbols):
                        delta = qm_geometry.positions[i] - environment_geometry.positions[j]
                        distance2 = np.dot(delta, delta)
                        if distance2 == 0:
                            raise ValueError("QM and environment sites must not coincide")
                        epsilon, sigma = self.parameters[(qm_symbol, environment_symbol)]
                        if epsilon == 0:
                            continue
                        sixth = (sigma / np.sqrt(distance2)) ** 6
                        twelfth = sixth * sixth
                        energy += 4 * epsilon * (twelfth - sixth)
                        force = (24 * epsilon / distance2) * (2 * twelfth - sixth) * delta
                        qm_forces[i] += force
                        environment_forces[j] -= force
        except (FloatingPointError, OverflowError) as exc:
            raise EvaluationError("Cross interaction is outside finite numerical range") from exc
        return EmbeddedEvaluation(
            energy,
            qm_geometry.atom_ids,
            environment_geometry.atom_ids,
            qm_forces,
            environment_forces,
            {
                "model": "cross-lennard-jones-12-6",
                "parameter_source": self.source,
                "pair_count": len(qm_geometry.atom_ids) * len(environment_geometry.atom_ids),
            },
        )


@dataclass(frozen=True, eq=False)
class AdditiveTerms:
    """One evaluation's three terms and total, with explicit atom ordering.

    The environment term follows environment_atom_ids. The two embedded-style
    terms follow qm_atom_ids and environment_atom_ids. The total follows
    atom_ids, the full input order. Each constituent force array is immutable.
    """

    atom_ids: tuple[int, ...]
    qm_atom_ids: tuple[int, ...]
    environment_atom_ids: tuple[int, ...]
    environment: Evaluation
    embedded: EmbeddedEvaluation
    cross: EmbeddedEvaluation
    total: Evaluation


def _embedded_evaluate(backend, qm_geometry, environment_geometry, name):
    result = backend.evaluate(qm_geometry, environment_geometry)
    if not isinstance(result, EmbeddedEvaluation):
        raise EvaluationError(f"{name} backend must return an EmbeddedEvaluation")
    if (
        result.qm_atom_ids != qm_geometry.atom_ids
        or result.environment_atom_ids != environment_geometry.atom_ids
    ):
        raise EvaluationError(f"{name} backend atom IDs must match the inputs in order")
    if (
        result.qm_forces_ev_per_angstrom.shape != qm_geometry.positions.shape
        or result.environment_forces_ev_per_angstrom.shape != environment_geometry.positions.shape
    ):
        raise EvaluationError(f"{name} backend force counts must match the input atoms")
    return result


class AdditiveCalculator:
    """Compose E_DPA(M) + E_QM_emb(Q; qM) + E_LJ(Q, M) on fixed nonempty regions.

    Atom membership is fixed at construction. Element identity is fixed on the
    first evaluation; full input atom order may change. Providers receive the
    configured region orders on every call, including after input reordering.
    The caller owns provider lifetimes and any external workers.
    """

    def __init__(
        self,
        environment_backend,
        embedded_backend,
        cross_backend,
        *,
        qm_atom_ids,
        environment_atom_ids,
    ):
        qm_ids = _atom_ids(qm_atom_ids)
        environment_ids = _atom_ids(environment_atom_ids)
        if set(qm_ids).intersection(environment_ids):
            raise ValueError("QM and environment atom IDs must be disjoint")
        self.environment_backend = environment_backend
        self.embedded_backend = embedded_backend
        self.cross_backend = cross_backend
        self._qm_atom_ids = qm_ids
        self._environment_atom_ids = environment_ids
        self._identity = None

    @property
    def qm_atom_ids(self):
        return self._qm_atom_ids

    @property
    def environment_atom_ids(self):
        return self._environment_atom_ids

    def evaluate_terms(self, full_geometry: Geometry) -> AdditiveTerms:
        """Evaluate each term once and return its forces and the assembled total."""
        if full_geometry.periodic:
            raise ValueError("Additive coupling requires a nonperiodic full geometry")
        if set(full_geometry.atom_ids) != set(self.qm_atom_ids).union(self.environment_atom_ids):
            raise ValueError("Fixed QM and environment regions must cover all input atoms exactly")
        qm = full_geometry.select(self.qm_atom_ids)
        environment = full_geometry.select(self.environment_atom_ids)
        identity = (qm.symbols, environment.symbols)
        if self._identity is not None and self._identity != identity:
            raise ValueError("Element symbols for fixed atom IDs must stay unchanged")
        self._identity = identity
        environment_result = _evaluate(self.environment_backend, environment)
        embedded = _embedded_evaluate(self.embedded_backend, qm, environment, "Embedded")
        cross = _embedded_evaluate(self.cross_backend, qm, environment, "Cross")
        lookup = {atom_id: index for index, atom_id in enumerate(full_geometry.atom_ids)}
        qm_indices = [lookup[atom_id] for atom_id in self.qm_atom_ids]
        environment_indices = [lookup[atom_id] for atom_id in self.environment_atom_ids]
        forces = np.empty_like(full_geometry.positions)
        forces[qm_indices] = embedded.qm_forces_ev_per_angstrom + cross.qm_forces_ev_per_angstrom
        forces[environment_indices] = (
            environment_result.forces
            + embedded.environment_forces_ev_per_angstrom
            + cross.environment_forces_ev_per_angstrom
        )
        total = Evaluation(environment_result.energy + embedded.energy_ev + cross.energy_ev, forces)
        return AdditiveTerms(
            full_geometry.atom_ids,
            self.qm_atom_ids,
            self.environment_atom_ids,
            environment_result,
            embedded,
            cross,
            total,
        )

    def evaluate(self, full_geometry: Geometry) -> Evaluation:
        """Return the total through the same path used by evaluate_terms."""
        return self.evaluate_terms(full_geometry).total
