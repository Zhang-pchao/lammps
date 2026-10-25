# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Complete fixed-region hybrid energies and forces for named selected states.

The existing single-state calculators own every Hamiltonian. Only orchestration
and exact, call-local sharing of state-independent evaluations are added here.
No electronic propagation, interstate coupling or moving partition is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .additive import AdditiveCalculator
from .core import Evaluation, EvaluationError, Geometry, SubtractiveCalculator
from .embedding import EmbeddedEvaluation
from .excited import TDASurfaceEvaluation
from .gap_anchor import (
    GapAnchorCalculator,
    make_gpu_rhf_gap_anchor,
    make_pbe0_gap_anchor,
    make_rhf_gap_anchor,
)
from .multistate import GPU4PySCFTDAStateSet, PySCFPBE0TDAStateSet, PySCFTDAStateSet


@dataclass(frozen=True, eq=False)
class HybridStateEvaluation:
    """A complete hybrid total in input order, plus its original quantum surface."""

    hamiltonian: str
    atom_ids: tuple[int, ...]
    qm_atom_ids: tuple[int, ...]
    environment_atom_ids: tuple[int, ...]
    total: Evaluation
    quantum: TDASurfaceEvaluation


def _geometry_key(geometry):
    return (geometry.atom_ids, geometry.symbols, geometry.positions.tobytes(),
            None if geometry.cell is None else geometry.cell.tobytes())


class _CachedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.charges = getattr(backend, "charges", None)
        self.results = {}

    def evaluate(self, *geometries):
        key = tuple(_geometry_key(geometry) for geometry in geometries)
        if key not in self.results:
            self.results[key] = self.backend.evaluate(*geometries)
        return self.results[key]


class _CallCache:
    """Share only the same object and exact ordered input within one request."""

    def __init__(self):
        self.backends = {}

    def wrap(self, backend):
        key = id(backend)
        if key not in self.backends:
            self.backends[key] = _CachedBackend(backend)
        return self.backends[key]


class _SurfaceBackend:
    """A checked, already evaluated surface for existing single-state assembly."""

    def __init__(self, surface, qm, environment, charges):
        self.charges = charges
        geometries = (qm,) if environment is None else (qm, environment)
        self.key = tuple(_geometry_key(geometry) for geometry in geometries)
        if environment is None:
            self.result = Evaluation(surface.energy_ev, surface.qm_forces_ev_per_angstrom)
        else:
            self.result = EmbeddedEvaluation(
                surface.energy_ev, qm.atom_ids, environment.atom_ids,
                surface.qm_forces_ev_per_angstrom,
                surface.environment_forces_ev_per_angstrom, surface.diagnostics,
            )

    def evaluate(self, *geometries):
        if tuple(_geometry_key(geometry) for geometry in geometries) != self.key:
            raise EvaluationError("Shared quantum result requested for different coordinates or IDs")
        return self.result


class _StateSetCoupling:
    def __init__(self, state_set, *, embedded):
        if type(state_set) not in (
            PySCFTDAStateSet, PySCFPBE0TDAStateSet, GPU4PySCFTDAStateSet,
        ):
            raise TypeError("Hybrid state sets require a supported pinned TDA state set")
        first = next(iter(state_set.references.values()))
        if (first.environment_geometry is not None) != embedded:
            raise ValueError("Quantum references must match the selected embedding model")
        self._state_set = state_set
        self._references = dict(state_set.references)
        self._qm_template = first.qm_geometry
        self._environment_template = first.environment_geometry
        self._charges = first.charges
        self._identity = None

    @property
    def state_set(self):
        return self._state_set

    def _geometries(self, full):
        if not isinstance(full, Geometry) or full.periodic:
            raise ValueError("Hybrid state sets require a nonperiodic full geometry")
        identity = tuple(sorted(zip(full.atom_ids, full.symbols)))
        if self._identity is not None and identity != self._identity:
            raise ValueError("Full-system atom IDs and element symbols must stay unchanged")
        qm = full.select(self._qm_template.atom_ids)
        if qm.symbols != self._qm_template.symbols:
            raise ValueError("QM element symbols differ from the state references")
        environment = None
        if self._environment_template is not None:
            ids = self._qm_template.atom_ids + self._environment_template.atom_ids
            if set(full.atom_ids) != set(ids):
                raise ValueError("Fixed QM and environment regions must cover all atoms exactly")
            environment = full.select(self._environment_template.atom_ids)
            if environment.symbols != self._environment_template.symbols:
                raise ValueError("Environment element symbols differ from the state references")
        self._identity = identity
        return qm, environment

    def evaluate(self, full_geometry):
        """Return an immutable name-to-HybridStateEvaluation mapping, or raise.

        Every state requires complete forces. Nothing from a previous call is
        reused, including after failure. Input reordering preserves stable IDs.
        """
        qm, environment = self._geometries(full_geometry)
        if dict(self.state_set.references) != self._references:
            raise ValueError("The configured state references changed")
        surfaces = self.state_set.evaluate_surfaces(qm, environment, forces=True)
        if set(surfaces) != set(self._references):
            raise EvaluationError("Shared result names differ from the configured states")
        providers = {}
        for name, reference in self._references.items():
            surface = surfaces[name]
            if not isinstance(surface, TDASurfaceEvaluation):
                raise EvaluationError("Shared quantum provider must return TDA surfaces")
            if (surface.qm_atom_ids != qm.atom_ids
                    or surface.environment_atom_ids != (() if environment is None
                                                         else environment.atom_ids)):
                raise EvaluationError("Shared quantum atom IDs differ from the configured regions")
            if surface.diagnostics.get("reference_sha256") != reference.sha256:
                raise EvaluationError("Shared quantum state reference differs from its name")
            if (surface.qm_forces_ev_per_angstrom is None
                    or (environment is not None
                        and surface.environment_forces_ev_per_angstrom is None)):
                raise EvaluationError("Complete hybrid evaluation requires all quantum forces")
            providers[name] = _SurfaceBackend(surface, qm, environment, self._charges)
        cache = _CallCache()
        results = {}
        env_ids = tuple(atom for atom in full_geometry.atom_ids if atom not in qm.atom_ids)
        for name, provider in providers.items():
            total = self._assemble(full_geometry, qm, provider, cache)
            results[name] = HybridStateEvaluation(
                self.hamiltonian, full_geometry.atom_ids, qm.atom_ids, env_ids,
                total, surfaces[name],
            )
        return MappingProxyType(results)


class MechanicalStateSet(_StateSetCoupling):
    """Named ME totals using an isolated shared quantum state set."""

    hamiltonian = "fixed-region-mechanical"

    def __init__(self, low_backend, state_set, *, full_low_backend=None):
        super().__init__(state_set, embedded=False)
        self.low_backend = low_backend
        self.full_low_backend = low_backend if full_low_backend is None else full_low_backend

    def _assemble(self, full, qm, provider, cache):
        return SubtractiveCalculator(
            cache.wrap(self.low_backend), provider,
            full_low=cache.wrap(self.full_low_backend),
        ).evaluate(full, qm)


class AdditiveStateSet(_StateSetCoupling):
    """Named additive EE totals, with shared environment and cross terms."""

    hamiltonian = "finite-electrostatic-additive"

    def __init__(self, environment_backend, state_set, cross_backend):
        super().__init__(state_set, embedded=True)
        self.environment_backend = environment_backend
        self.cross_backend = cross_backend

    def _assemble(self, full, qm, provider, cache):
        return AdditiveCalculator(
            cache.wrap(self.environment_backend), provider, cache.wrap(self.cross_backend),
            qm_atom_ids=qm.atom_ids,
            environment_atom_ids=self._environment_template.atom_ids,
        ).evaluate(full)


class GapAnchorStateSet(_StateSetCoupling):
    """Named plain gap-anchor totals with matched, shared ground terms.

    Ground settings are constructed by the existing method-specific factories.
    This does not admit corrected-fragment models or selected GPU DFT.
    """

    hamiltonian = "finite-electrostatic-gap-anchor"

    def __init__(self, full_low_backend, region_low_backend, state_set):
        super().__init__(state_set, embedded=True)
        factory = {
            PySCFTDAStateSet: make_rhf_gap_anchor,
            PySCFPBE0TDAStateSet: make_pbe0_gap_anchor,
            GPU4PySCFTDAStateSet: make_gpu_rhf_gap_anchor,
        }[type(state_set)]
        self._anchor = factory(full_low_backend, region_low_backend,
                               next(iter(self._references.values())))

    def _assemble(self, full, qm, provider, cache):
        anchor = self._anchor
        return GapAnchorCalculator(
            cache.wrap(anchor.full_low_backend), cache.wrap(anchor.region_low_backend),
            cache.wrap(anchor.isolated_ground_backend),
            cache.wrap(anchor.embedded_ground_backend), provider,
            qm_atom_ids=qm.atom_ids,
            environment_atom_ids=self._environment_template.atom_ids,
        ).evaluate(full)
