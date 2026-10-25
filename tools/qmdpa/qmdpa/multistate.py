# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Several fixed-reference TDA surfaces from one common solution.

Only the SCF/TDA eigensystem is shared. Every requested state's orbital response
and complete QM/environment forces use the existing single-state implementation.
No electronic propagation, inter-state coupling or native multi-state MD follows.
"""

from collections.abc import Mapping
from types import MappingProxyType

from .core import EvaluationError
from .excited import (
    PySCFTDABackend,
    PySCFTDAElectrostaticBackend,
    TDAReference,
    TDASettings,
    _charges_record,
    _load_runtime,
    _reference_from_solved,
    _solve,
    _validate_geometry,
)


def _dense_settings(settings, settings_type=TDASettings):
    if type(settings) is not settings_type or settings.tda_solver != "dense":
        method = {"TDASettings": "CPU RHF", "PBE0TDASettings": "CPU PBE0",
                  "GPUTDASettings": "GPU RHF"}.get(settings_type.__name__, settings_type.__name__)
        raise ValueError(f"State sets require explicit {method} dense TDA settings")


def _state_names(states):
    if (not isinstance(states, Mapping) or len(states) < 2
            or any(not isinstance(name, str) or not name.strip() for name in states)):
        raise ValueError("A state set requires at least two nonempty state names")
    return dict(states)


def _requested_roots(settings, states, *, settings_type=TDASettings):
    """Use the same root contract in Python and file preparation."""
    _dense_settings(settings, settings_type)
    states = _state_names(states)
    if (any(type(root) is not int or not 1 <= root <= settings.nstates
            for root in states.values()) or len(set(states.values())) != len(states)):
        raise ValueError("State roots must be distinct integers within the candidate set")
    return states


def build_tda_references(
    qm_geometry, settings, *, states, environment_geometry=None, charges=None,
):
    """Prepare name-to-root references together; root numbers are one-based.

    All requested roots must be distinct members of the unchanged candidate set.
    Output names label fixed references, not a root number to follow during motion.
    """
    return _build_references(
        qm_geometry, settings, states=states, environment_geometry=environment_geometry,
        charges=charges, reference_type=TDAReference, runtime_loader=_load_runtime,
        solver=_solve,
    )


def build_pbe0_tda_references(
    qm_geometry, settings, *, states, environment_geometry=None, charges=None,
):
    """Prepare bounded CPU PBE0 states with the existing operator assembly."""
    from .pbe0_tda import PBE0TDAReference, _load_runtime, _solve

    return _build_references(
        qm_geometry, settings, states=states, environment_geometry=environment_geometry,
        charges=charges, reference_type=PBE0TDAReference, runtime_loader=_load_runtime,
        solver=_solve,
    )


def build_gpu_tda_references(
    qm_geometry, settings, *, states, environment_geometry=None, charges=None,
):
    """Prepare pinned GPU RHF states using one existing dense GPU solution."""
    from .gpu_excited import GPUTDAReference, _load_runtime, _solve

    return _build_references(
        qm_geometry, settings, states=states, environment_geometry=environment_geometry,
        charges=charges, reference_type=GPUTDAReference,
        runtime_loader=lambda value: _load_runtime(value, electrostatic=charges is not None),
        solver=_solve,
    )


def _build_references(qm_geometry, settings, *, states, environment_geometry, charges,
                      reference_type, runtime_loader, solver):
    states = _requested_roots(settings, states, settings_type=reference_type._settings_type)
    _validate_geometry(qm_geometry, environment_geometry, charges)
    if charges is not None and charges.radii_angstrom is not None:
        raise ValueError("State sets currently support fixed point charges only")
    runtime = runtime_loader(settings)
    solved = solver(qm_geometry, settings, runtime, environment_geometry, charges)
    return MappingProxyType({
        name: _reference_from_solved(
            qm_geometry, settings, solved, runtime, reference_type=reference_type,
            initial_root=root, label=name,
            environment_geometry=environment_geometry, charges=charges,
        ) for name, root in states.items()
    })


def _region(geometry):
    return None if geometry is None else (geometry.atom_ids, geometry.symbols)


class PySCFTDAStateSet:
    """Evaluate several distinct, fixed-reference states without mutable history.

    References must describe identical settings, atom identities and fixed field.
    Their reference coordinates may differ. A collision, ambiguous match or failed
    response raises before returning any state. There is no cached previous result.
    """

    @staticmethod
    def _types():
        return TDASettings, TDAReference, PySCFTDABackend, PySCFTDAElectrostaticBackend

    def __init__(self, references):
        settings_type, reference_type, isolated_type, embedded_type = self._types()
        references = _state_names(references)
        if any(type(reference) is not reference_type for reference in references.values()):
            raise TypeError(f"State sets require pinned {reference_type.__name__} objects")
        first = next(iter(references.values()))
        _dense_settings(first.settings, settings_type)
        if first.charges is not None and first.charges.radii_angstrom is not None:
            raise ValueError("State sets currently support fixed point charges only")
        if len(references) > first.settings.nstates:
            raise ValueError("Requested states exceed the common candidate set")
        for reference in references.values():
            if (reference.settings != first.settings or reference.runtime != first.runtime
                    or _region(reference.qm_geometry) != _region(first.qm_geometry)
                    or _region(reference.environment_geometry) != _region(first.environment_geometry)
                    or _charges_record(reference.charges) != _charges_record(first.charges)):
                raise ValueError("State references must share settings, runtime, regions and fixed charges")
        if len({reference.sha256 for reference in references.values()}) != len(references):
            raise ValueError("State references must be distinct")
        self._references = MappingProxyType(references)
        self._charges = first.charges
        self._providers = tuple(
            (name, isolated_type(reference.settings, reference)
             if self._charges is None else
             embedded_type(reference.settings, reference, self._charges))
            for name, reference in references.items()
        )

    @property
    def references(self):
        return self._references

    def evaluate_surfaces(self, qm_geometry, environment_geometry=None, *, forces=True):
        """Return immutable name-to-TDASurfaceEvaluation results in eV and eV/A.

        forces=False skips every gradient and returns None for missing force arrays.
        Selection of all requested states precedes any state-specific gradient.
        """
        if type(forces) is not bool:
            raise ValueError("forces must be true or false")
        for _, provider in self._providers:
            provider._validate_identity(qm_geometry, environment_geometry, self._charges)
            # Also reject replacement of settings/reference/runtime on a provider.
            # This identity check creates no cache and changes no state selection.
            provider._reuse_key(qm_geometry, environment_geometry, self._charges, forces)
        first = self._providers[0][1]
        try:
            solved = first._solve_surface(
                qm_geometry, first.settings, first._runtime, environment_geometry, self._charges
            )
            selections = [(name, provider, *provider._select_surface(solved, forces))
                          for name, provider in self._providers]
            if len({index for _, _, index, _ in selections}) != len(selections):
                raise EvaluationError("Distinct fixed references selected the same candidate state")
            results = {}
            for name, provider, index, diagnostics in selections:
                results[name] = provider._evaluate_selected_surface(
                    solved, index, diagnostics, qm_geometry, environment_geometry,
                    self._charges, forces,
                )
            return MappingProxyType(results)
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"TDA state-set evaluation failed: {exc}") from exc


class PySCFPBE0TDAStateSet(PySCFTDAStateSet):
    """Bounded CPU PBE0 state set with each state's complete grid response.

    Only orchestration is shared with RHF. Method-specific references, source
    pins and the existing single-state PBE0 gradients remain mandatory. This
    route uses operator matrix assembly and does not admit fitted or GPU states.
    """

    @staticmethod
    def _types():
        from .providers import _tda_types

        return _tda_types("pbe0-tda")


class GPU4PySCFTDAStateSet(PySCFTDAStateSet):
    """Pinned conventional GPU RHF states with complete separate responses.

    Only orchestration is shared with the CPU state sets. The common SCF/TDA
    solve and each state's existing gradient stay on the declared GPU route.
    CPU references, selected DFT and fitted integrals are not accepted.
    """

    @staticmethod
    def _types():
        from .providers import _tda_types

        return _tda_types("rhf-tda", engine="gpu4pyscf")
