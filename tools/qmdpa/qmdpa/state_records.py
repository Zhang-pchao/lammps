# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Same-evaluation electronic-state records for accepted native frames.

Records never choose a state or cache a calculation. The provider selects
against its immutable reference; optional exact-input result reuse keeps the
same validated state and renews the record on every call. No quantum engine is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise

from .core import Evaluation, EvaluationError, Geometry
from .embedding import EmbeddedEvaluation
from .excited import select_tda_state
from .providers import load_tda_reference, reference_engine, reference_method


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Electronic-state {name} must be a finite number")
    return value


def _sha(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Electronic-state {name} must be a SHA256")


def _geometry_record(geometry):
    if geometry is None:
        return None
    if not isinstance(geometry, Geometry) or geometry.periodic:
        raise ValueError("Electronic-state records require finite geometries")
    return {
        "atom_ids": list(geometry.atom_ids),
        "symbols": list(geometry.symbols),
        # Mechanical image unwrapping can change -0.0 into +0.0. These
        # coordinates are identical; preserve every nonzero float exactly.
        "positions_angstrom": [
            [0.0 if value == 0.0 else value for value in row] for row in geometry.positions.tolist()
        ],
    }


def geometry_digest(qm, environment=None):
    """Hash both actual provider inputs, with their atom identity and order."""
    payload = {"qm": _geometry_record(qm), "environment": _geometry_record(environment)}
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def configured_reference(config):
    """Load the already-declared inline state without importing PySCF."""
    state = config.get("provider", {}).get("state", {"method": "ground"})
    if state["method"] == "ground":
        return None
    if (state["method"] == "sa-casscf22"
            and config.get("hamiltonian") == "finite-two-state-ehrenfest-mechanical"):
        # Coherent amplitudes have their own complete history validation.
        return None
    if state["method"] not in ("rhf-tda", "pbe0-tda") or config.get("periodic") is not False:
        raise ValueError("Native state records require finite selected TDA")
    reference = load_tda_reference(state["method"], state["reference"],
                                   engine=config["provider"]["engine"])
    if reference.settings.tda_solver != "dense":
        raise ValueError("Native state records require explicit dense TDA")
    return reference



def configured_auxiliary_providers(config):
    """Return the explicit selected providers required by this Hamiltonian."""
    from .fragment_response_config import FRAGMENT_GAP_HAMILTONIAN

    enabled = config.get("hamiltonian") == FRAGMENT_GAP_HAMILTONIAN
    if enabled != ("fragment_response" in config):
        raise ValueError("Fragment-response Hamiltonian and configuration must be selected together")
    if not enabled:
        return {}
    gap = config["fragment_response"]["cluster_gap"]
    return {"cluster_pair": gap["pair_provider"], "cluster_field": gap["field_provider"]}

def frame_geometry_digest(frame, reference):
    """Recover the evaluated subset from a complete, sorted accepted frame."""
    atoms = {atom["id"]: atom for atom in frame["atoms"]}
    if len(atoms) != len(frame["atoms"]):
        raise ValueError("Electronic-state frame contains duplicate atom IDs")

    def geometry(template):
        if template is None:
            return None
        selected = [atoms[atom_id] for atom_id in template.atom_ids]
        if any(atom["image"] != [0, 0, 0] for atom in selected):
            raise ValueError("Finite electronic-state atoms require zero image flags")
        return Geometry(template.atom_ids, template.symbols, [atom["x"] for atom in selected])

    return geometry_digest(
        geometry(reference.qm_geometry), geometry(reference.environment_geometry)
    )


def validate_state_record(record, reference, *, matrix_assembly="operator"):
    """Check declared state/solver gates, independently of a success flag."""
    if not isinstance(record, dict) or set(record) != {
        "schema_version",
        "method",
        "reference_sha256",
        "geometry_sha256",
        "selected_root",
        "provider_energy_ev",
        "diagnostics",
    }:
        raise ValueError("Incomplete electronic-state record")
    method = reference_method(reference)
    gpu = reference_engine(reference) == "gpu4pyscf"
    _json(record)  # Reject nonfinite data, including optional diagnostic fields.
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["method"] != method
    ):
        raise ValueError("Unsupported electronic-state record")
    for name in ("reference_sha256", "geometry_sha256"):
        _sha(record[name], name)
    if record["reference_sha256"] != reference.sha256:
        raise ValueError("Electronic-state reference differs from the configuration")
    settings = reference.settings
    diagnostic = record["diagnostics"]
    if not isinstance(diagnostic, dict):
        raise TypeError("Electronic-state diagnostics must be a mapping")
    index, selected = select_tda_state(
        diagnostic["candidate_overlaps"], diagnostic["candidate_excitation_energies_ev"], settings
    )
    if (
        type(record["selected_root"]) is not int
        or type(diagnostic.get("selected_root")) is not int
        or record["selected_root"] != index + 1
        or any(
            diagnostic.get(name) != value
            for name, value in selected.items()
            if name not in ("candidate_overlaps", "candidate_excitation_energies_ev")
        )
        or diagnostic.get("reference_sha256") != reference.sha256
        or diagnostic.get("tracking") != "fixed-reference-ao-transition-density"
        or diagnostic.get("scf_converged") is not True
        or diagnostic.get("candidate_roots_converged") is not True
        or diagnostic.get("forces_evaluated") is not True
        or diagnostic.get("method") != (
            "gpu-rhf-tda-singlet" if gpu else {
                "rhf-tda": "cpu-rhf-tda-singlet", "pbe0-tda": "cpu-pbe0-tda-singlet"
            }[method])
        or diagnostic.get("energy_definition") != {"rhf-tda": "E_RHF+omega",
                                                       "pbe0-tda": "E_PBE0+omega"}[method]
    ):
        raise ValueError("Electronic-state selection or force evaluation is inconsistent")
    energy = _number(record["provider_energy_ev"], "provider energy")
    ground = _number(diagnostic["ground_energy_ev"], "ground energy")
    excitation = _number(diagnostic["excitation_energy_ev"], "excitation energy")
    if (
        excitation != diagnostic["candidate_excitation_energies_ev"][index]
        or abs(energy - (ground + excitation)) > 1e-7
    ):
        raise ValueError("Electronic-state total energy must equal the declared ground energy + omega")
    if matrix_assembly == "operator":
        if (diagnostic.get("tda_matrix_assembly", "operator") != "operator"
                or diagnostic.get("tda_dense_batch_size") != 16):
            raise ValueError("Electronic-state matrix assembly differs from the configuration")
    elif matrix_assembly == "official-get-ab-896" and method == "pbe0-tda":
        points = diagnostic.get("tda_matrix_grid_points")
        blocks = diagnostic.get("tda_matrix_grid_blocks")
        if (diagnostic.get("tda_matrix_assembly") != matrix_assembly
                or "tda_dense_batch_size" in diagnostic
                or type(points) is not int or type(blocks) is not int
                or not 0 < blocks <= points <= 896 * blocks
                or points != diagnostic.get("grid_points")):
            raise ValueError("Electronic-state official matrix coverage or identity failed")
    else:
        raise ValueError("Unsupported electronic-state matrix assembly")
    dimension = diagnostic["tda_dimension"]
    if (
        diagnostic["tda_solver"] != "dense"
        or type(dimension) is not int
        or not settings.nstates <= dimension <= settings.tda_dense_max_dimension
        or diagnostic["tda_dense_max_dimension"] != settings.tda_dense_max_dimension
        or diagnostic["tda_hermiticity_tolerance_hartree"] != 1e-12
        or diagnostic["tda_positive_eigenvalue_threshold_hartree"] != 0.001
        or not 0 <= _number(diagnostic["tda_hermiticity_max_error_hartree"], "Hermiticity") <= 1e-12
    ):
        raise ValueError("Electronic-state dense solver identity or Hermiticity failed")
    for name in ("tda_matrix_residuals_hartree", "tda_eigensystem_residuals_hartree"):
        residuals = diagnostic[name]
        if len(residuals) != settings.nstates or any(
            not 0 <= _number(value, name) < settings.tda_conv_tol for value in residuals
        ):
            raise ValueError("Electronic-state normalized eigenpair residual failed")
    if (
        diagnostic["tda_eigensystem_residual_max_hartree"]
        != max(diagnostic["tda_eigensystem_residuals_hartree"])
        or diagnostic["tda_eigensystem_residual_tolerance_hartree"] != settings.tda_conv_tol
    ):
        raise ValueError("Electronic-state eigenpair residual summary differs")
    if method == "pbe0-tda":
        _validate_pbe0_response(diagnostic, reference)
        return
    history = diagnostic["cphf_residual_history"]
    iterations = diagnostic["cphf_refinement_iterations"]
    if (
        type(iterations) is not int
        or not 0 <= iterations <= 2
        or len(history) != iterations + 1
        or any(_number(value, "CPHF history") < 0 for value in history)
        or any(right >= left for left, right in pairwise(history))
        or history[0] != diagnostic["cphf_residual_before_refinement"]
        or history[-1] != diagnostic["cphf_residual_after_refinement"]
        or history[-1] > settings.response_residual_tol
        or not 0
        <= _number(diagnostic["cphf_residual_max"], "CPHF residual")
        <= settings.response_residual_tol
        or diagnostic["cphf_residual_tolerance"] != settings.response_residual_tol
        or diagnostic["cphf_refinement_method"] != "normalized-defect-correction"
        or (not gpu and _number(diagnostic["density_trace"], "density trace") <= 0)
    ):
        raise ValueError("Electronic-state CPHF response diagnostics failed")
    if gpu and (
        diagnostic.get("engine") != "gpu4pyscf"
        or diagnostic.get("device") != "cuda"
        or diagnostic.get("cpu_fallback") is not False
        or diagnostic.get("cphf_solver") != "official-gpu-normalized-defect-correction"
        or type(diagnostic.get("cphf_refinement_count")) is not int
        or diagnostic["cphf_refinement_count"] != iterations
        or diagnostic["cphf_residual_max"] != history[-1]
    ):
        raise ValueError("Electronic-state GPU response identity is inconsistent")
    if gpu and reference.charges is not None:
        if (
            diagnostic.get("relaxed_density") != "official-total-rhf-tda"
            or diagnostic.get("density_export") != "base-zero-response-bookkeeping"
            or diagnostic.get("density_trace_tolerance") != settings.density_trace_tol
            or _number(diagnostic.get("density_trace"), "density trace") <= 0
            or not 0 <= _number(diagnostic.get("density_trace_error"), "density trace error")
            <= settings.density_trace_tol
        ):
            raise ValueError("Electronic-state GPU environment density is incomplete or inconsistent")


def _validate_pbe0_response(diagnostic, reference):
    """Reject missing or foreign orbital/quadrature response on a PBE0 frame."""
    settings = reference.settings
    expected = {
        "xc": settings.xc, "grid_level": settings.grid_level,
        "grid_prune": settings.grid_prune,
        "rks_grid_response": True, "density_fitting": False, "nlc": False,
        "cphf_solver": "official-unrefined-residual-checked", "cphf_refinement_count": 0,
        "grid_response_model": "restricted-pbe0-tda-xc-lagrangian-v1",
        "grid_response_force_assembly": True, "weight_response": True,
        "coordinate_response": True,
    }
    if any(type(diagnostic.get(key)) is not type(value) or diagnostic[key] != value
           for key, value in expected.items()):
        raise ValueError("Electronic-state PBE0 orbital or grid response identity failed")
    if any(key in diagnostic for key in ("cphf_refinement_method", "cphf_residual_history")):
        raise ValueError("PBE0 state cannot use an RHF response protocol")
    if (not 0 <= _number(diagnostic["cphf_residual_max"], "CPHF residual")
            <= settings.response_residual_tol
            or _number(diagnostic["density_trace"], "density trace") <= 0):
        raise ValueError("Electronic-state PBE0 response residual or density failed")
    for name in ("grid_points", "quadrature_response_points"):
        value = diagnostic.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError("Electronic-state PBE0 quadrature is incomplete")
    correction = diagnostic.get("grid_response_gradient_hartree_bohr")
    if (not isinstance(correction, list)
            or len(correction) != len(reference.qm_geometry.atom_ids)
            or any(not isinstance(row, list) or len(row) != 3 for row in correction)):
        raise ValueError("Electronic-state PBE0 grid response requires every QM atom")
    for row in correction:
        for value in row:
            _number(value, "grid response gradient")


def validate_frame_state(frame, config):
    """Reject missing, foreign or stale records for every selected-state term."""
    reference = configured_reference(config)
    record = frame.get("electronic_state")
    providers = configured_auxiliary_providers(config)
    if reference is None:
        if record is not None:
            raise ValueError("Ground-state frame must not contain an excited-state record")
        if providers:
            raise ValueError("Auxiliary states require a declared primary selected state")
    else:
        validate_state_record(record, reference,
            matrix_assembly=config.get("provider", {}).get("matrix_assembly", "operator"))
        if record["geometry_sha256"] != frame_geometry_digest(frame, reference):
            raise ValueError("Electronic-state geometry differs from the accepted frame")
    if not providers:
        if "auxiliary_electronic_states" in frame:
            raise ValueError("This Hamiltonian does not declare auxiliary electronic states")
        return
    records = frame.get("auxiliary_electronic_states")
    if not isinstance(records, dict) or set(records) != set(providers):
        raise ValueError("Missing or unexpected auxiliary electronic states")
    for role, provider in providers.items():
        state = provider["state"]
        auxiliary = load_tda_reference(state["method"], state["reference"])
        validate_state_record(records[role], auxiliary,
                              matrix_assembly=provider.get("matrix_assembly", "operator"))
        if records[role]["geometry_sha256"] != frame_geometry_digest(frame, auxiliary):
            raise ValueError("Auxiliary electronic-state geometry differs from the accepted frame")


class RecordingTDAProvider:
    """Keep a detached record from the same force evaluation, on its owner rank."""

    def __init__(self, backend):
        self.backend = backend
        self.reference = backend.reference
        self.method = reference_method(self.reference)
        if self.reference.settings.tda_solver != "dense":
            raise ValueError("Native state recording requires explicit dense TDA")
        self.clear()

    def clear(self):
        """Invalidate the prior record before any new or failed callback."""
        self._record_json = None

    def evaluate(self, qm_geometry, environment_geometry=None):
        self.clear()
        try:
            return self._evaluate(qm_geometry, environment_geometry)
        except BaseException:
            # A failed record/force check must not retain a reusable result.
            clear_reuse = getattr(self.backend, "clear_reuse", None)
            if clear_reuse is not None:
                clear_reuse()
            raise

    def _evaluate(self, qm_geometry, environment_geometry):
        arguments = (
            (qm_geometry,) if environment_geometry is None else (qm_geometry, environment_geometry)
        )
        surface = self.backend.evaluate_surface(*arguments, forces=True)
        if surface.qm_atom_ids != qm_geometry.atom_ids or surface.environment_atom_ids != (
            () if environment_geometry is None else environment_geometry.atom_ids
        ):
            raise EvaluationError("Electronic-state force identity differs from the input")
        record = {
            "schema_version": 1,
            "method": self.method,
            "reference_sha256": self.reference.sha256,
            "geometry_sha256": geometry_digest(qm_geometry, environment_geometry),
            "selected_root": surface.diagnostics["selected_root"],
            "provider_energy_ev": surface.energy_ev,
            "diagnostics": dict(surface.diagnostics),
        }
        # A serialized snapshot prevents later backend/caller mutation.
        record_json = _json(record)
        validate_state_record(json.loads(record_json), self.reference,
                              matrix_assembly=getattr(self.backend, "matrix_assembly", "operator"))
        if environment_geometry is None:
            result = Evaluation(surface.energy_ev, surface.qm_forces_ev_per_angstrom)
        else:
            result = EmbeddedEvaluation(
                surface.energy_ev,
                surface.qm_atom_ids,
                surface.environment_atom_ids,
                surface.qm_forces_ev_per_angstrom,
                surface.environment_forces_ev_per_angstrom,
                json.loads(record_json)["diagnostics"],
            )
        self._record_json = record_json
        return result

    def accepted_record(self, frame):
        """Return a copy only when the sampled coordinates match the evaluation."""
        if self._record_json is None:
            raise EvaluationError("No successful electronic-state evaluation is available")
        record = json.loads(self._record_json)
        validate_state_record(record, self.reference,
                              matrix_assembly=getattr(self.backend, "matrix_assembly", "operator"))
        if record["geometry_sha256"] != frame_geometry_digest(frame, self.reference):
            raise EvaluationError("Electronic-state evaluation does not match the sampled geometry")
        return record


class StateRecordGroup:
    """Invalidate and report all selected terms of one native callback together.

    Providers keep their own immutable references and force evaluations. This
    object never chooses a state or computes an extra energy. Legacy frames keep
    their primary electronic_state; auxiliary records have explicit term names.
    """

    def __init__(self, primary, auxiliary):
        if (not isinstance(primary, RecordingTDAProvider)
                or not isinstance(auxiliary, dict) or not auxiliary
                or any(not isinstance(k, str) or not k
                       or not isinstance(v, RecordingTDAProvider)
                       for k, v in auxiliary.items())):
            raise TypeError("State groups require a primary and named recording providers")
        providers = (primary, *auxiliary.values())
        if len({id(v) for v in providers}) != len(providers):
            raise ValueError("Each selected term requires its own recording provider")
        self.primary = primary
        self.auxiliary = dict(auxiliary)
        self.clear()

    def clear(self):
        for provider in (self.primary, *self.auxiliary.values()):
            provider.clear()

    def accepted_record(self, frame):
        try:
            return self.primary.accepted_record(frame)
        except BaseException:
            self.clear()
            raise

    def accepted_auxiliary_records(self, frame):
        try:
            return {role: provider.accepted_record(frame)
                    for role, provider in self.auxiliary.items()}
        except BaseException:
            self.clear()
            raise
