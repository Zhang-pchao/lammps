#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulation
# SPDX-License-Identifier: GPL-2.0-only
"""cDFT umbrella admission, restraint and accepted-frame controls."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from qmdpa.cdft_umbrella import FORM, GAP_NAME, bias_energy, bind, enabled, restraint
from qmdpa.cdft import HARTREE_EV
from qmdpa.cdft_umbrella_runtime import (capture, close, compare_restart, identity,
                                         validate_frame)
from qmdpa.core import Evaluation
from qmdpa.named_cv import digest


def options():
    return {"xc": "pbe0", "basis": "def2-svpd", "charge": 1, "multiplicity": 2,
            "grid_level": 2, "grid_prune": None, "grid_response": True, "conv_tol": 1e-12,
            "conv_tol_grad": 1e-7, "max_cycle": 100, "constraint_tol": 1e-10,
            "max_multiplier": 1., "max_constraint_iterations": 40,
            "spin_square_max": 1., "spin_population_tolerance": .2,
            "inner_solver": "newton", "inner_lindep": 1e-24}


def fixture(tmp_path):
    """A minimal admitted umbrella configuration with pinned stub libraries."""
    library = tmp_path/"libplumedKernel.so"
    library.write_bytes(b"stub kernel")
    from qmdpa.config import file_digest

    digest_value = file_digest(library)
    raw = {"form": FORM, "center_eV": 0.5, "kappa_per_eV": 0.25,
           "kernel": {"path": "libplumedKernel.so", "sha256": digest_value},
           "wrapper": {"path": "libplumedKernel.so", "sha256": digest_value}}
    return raw, tmp_path


def config_with(raw, tmp_path):
    """Build the smallest configuration the umbrella binder accepts."""
    from qmdpa.cdft import FORM as CDFT_FORM, RUNTIME_SOURCES

    config = {"hamiltonian": "finite-electrostatic-additive", "periodic": False,
              "provider": {"engine": "pyscf", "device": "cpu",
                           "state": {"method": "uks-hirshfeld-cdft",
                                     "reference": {"sampled_state": "R"}}},
              "sample_interval": 1, "timestep_fs": .02, "steps": 4,
              "model_sha256": "a"*64, "physics_sha256": "f"*64,
              "cross": {"source": "synthetic"},
              "qm_region": {"atom_ids": [1, 2], "symbols": ["O", "H"]},
              "environment_region": {"atom_ids": [3, 4], "symbols": ["O", "H"]}}
    return config


class FakeBias:
    """A deterministic stand-in for the official PLUMED EXTRACV bridge."""

    def __init__(self, schema, parameters, identity_record=None):
        self.schema = schema
        self.parameters = parameters
        self.identity = identity_record or {}
        self.step = -1
        self.count = 0
        self.last_frame_sha256 = None

    def evaluate(self, frame, step):
        value = float(frame.values[0])
        row = self.parameters[GAP_NAME]
        delta = value-row["center"]
        derivative = row["kappa"]*delta
        energy = 0.5*derivative*delta
        return {"bias_energy_eV": energy,
                "derivatives_eV_per_unit": {GAP_NAME: derivative},
                "forces_eV_per_angstrom": (-derivative*frame.jacobian[0]).tolist(),
                "frame": frame.record(), "frame_sha256": frame.sha256,
                "identity_sha256": "b"*64, "accepted_step": self.step,
                "accepted_count": self.count}

    def accept(self, frame, step):
        if self.step >= 0 and step != self.step+1:
            raise ValueError("Accepted umbrella steps must be consecutive")
        result = self.evaluate(frame, step)
        self.step = step
        self.count += 1
        self.last_frame_sha256 = frame.sha256
        result["accepted_step"] = step
        result["accepted_count"] = self.count
        return result

    def checkpoint(self):
        record = {"identity_sha256": digest(self.identity), "identity": self.identity,
                  "step": self.step, "frame_sha256": self.last_frame_sha256,
                  "accepted_count": self.count}
        return {**record, "sha256": digest(record)}


class FakeCalculator:
    """A calculator exposing the same shape as the cDFT pair calculator."""

    def __init__(self, geometry, gap):
        self.geometry = geometry
        self.gap = gap
        self.bias = None
        self.last = None

    def clear(self):
        self.last = None

    def _frame(self):
        from qmdpa.cdft_runtime import schema
        from qmdpa.electronic_gap import gap_frame
        from qmdpa.core import Geometry

        geometry = Geometry(self.geometry["atom_ids"], self.geometry["symbols"],
                            np.asarray(self.geometry["positions"]))
        forces = np.zeros_like(geometry.positions)
        r = Evaluation(self.gap, forces)
        p = Evaluation(0., forces)
        return gap_frame(schema(self.config), geometry, {"R": r, "P": p}, ("R", "P"))


def umbrella_config(tmp_path):
    """Return an admitted umbrella configuration and its raw declaration."""
    raw, directory = fixture(tmp_path)
    config = config_with(raw, tmp_path)
    # The binder needs the real cDFT schema, which needs the reference digest.
    from qmdpa.cdft_runtime import schema

    config["provider"]["state"]["reference"] = reference()
    config["qm_region"] = {"atom_ids": [1, 2], "symbols": ["O", "H"],
                           "molecules": [[1, 2]]}
    config["environment_region"] = {"atom_ids": [3, 4], "symbols": ["O", "H"],
                                    "molecules": [[3, 4]]}
    config["cross"] = {"source": "synthetic"}
    return config, raw, directory


def reference():
    """A complete constrained-pair reference over the synthetic four-atom region.

    The record must be self-consistent: the diagnostics validator checks the
    electron ledger, the signed population and spin identities and the solver
    iteration trail against the frozen reference.
    """
    from qmdpa.cdft import FORM, RUNTIME_SOURCES

    def record(atom_ids, symbols, positions):
        return {"atom_ids": list(atom_ids), "symbols": list(symbols),
                "positions_angstrom": positions}

    ref = {"schema_version": 1, "form": FORM, "settings": options(),
           "qm_geometry": record([1, 2], ["O", "H"],
                                 [[0., 0., 0.], [.96, 0., 0.]]),
           "environment_geometry": record([3, 4], ["O", "H"],
                                          [[3.2, 0., 0.], [4.1, 0., 0.]]),
           "charges": {"atom_ids": [3, 4], "charges_e": [-.8, .4], "source": "synthetic"},
           "atom_weights": [1, -1],
           "atomic_references": {s: {"basis": [[0, [1., 1.]]], "density": [[float(n)]]}
                                 for s, n in (("H", 1), ("O", 8))},
           "states": {r: {"target": t, "density": [[[1.]], [[0.]]], "multiplier": .1,
                           "spin_population": -t, "spin_square": .75}
                      for r, t in (("R", -1.), ("P", 1.))},
           "sampled_state": "R",
           "runtime": {"sources": {n: "a"*64 for n in RUNTIME_SOURCES},
                       "libraries": {"libcgto.so": "b"*64},
                       "versions": {n: "test" for n in ("pyscf", "numpy", "scipy")}},
           "electron_counts": [5, 4]}
    ref["sha256"] = digest(ref)
    return ref


def test_umbrella_admission_requires_the_constrained_provider(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    config["provider"]["state"]["method"] = "ground"
    with pytest.raises(ValueError, match="constrained-state provider"):
        bind(config, raw, directory)


def test_umbrella_admission_rejects_a_second_bias_owner(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    config["sampling"] = {"engine": "plumed"}
    with pytest.raises(ValueError, match="another bias owner"):
        bind(config, raw, directory)


def test_umbrella_admission_rejects_a_nonpositive_kappa(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    raw["kappa_per_eV"] = 0.
    with pytest.raises(ValueError, match="positive kappa"):
        bind(config, raw, directory)


def test_umbrella_admission_rejects_a_changed_library(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    raw["kernel"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="checksum differs"):
        bind(config, raw, directory)


def test_admitted_umbrella_publishes_one_named_restraint(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    bind(config, raw, directory)
    assert enabled(config)
    assert config["cdft_umbrella"]["center_eV"] == 0.5
    assert config["cdft_umbrella"]["kappa_per_eV"] == 0.25
    rows = restraint(config)
    assert set(rows) == {GAP_NAME}
    assert rows[GAP_NAME] == {"center": 0.5, "kappa": 0.25, "unit": "eV"}


def test_umbrella_energy_and_derivative_are_the_harmonic_restraint(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    bind(config, raw, directory)
    energy, derivative = bias_energy(0.8, config)
    assert derivative == pytest.approx(0.25*0.3)
    assert energy == pytest.approx(0.5*0.25*0.3*0.3)
    energy, derivative = bias_energy(0.5, config)
    assert energy == 0. and derivative == 0.


def test_umbrella_energy_is_symmetric_about_the_window_center(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    bind(config, raw, directory)
    below, _ = bias_energy(0.4, config)
    above, _ = bias_energy(0.6, config)
    assert below == pytest.approx(above)


def accepted(tmp_path, gap=0.8, step=0):
    """Build an accepted umbrella frame over a synthetic four-atom region."""
    from qmdpa.cdft_runtime import schema
    from qmdpa.core import Geometry
    from qmdpa.electronic_gap import gap_frame

    config, raw, directory = umbrella_config(tmp_path)
    bind(config, raw, directory)
    geometry = Geometry([1, 2, 3, 4], ["O", "H", "O", "H"],
                         np.array([[0., 0., 0.], [0.96, 0., 0.],
                                   [3.2, 0., 0.], [4.1, 0., 0.]]))
    # The two constrained states carry different forces, so the constrained gap
    # has a non-zero Jacobian and the restraint pullback is exercised.
    forces = np.array([[.1, 0., 0.], [.1, 0., 0.], [.1, 0., 0.], [.1, 0., 0.]])
    partner = np.array([[-.2, 0., 0.], [-.2, 0., 0.], [-.2, 0., 0.], [-.2, 0., 0.]])
    cv = gap_frame(schema(config), geometry,
                   {"R": Evaluation(gap, forces), "P": Evaluation(0., partner)},
                   ("R", "P"))

    class Calculator:
        def __init__(self):
            self.last = (geometry, None, None, cv)
            self.bias = FakeBias(schema(config), restraint(config), identity(config))

        def clear(self):
            pass

    calculator = Calculator()
    frame = {"step": step, "potential_eV": 0.,
             "atoms": [{"id": i, "x": x.tolist(), "f": f.tolist(), "image": [0, 0, 0]}
                       for i, x, f in zip(geometry.atom_ids, geometry.positions, forces,
                                          strict=True)]}
    # The umbrella owns the native-total assembly while it is the bias owner:
    # the native total is the sampled state plus the restraint. The frame is
    # completed here so that the accepted record can be validated against it.
    # The constrained-state record is attached before the umbrella capture in
    # the native flow, and the fixture follows the same order.
    energy, derivative = bias_energy(float(cv.values[0]), config)
    pullback = (-derivative*cv.jacobian[0]).tolist()
    frame["cdft_states"] = pair_record(config, geometry, cv, gap, forces, partner)
    frame["potential_eV"] = gap+energy
    for atom, sampled, pull in zip(frame["atoms"], forces, pullback, strict=True):
        atom["f"] = (sampled+pull).tolist()
    coupling = SimpleNamespace(failure=None, last_step=step, calculator=calculator)
    capture(frame, coupling, config)
    return config, frame


def pair_record(config, geometry, cv, gap, forces, partner=None):
    """Build a complete constrained-pair record for the synthetic geometry."""
    from qmdpa.cdft_runtime import geometry_digest

    if partner is None:
        partner = forces
    q = geometry.select(config["qm_region"]["atom_ids"])
    m = geometry.select(config["environment_region"]["atom_ids"])
    q_force = np.asarray([forces[geometry.atom_ids.index(i)]
                          for i in config["qm_region"]["atom_ids"]])
    m_force = np.asarray([forces[geometry.atom_ids.index(i)]
                          for i in config["environment_region"]["atom_ids"]])
    reference = config["provider"]["state"]["reference"]
    states = {}
    shared = {"environment_energy_eV": 0.,
              "environment_forces_eV_per_angstrom": np.zeros_like(m_force).tolist(),
              "cross_energy_eV": 0.,
              "cross_qm_forces_eV_per_angstrom": np.zeros_like(q_force).tolist(),
              "cross_environment_forces_eV_per_angstrom": np.zeros_like(m_force).tolist()}
    for role, energy, force_set in (("R", gap, forces), ("P", 0., partner)):
        rq = np.asarray([force_set[geometry.atom_ids.index(i)]
                         for i in config["qm_region"]["atom_ids"]])
        rm = np.asarray([force_set[geometry.atom_ids.index(i)]
                         for i in config["environment_region"]["atom_ids"]])
        states[role] = {
            "energy_eV": energy,
            "forces_eV_per_angstrom": force_set.tolist(),
            "quantum": {"energy_eV": energy,
                        "qm_forces_eV_per_angstrom": rq.tolist(),
                        "environment_forces_eV_per_angstrom": rm.tolist(),
                        "diagnostics": _diagnostics(config, role, energy)},
        }
    record = {"schema_version": 1, "reference_sha256": reference["sha256"],
              "geometry_sha256": geometry_digest(q, m),
              "states": states, "shared": shared, "cv": cv.record()}
    record["sha256"] = digest(record)
    return record


def _diagnostics(config, role, energy):
    """Diagnostics that satisfy the constrained-state validator."""
    from qmdpa.cdft import atomic_reference_digest

    ref = reference()
    multiplier = 0.1 if role == "R" else -0.1
    target = -1. if role == "R" else 1.
    population = target+1e-12
    # The fixture carries energies in electronvolt; the diagnostics record them
    # in hartree, as the solver does.
    hartree = energy/HARTREE_EV
    iterations = [{"multiplier_hartree": multiplier,
                   "population": population,
                   "physical_energy_hartree": hartree,
                   "orbital_residual": 1e-9, "scf_cycles": 3}]
    # The electron ledger and the signed identities must agree with the frozen
    # reference: two QM atoms with weights +1 and -1 and five/nine electrons.
    populations = [5., 4.]
    spins = [1., 0.]
    weights = np.asarray(ref["atom_weights"])
    return {"target": target, "population": population,
            "density_electron_counts": [5., 4.],
            "lowdin_atom_populations": populations,
            "lowdin_atom_spins": spins,
            "lowdin_signed_population": float(weights @ populations),
            "lowdin_signed_spin": float(weights @ spins),
            "residual": population-target,
            "multiplier_hartree": multiplier,
            "residual_energy_hartree": multiplier*(population-target),
            "physical_energy_hartree": hartree,
            "augmented_energy_hartree": hartree+multiplier*(population-target),
            "spin_square": 0.75,
            "spin_population": -target,
            "electron_counts": [5, 4],
            "reference_sha256": atomic_reference_digest(ref["atomic_references"]),
            "iterations": iterations}


def test_accepted_umbrella_frame_records_the_gap_and_window(tmp_path):
    config, frame = accepted(tmp_path)
    record = frame["cdft_umbrella"]
    # The gap is the P-minus-R energy difference of the two constrained states.
    assert record["gap_eV"] == pytest.approx(-0.8)
    assert record["window_center_eV"] == 0.5
    assert record["bias_energy_eV"] == pytest.approx(0.5*0.25*1.3*1.3)
    assert record["derivatives_eV_per_unit"][GAP_NAME] == pytest.approx(0.25*(-1.3))
    validate_frame(frame, config)


def test_accepted_umbrella_frame_rejects_a_tampered_gap(tmp_path):
    config, frame = accepted(tmp_path)
    bad = deepcopy(frame)
    bad["cdft_umbrella"]["gap_eV"] = 0.1
    bad["cdft_umbrella"]["sha256"] = digest(
        {k: v for k, v in bad["cdft_umbrella"].items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_frame(bad, config)


def test_accepted_umbrella_frame_rejects_a_tampered_window(tmp_path):
    config, frame = accepted(tmp_path)
    bad = deepcopy(frame)
    bad["cdft_umbrella"]["kappa_per_eV"] = 0.5
    bad["cdft_umbrella"]["sha256"] = digest(
        {k: v for k, v in bad["cdft_umbrella"].items() if k != "sha256"})
    with pytest.raises(ValueError, match="window differs"):
        validate_frame(bad, config)


def test_accepted_umbrella_frame_rejects_a_tampered_bias_energy(tmp_path):
    config, frame = accepted(tmp_path)
    bad = deepcopy(frame)
    bad["cdft_umbrella"]["bias_energy_eV"] = 0.
    bad["cdft_umbrella"]["sha256"] = digest(
        {k: v for k, v in bad["cdft_umbrella"].items() if k != "sha256"})
    with pytest.raises(ValueError, match="restraint energy"):
        validate_frame(bad, config)


def test_umbrella_restart_requires_the_same_gap_and_bias(tmp_path):
    config, frame = accepted(tmp_path)
    compare_restart(frame, deepcopy(frame), config)
    # A restart recomputes the gap at the same geometry. A different gap means
    # the recomputation is not the same physical state and must be rejected.
    other = accepted(tmp_path, gap=0.6, step=0)[1]
    with pytest.raises(ValueError, match="restart gap"):
        compare_restart(other, frame, config)


def test_umbrella_owns_the_native_total_assembly(tmp_path):
    """The umbrella, not the cDFT provider, owns the biased native total.

    Regression control for the ownership defect that made every biased native
    run fail: the cDFT provider asserted the native total against the bare
    sampled surface while the umbrella was the active bias owner, so the two
    owners contradicted each other. The umbrella must assert that the native
    total is the sampled state plus the restraint, and a wrong native total
    must be rejected.
    """
    from qmdpa.cdft_runtime import validate_frame as validate_cdft_frame

    config, frame = accepted(tmp_path)
    # The cDFT provider must not claim the native total while the umbrella is
    # the bias owner.
    validate_cdft_frame(frame, config)
    # The native total must equal the sampled state plus the restraint.
    validate_frame(frame, config)
    # A native total that omits the restraint must be rejected.
    bad = deepcopy(frame)
    bad["potential_eV"] = frame["cdft_states"]["states"]["R"]["energy_eV"]
    with pytest.raises(ValueError, match="native total energy"):
        validate_frame(bad, config)
    # A native force that omits the pullback must be rejected.
    bad = deepcopy(frame)
    for atom in bad["atoms"]:
        atom["f"] = [0., 0., 0.]
    with pytest.raises(ValueError, match="native total forces"):
        validate_frame(bad, config)


def test_umbrella_identity_binds_the_admitted_window(tmp_path):
    config, raw, directory = umbrella_config(tmp_path)
    bind(config, raw, directory)
    identity_record = identity(config)
    assert identity_record["parameters"][GAP_NAME] == {"center": 0.5, "kappa": 0.25,
                                                        "unit": "eV"}
    assert identity_record["history_kind"] == "static-restraint-v1"
