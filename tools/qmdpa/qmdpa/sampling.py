#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Declared finite CV sampling through the official LAMMPS/PLUMED fix.

Production forces and bias energies belong to PLUMED. Geometry calculations here
validate the CV domain and record coordinates; they do not implement a force.
"""

import math
import os
import shutil
from pathlib import Path

import numpy as np

from . import metadynamics, opes, water_cv
from .bias import MIN_DISTANCE_ANGSTROM, NativeBiasAtoms, _coordinates, _scalar
from .config import file_digest
from .dynamics import extend_identity, fixed_sampling_region, positive_scalar, require_ground_sampling

FIX_ID = "qmdpa_sampling"


def _path_token(path):
    value = str(path)
    if not value or any(c.isspace() or c in "'\"#$;,{}" for c in value):
        raise ValueError("PLUMED paths must not contain whitespace, quotes or command characters")
    return value


def bind_sampling(config, raw, base_dir):
    require_ground_sampling(config)
    if "bias" in config:
        raise ValueError("Choose one bias interface: sampling or legacy bias")
    if not isinstance(raw, dict) or set(raw) != {"engine", "kernel", "cv", "bias"}:
        raise ValueError("sampling requires engine, kernel, cv and bias")
    if raw["engine"] != "plumed":
        raise ValueError("sampling.engine must be plumed")
    if config["hamiltonian"] == "finite-reactive-water-mechanical" and (
            not isinstance(raw["cv"], dict) or raw["cv"].get("form") != water_cv.FORM
            or not isinstance(raw["bias"], dict) or raw["bias"].get("form") != opes.FORM):
        raise ValueError("Reactive-water sampling requires OPES with the complete declared water-pool CV")
    kernel = raw["kernel"]
    if not isinstance(kernel, dict) or set(kernel) != {"path", "sha256"}:
        raise ValueError("sampling.kernel requires path and sha256")
    if not isinstance(kernel["path"], str):
        raise ValueError("sampling.kernel.path must be a string")  # noqa: TRY004 - Configuration API.
    path = (Path(base_dir) / kernel["path"]).resolve()
    _path_token(path)
    if not path.is_file() or file_digest(path) != kernel["sha256"]:
        raise ValueError("PLUMED kernel SHA256 differs from the declared file")
    cv = raw["cv"]
    if isinstance(cv, dict) and cv.get("form") == water_cv.FORM:
        value_cv = water_cv.validate_cv(cv, config)
    else:
        value_cv = _distance_cv(cv, config)
    bias = raw["bias"]
    if not isinstance(bias, dict):
        raise ValueError("sampling.bias must be a declared bias object")  # noqa: TRY004 - Configuration API.
    if bias.get("form") == opes.FORM:
        if value_cv["form"] != water_cv.FORM:
            raise ValueError("OPES currently requires the water self-ionization CV pair")
        value_bias = opes.validate_bias(bias, config)
    elif value_cv["form"] == water_cv.FORM:
        raise ValueError("Water Voronoi sampling currently requires OPES")
    elif bias.get("form") == metadynamics.FORM:
        value_bias = metadynamics.validate_bias(bias, config)
    else:
        if set(bias) != {"form", "center_angstrom", "k_ev_per_angstrom2"}:
            raise ValueError("sampling.bias requires form, center_angstrom and k_ev_per_angstrom2")
        if bias["form"] != "harmonic":
            raise ValueError("Choose harmonic or well-tempered-metadynamics sampling")
        center = _scalar(bias["center_angstrom"], "sampling.bias.center_angstrom")
        if cv["form"] == "distance" and center <= 0:
            raise ValueError("A distance restraint center must be positive")
        value_bias = {
            "form": "harmonic", "center_angstrom": center,
            "k_ev_per_angstrom2": positive_scalar(
                bias["k_ev_per_angstrom2"], "sampling.bias.k_ev_per_angstrom2"),
        }
    value = {
        "engine": "plumed",
        "kernel": {"path": str(path), "sha256": kernel["sha256"]},
        "cv": value_cv,
        "bias": value_bias,
    }
    extend_identity(config, "sampling", value)
    return config


def _distance_cv(cv, config):
    if not isinstance(cv, dict) or set(cv) != {"form", "atom_ids"}:
        raise ValueError("sampling.cv requires form and atom_ids")
    sizes = {"distance": 2, "distance-difference": 3}
    if not isinstance(cv["form"], str) or cv["form"] not in sizes:
        raise ValueError("sampling.cv.form must be distance or distance-difference")
    ids = cv["atom_ids"]
    if (not isinstance(ids, list) or len(ids) != sizes[cv["form"]]
            or any(type(i) is not int or i <= 0 for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("sampling.cv requires distinct positive integer atom IDs")
    if not set(ids) <= set(fixed_sampling_region(config)["atom_ids"]):
        raise ValueError("All sampling CV atoms must remain in the fixed QM region")
    return {"form": cv["form"], "atom_ids": sorted(ids) if cv["form"] == "distance" else ids}


def history_contract(config):
    if opes.active(config):
        return opes
    return metadynamics if metadynamics.active(config) else None


def validate_checkpoint_history(checkpoint, checkpoint_path, config):
    contract = history_contract(config)
    if contract is not None:
        contract.validate_checkpoint_history(checkpoint, checkpoint_path, config)
    elif checkpoint.get("sampling_history") is not None:
        raise ValueError("Unexpected sampling checkpoint history")


def cv_value(cv, coordinates):
    if cv["form"] == water_cv.FORM:
        return water_cv.values(cv, coordinates)
    ids = cv["atom_ids"]
    pairs = [(ids[0], ids[1])]
    if cv["form"] == "distance-difference":
        # The middle atom is the transferring H: donor-H minus acceptor-H.
        pairs.append((ids[2], ids[1]))
    distances = [float(np.linalg.norm(coordinates[a] - coordinates[b])) for a, b in pairs]
    if any(not math.isfinite(r) or r <= MIN_DISTANCE_ANGSTROM for r in distances):
        raise ValueError("Sampling distances must exceed 1e-10 Angstrom")
    return distances[0] if len(distances) == 1 else distances[0] - distances[1]


def render_plumed(sampling, directory, *, restart=False):
    directory = Path(directory)
    ids = sampling["cv"]["atom_ids"]
    rows = (["RESTART"] if restart else []) + ["UNITS LENGTH=A ENERGY=eV TIME=ps"]
    if sampling["cv"]["form"] == water_cv.FORM:
        rows += water_cv.render(sampling["cv"])
    elif sampling["cv"]["form"] == "distance":
        rows.append(f"qmdpa_cv: DISTANCE ATOMS={ids[0]},{ids[1]} NOPBC")
    else:
        rows += [
            f"qmdpa_donor: DISTANCE ATOMS={ids[0]},{ids[1]} NOPBC",
            f"qmdpa_acceptor: DISTANCE ATOMS={ids[2]},{ids[1]} NOPBC",
            ("qmdpa_cv: COMBINE ARG=qmdpa_donor,qmdpa_acceptor "
             "COEFFICIENTS=1,-1 PERIODIC=NO"),
        ]
    bias = sampling["bias"]
    if bias["form"] == opes.FORM:
        rows.append(opes.render(bias, directory, _path_token, restart=restart))
    elif bias["form"] == "harmonic":
        rows.append(
            "qmdpa_bias: RESTRAINT ARG=qmdpa_cv "
            f"AT={bias['center_angstrom']:.17g} KAPPA={bias['k_ev_per_angstrom2']:.17g}"
        )
    else:
        rows.append(
            "qmdpa_bias: METAD ARG=qmdpa_cv "
            f"SIGMA={bias['sigma_angstrom']:.17g} HEIGHT={bias['height_ev']:.17g} "
            f"PACE={bias['pace_steps']} BIASFACTOR={bias['bias_factor']:.17g} "
            f"TEMP={bias['temperature_K']:.17g} FILE={_path_token(directory / 'HILLS')} "
            "FMT={ %.17g}"
        )
    observables = (",".join((*water_cv.LABELS, "qmdpa_iondistance"))
                   if sampling["cv"]["form"] == water_cv.FORM else "qmdpa_cv")
    rows += [
        (f"PRINT ARG={observables},qmdpa_bias.bias "
         f"FILE={_path_token(directory / 'COLVAR')} STRIDE=1 FMT={{ %.17g}}"),
        "FLUSH STRIDE=1",
    ]
    return "\n".join(rows) + "\n"


def verify_loaded_kernel(sampling):
    """Check the actual Linux native mapping, not only an environment variable."""
    if (sampling["bias"]["form"] == metadynamics.FORM
            and "PLUMED_DP2CUTOFF_NOSTRETCH" in os.environ):
        raise ValueError("WTMetaD requires the default stretched Gaussian kernel")
    declared = Path(sampling["kernel"]["path"]).resolve()
    expected = sampling["kernel"]["sha256"]
    if file_digest(declared) != expected:
        raise ValueError("PLUMED kernel changed after configuration validation")
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise ValueError("This sampling profile requires Linux native-library identity checks")
    paths = {
        Path(line.split(None, 5)[5]).resolve()
        for line in maps.read_text().splitlines()
        if len(line.split(None, 5)) == 6 and "libplumedKernel" in line.split(None, 5)[5]
    }
    if not paths or any(path != declared for path in paths):
        raise ValueError("Loaded PLUMED kernel differs from the declared runtime")
    return {"path": str(declared), "sha256": expected}


class NativePlumedBias(NativeBiasAtoms):
    def __init__(self, lmp, config, output, comm, previous=None):
        from .runtime import _quoted, _root_call

        self.lmp = lmp
        self.config = config
        self.sampling = config["sampling"]
        self.ids = set(self.sampling["cv"]["atom_ids"])
        if lmp.extract_global("units") != "metal" or any(lmp.extract_box()[5]):
            raise ValueError("Sampling requires metal units and finite native boundaries")
        if not lmp.has_style("fix", "plumed"):
            raise ValueError("The loaded LAMMPS library has no plumed fix")
        if lmp.has_id("fix", FIX_ID):
            raise ValueError("A native sampling fix already exists")
        self.directory = Path(output) / "sampling"
        self.input_path = self.directory / "plumed.dat"
        self.history = history_contract(config)
        self.time_dependent = self.history is not None
        self.history_file = "STATE" if opes.active(config) else "HILLS"
        self.input_text = render_plumed(
            self.sampling, self.directory, restart=self.time_dependent and previous is not None
        )

        def prepare():
            self.directory.mkdir()
            if self.time_dependent and previous is not None:
                source = Path(previous["sampling_history_path"])
                if self.history.history_record(source, config, previous["frame"]["step"]) != previous["frame"]["sampling"]["history"]:
                    raise ValueError("Sampling accepted history changed before loading")
                target = "PARENT_STATE" if opes.active(config) else "HILLS"
                shutil.copyfile(source, self.directory / target)
                if opes.active(config):
                    # Before the next deposition, the parent is the exact current state.
                    shutil.copyfile(source, self.directory / "STATE")
            self.input_path.write_text(self.input_text)
            return file_digest(self.input_path)
        self.input_sha256 = _root_call(comm, prepare)
        lmp.command(
            f"fix {FIX_ID} all plumed plumedfile {_quoted(self.input_path)} "
            f"outfile {_quoted(self.directory / 'plumed.log')}"
        )
        lmp.command(f"fix_modify {FIX_ID} energy yes")
        # Every rank checks its own loaded native mapping before entering run.
        try:
            packet = (None, verify_loaded_kernel(self.sampling))
        except Exception as exc:  # noqa: BLE001 - Preserve collective or top-level failure reporting.
            packet = (f"{type(exc).__name__}: {exc}", None)
        packets = [packet] if comm is None else comm.allgather(packet)
        if any(error is not None for error, _ in packets):
            raise ValueError("; ".join(error for error, _ in packets if error is not None))
        if any(identity != packets[0][1] for _, identity in packets):
            raise ValueError("MPI ranks loaded different PLUMED kernels")

    def validate_packets(self, packets):
        coordinates = _coordinates([row for packet in packets for row in packet], self.ids)
        cv_value(self.sampling["cv"], coordinates)

    def capture(self, frame, comm):
        from .runtime import _root_call

        # The native scalar is queried on all ranks.
        energy = float(self.lmp.extract_fix(FIX_ID, 0, 0))

        def attach():
            if not math.isfinite(energy):
                raise ValueError("Native PLUMED bias energy must be finite")
            if file_digest(self.input_path) != self.input_sha256:
                raise ValueError("Generated PLUMED input changed during the run")
            xyz = {atom["id"]: np.asarray(atom["x"]) for atom in frame["atoms"]
                   if atom["id"] in self.ids}
            frame["sampling"] = {
                "engine": "plumed",
                ("cv_values" if opes.active(self.config) else "cv_angstrom"):
                    cv_value(self.sampling["cv"], xyz),
                "bias_energy_eV": energy,
                "unbiased_potential_eV": frame["potential_eV"] - energy,
            }
            if self.time_dependent:
                frame["sampling"]["history"] = self.history.history_record(
                    self.directory / self.history_file, self.config, frame["step"]
                )
            validate_sampling_frame(frame, self.config)
        _root_call(comm, attach)

    def advance(self, coupling, count):
        # OPES can change the force at deposition, including through normalization
        # and kernel merging. Refresh after EVERY step, independently of output
        # cadence, so the next half-kick and any restart use the same current force.
        if opes.active(self.config):
            for _ in range(count):
                coupling.run(1)
                coupling.run(0)
        else:
            coupling.run(count)

    def refresh(self, coupling):
        # Capture the post-deposition potential with unchanged coordinates.
        # Fixed-width hills have zero force at their deposition center.
        # The native fix suppresses a second history update at this MD step.
        if metadynamics.active(self.config):
            coupling.run(0)

    def save_history(self, checkpoint, output):
        if self.time_dependent:
            self.history.snapshot_history(
                checkpoint, output, self.directory / self.history_file, self.config)


def validate_sampling_frame(frame, config):
    record = frame.get("sampling")
    if "sampling" not in config:
        if record is not None:
            raise ValueError("Unexpected sampling frame record")
        return
    cv_field = "cv_values" if opes.active(config) else "cv_angstrom"
    fields = {"engine", cv_field, "bias_energy_eV", "unbiased_potential_eV"}
    contract = history_contract(config)
    if contract is not None:
        fields.add("history")
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Missing or invalid sampling frame record")
    if contract is not None:
        contract.validate_frame_history(frame, config)
    if record["engine"] != "plumed":
        raise ValueError("Sampling frame engine differs")
    for name in ("bias_energy_eV", "unbiased_potential_eV"):
        if type(record[name]) not in (int, float) or not math.isfinite(record[name]):
            raise ValueError("Sampling frame values must be finite scalars")
    cv = config["sampling"]["cv"]
    expected = set(cv["atom_ids"])
    xyz = _coordinates([(atom["id"], atom["x"]) for atom in frame["atoms"]
                        if atom["id"] in expected], expected)
    observed = record[cv_field]
    expected_cv = cv_value(cv, xyz)
    if opes.active(config):
        if not isinstance(observed, dict) or set(observed) != set(water_cv.OBSERVABLES):
            raise ValueError("Sampling frame water CV fields differ")
        pairs = [(observed[key], expected_cv[key]) for key in water_cv.OBSERVABLES]
    else:
        pairs = [(observed, expected_cv)]
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or abs(value - target) > 1e-10 for value, target in pairs):
        raise ValueError("Sampling frame CV differs from its coordinates")
    if abs(frame["potential_eV"] - record["bias_energy_eV"]
           - record["unbiased_potential_eV"]) > 1e-9:
        raise ValueError("Sampling frame energy ledger differs")


def compare_sampling_restart(frame, reference, config):
    for item in (frame, reference):
        validate_sampling_frame(item, config)
    if history_contract(config) is not None and frame["sampling"]["history"] != reference["sampling"]["history"]:
        raise ValueError("Restart sampling accepted history differs")
    if "sampling" in config:
        if opes.active(config):
            for key in water_cv.OBSERVABLES:
                if abs(frame["sampling"]["cv_values"][key] - reference["sampling"]["cv_values"][key]) > 1e-10:
                    raise ValueError(f"Restart sampling {key} differs")
        elif abs(frame["sampling"]["cv_angstrom"] - reference["sampling"]["cv_angstrom"]) > 1e-10:
            raise ValueError("Restart sampling cv_angstrom differs")
        for key, atol in (("bias_energy_eV", 1e-9), ("unbiased_potential_eV", 1e-6)):
            if abs(frame["sampling"][key] - reference["sampling"][key]) > atol:
                raise ValueError(f"Restart sampling {key} differs")
