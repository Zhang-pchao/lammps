#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Fixed CPU ground/TDA gap, complete two-surface forces and named restraint.

Existing single-surface calculators own the Hamiltonians and quantum solvers.
This module supplies explicit admission, call-local sharing and the chain rule.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .core import Evaluation, Geometry, SubtractiveCalculator
from .named_cv import CVFrame, NamedCVSchema, digest, geometry_identity

FORM = "ground-tda-gap-restraint-v1"
ME = "fixed-region-mechanical"
EE = "finite-electrostatic-additive"


def bind(config, raw, directory):
    """Admit a finite CPU pair without broadening legacy ground sampling."""
    from .config import file_digest, validate_region
    from .dynamics import extend_identity
    from .state_records import configured_reference

    required = {"form", "name", "order", "sampled_surface", "full_region",
                "restraint", "kernel", "wrapper"}
    if not isinstance(raw, dict) or set(raw) != required or raw["form"] != FORM:
        raise ValueError("electronic_gap requires explicit pair, full region, restraint and libraries")
    if (config["hamiltonian"] not in (ME, EE) or config["periodic"]
            or config["provider"]["engine"] != "pyscf"
            or config["provider"]["device"] != "cpu"
            or config["provider"]["state"]["method"] != "rhf-tda"
            or any(k in config for k in ("sampling", "bias", "thermostat", "execution"))
            or config["sample_interval"] != 1):
        raise ValueError("Electronic gap requires finite fixed CPU RHF/TDA ME or additive NVE, sampled every step")
    reference = configured_reference(config)
    if (reference.settings.tda_solver != "dense"
            or reference.settings.xc != "hf"
            or config["model"].endswith(".pt") is False):
        raise ValueError("Electronic gap requires dense RHF/TDA and an explicit raw PyTorch model")
    if raw["order"] not in (["ground", "selected"], ["selected", "ground"]):
        raise ValueError("Gap order must contain ground and selected exactly once")
    if raw["sampled_surface"] not in ("ground", "selected"):
        raise ValueError("The sampled surface must be ground or selected")
    validate_region(raw["full_region"])
    full = deepcopy(raw["full_region"])
    q = config["region" if config["hamiltonian"] == ME else "qm_region"]
    mapping = dict(zip(full["atom_ids"], full["symbols"], strict=True))
    regions = [q] + ([config["environment_region"]] if config["hamiltonian"] == EE else [])
    for region in regions:
        if any(mapping.get(i) != s for i, s in zip(region["atom_ids"], region["symbols"], strict=True)):
            raise ValueError("Gap full region differs from the fixed state atom identity")
    if config["hamiltonian"] == EE and set(mapping) != set(q["atom_ids"]+regions[1]["atom_ids"]):
        raise ValueError("Gap full region must cover QM and environment exactly")
    if set(mapping) == set(q["atom_ids"]):
        raise ValueError("Electronic gap currently requires a nonempty explicit environment")
    params = raw["restraint"]
    if (not isinstance(params, dict) or set(params) != {"center", "kappa", "unit"}
            or params["unit"] != "eV"
            or any(type(params[k]) not in (int, float) or not np.isfinite(params[k])
                   for k in ("center", "kappa")) or params["kappa"] <= 0):
        raise ValueError("Gap restraint requires center in eV and positive kappa in 1/eV")
    value = deepcopy(raw)
    paths = {}
    for name in ("kernel", "wrapper"):
        item = value[name]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Gap libraries require path and SHA256")
        path = (Path(directory)/item["path"]).resolve()
        if file_digest(path) != item["sha256"]:
            raise ValueError("Gap PLUMED library checksum differs")
        paths[name] = str(path)
        value[name] = {"sha256": item["sha256"]}
    value["full_region"] = full
    # Schema validation checks name and the ordered dependency identity now.
    schema_for(config, value)
    extend_identity(config, "electronic_gap", value)
    config["electronic_gap_libraries"] = paths
    return config


def schema_for(config, extension=None):
    from .state_records import configured_reference

    value = config["electronic_gap"] if extension is None else extension
    reference = configured_reference(config)
    dependencies = {"selected_reference": reference.sha256,
                    "ground_settings": digest(config["qm"]),
                    "pair_and_partition": digest({"hamiltonian": config["hamiltonian"],
                        "order": value["order"], "full_region": value["full_region"],
                        "qm": config.get("region", config.get("qm_region")),
                        "embedding": config.get("embedding"), "cross": config.get("cross")}),
                    "model": config["model_sha256"]}
    return NamedCVSchema({"schema_version": 1, "coordinate_unit": "angstrom", "variables": [
        {"name": value["name"], "shape": [], "unit": "eV", "dependencies": dependencies}]})


def native_geometry(frame, full_region):
    atoms = {a["id"]: a for a in frame["atoms"]}
    ids = full_region["atom_ids"]
    if len(atoms) != len(frame["atoms"]) or set(atoms) != set(ids):
        raise ValueError("Electronic gap frame must cover the complete atom identity")
    if any(atoms[i]["image"] != [0, 0, 0] for i in ids):
        raise ValueError("Electronic gap requires finite coordinates without images")
    return Geometry(ids, full_region["symbols"], [atoms[i]["x"] for i in ids])


def gap_frame(schema, geometry, states, order):
    """Convert complete same-call forces into the full signed gap Jacobian."""
    a, b = (states[name] for name in order)
    spec = schema.variables[0]
    return CVFrame(schema, geometry, {spec.name: {
        "value": b.energy-a.energy, "unit": "eV", "atom_ids": list(geometry.atom_ids),
        "jacobian": (a.forces-b.forces).tolist(),
        "geometry_sha256": geometry_identity(geometry), "dependencies": spec.dependencies}})


class GroundTDARestraint:
    """Root-owned evaluator; queries never advance the external bias history."""

    def __init__(self, config, ground, selected, low, *, full_low=None, cross=None, torch=None):
        self.config = deepcopy(config)
        self.ground, self.selected, self.low = ground, selected, low
        self.full_low, self.cross, self.torch = full_low, cross, torch
        self.schema = schema_for(config)
        self.bias = None
        self.clear()

    def clear(self):
        self.last = None
        self.selected.clear()

    def accepted_record(self, frame):
        return self.selected.accepted_record(frame)

    def evaluate(self, geometry, step=0):
        from .additive import AdditiveCalculator
        from .inference import strict_runtime_state
        from .multistate_coupling import _CallCache

        self.clear()
        if self.torch is not None:
            strict_runtime_state(self.torch)
        cfg, ext = self.config, self.config["electronic_gap"]
        full = ext["full_region"]
        ordered = geometry.select(full["atom_ids"])
        if geometry.periodic or set(geometry.atom_ids) != set(full["atom_ids"]) or ordered.symbols != tuple(full["symbols"]):
            raise ValueError("Electronic gap atom identity or finite boundary changed")
        q = cfg["region" if cfg["hamiltonian"] == ME else "qm_region"]
        qm = geometry.select(q["atom_ids"])
        cache = _CallCache()
        states = {}
        try:
            # One call-local cache shares every state-independent term only.
            for name, provider in (("ground", self.ground), ("selected", self.selected)):
                if cfg["hamiltonian"] == ME:
                    states[name] = SubtractiveCalculator(
                        cache.wrap(self.low), cache.wrap(provider),
                        full_low=cache.wrap(self.full_low)).evaluate(geometry, qm)
                else:
                    states[name] = AdditiveCalculator(
                        cache.wrap(self.low), cache.wrap(provider), cache.wrap(self.cross),
                        qm_atom_ids=q["atom_ids"],
                        environment_atom_ids=cfg["environment_region"]["atom_ids"]).evaluate(geometry)
            pseudo = {"atoms": [{"id": i, "x": x.tolist(), "image": [0, 0, 0]}
                                for i, x in zip(geometry.atom_ids, geometry.positions, strict=True)]}
            state_record = self.selected.accepted_record(pseudo)
            ground_results = cache.wrap(self.ground).results
            if len(ground_results) != 1:
                raise ValueError("Gap ground provider must evaluate one same-frame input")
            ground = next(iter(ground_results.values()))
            energy = ground.energy if isinstance(ground, Evaluation) else ground.energy_ev
            parity = energy-state_record["diagnostics"]["ground_energy_ev"]
            if abs(parity) > 1e-7:
                raise ValueError("Ground and selected RHF baselines differ by more than 1e-7 eV")
            cv = gap_frame(self.schema, geometry, states, ext["order"])
            sampled = states[ext["sampled_surface"]]
            bias = None if self.bias is None else self.bias.evaluate(cv, step)
            total = sampled if bias is None else Evaluation(
                sampled.energy+bias["bias_energy_eV"],
                sampled.forces+np.asarray(bias["forces_eV_per_angstrom"]))
            if self.torch is not None:
                strict_runtime_state(self.torch)
            self.last = (geometry, states, cv, bias, total, parity)
            return total
        except BaseException:
            self.clear()
            raise


def make_calculator(config):
    """Construct the existing matched CPU solvers and raw DPA inference."""
    from .additive import CrossLennardJones
    from .additive_config import cross_parameters
    from .backends import DeepMDBackend, PySCFBackend
    from .embedding import FixedEmbeddingCharges, PySCFElectrostaticBackend
    from .inference import prepare_raw_inference, strict_runtime_state
    from .providers import make_qm_provider
    from .state_records import configured_reference

    torch = prepare_raw_inference(config["model"])
    from deepmd.pt.utils.env import DEVICE

    if DEVICE.type != "cpu":
        raise ValueError("Electronic gap currently requires actual CPU DeePMD inference")
    settings = configured_reference(config).settings
    ground_settings = {k: getattr(settings, k) for k in
                       ("xc", "basis", "charge", "multiplicity", "conv_tol",
                        "conv_tol_grad", "max_cycle", "threads")}
    if config["hamiltonian"] == ME:
        ground = PySCFBackend(**ground_settings)
        selected = make_qm_provider(config["qm"], provider=config["provider"])
        low = DeepMDBackend(config["model"], charge_spin=config["region_charge_spin"], no_jit=True)
        full = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"], no_jit=True)
        cross = None
    else:
        charges = FixedEmbeddingCharges(**config["embedding"])
        ground = PySCFElectrostaticBackend(charges, **ground_settings)
        selected = make_qm_provider(config["qm"], provider=config["provider"], charges=charges)
        low = DeepMDBackend(config["model"], charge_spin=config["environment_charge_spin"], no_jit=True)
        full = None
        cross = CrossLennardJones(cross_parameters(config["cross"]), source=config["cross"]["source"])
    config["inference_runtime"] = {**strict_runtime_state(torch), "device": str(DEVICE)}
    return GroundTDARestraint(config, ground, selected, low, full_low=full, cross=cross, torch=torch)


def make_coupling(lmp, comm, config):
    from .lammps import AdditiveExternalCoupling
    from .region import FixedRegion
    from .runtime import _root_call

    class GapExternalCoupling(AdditiveExternalCoupling):
        _calculator_kind = "ground-tda-gap-restraint"

        def _evaluate_geometry(self, geometry):
            return self.calculator.evaluate(geometry, int(self.lmp.extract_global("ntimestep")))

    calculator = None

    def construct():
        nonlocal calculator
        calculator = make_calculator(config)

    _root_call(comm, construct)
    full = config["electronic_gap"]["full_region"]
    return GapExternalCoupling(lmp, FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"])),
                               calculator, comm=comm, type_symbols=tuple(config["type_symbols"]),
                               state_observer=calculator, fix_id="qmdpa_electronic_gap")
