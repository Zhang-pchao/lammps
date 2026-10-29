#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Native total-potential execution for finite reactive-water Hamiltonians."""

from .additive_cli import _zero_pair  # noqa: F401 - Unified route reuses this entrypoint.


def _reactive_coupling(lmp, comm, config, *, provider_resources=None):
    from .backends import DeepMDBackend
    from .inference import (
        COMPRESSED_PROFILE, StrictRawCalculator, prepare_raw_inference, strict_runtime_state,
    )
    from .lammps import AdditiveExternalCoupling
    from .providers import make_qm_provider
    from .reactive_config import build_mixer
    from .reactive_water import ADDITIVE_HAMILTONIAN
    from .region import FixedRegion
    from .runtime import _root_call

    full = config["full_region"]
    region = FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"]))
    types = tuple(config["type_symbols"])
    _root_call(comm, lambda: AdditiveExternalCoupling._validate_native_assembly(lmp, region, types))
    calculator = None

    def make_calculator():
        nonlocal calculator
        additive = config["hamiltonian"] == ADDITIVE_HAMILTONIAN
        compressed = config["inference_profile"] == COMPRESSED_PROFILE
        if compressed:
            from .execution import uses_native_domains
            from .native_deepmd import NativeDeepMDDomainBackend

            if not additive or not uses_native_domains(config) or provider_resources is None:
                raise ValueError("Compressed reactive EE requires owned native environment resources")
            low = provider_resources.register(NativeDeepMDDomainBackend(
                config["model"], model_sha256=config["model_sha256"], type_symbols=types,
                pair_style=config["execution"]["pair_style"],
                charge_spin=config["environment_charge_spin"]))
            runtime = {"profile": COMPRESSED_PROFILE, "domain": "environment",
                       "pair_style": low.pair_style, "model_sha256": low.model_sha256}
        else:
            torch = prepare_raw_inference(config["model"])
            low = DeepMDBackend(config["model"], charge_spin=config[
                "environment_charge_spin" if additive else "full_charge_spin"], no_jit=True)
            if low.no_jit is not True:
                raise RuntimeError("The reactive DPA provider did not retain no_jit=True")
            backend = low._potential.deep_eval
            runtime = {
                **strict_runtime_state(torch), "no_jit": True,
                "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
                "shared_full_and_fragment_model": not additive, "shared_environment_model": additive,
            }
        high = ((lambda field: make_qm_provider(config["qm"], provider=config["provider"], charges=field))
                if additive else make_qm_provider(config["qm"], provider=config["provider"]))
        mixer = build_mixer(config, low, high)
        config["inference_runtime"] = runtime
        calculator = mixer if compressed else StrictRawCalculator(mixer, torch)

    _root_call(comm, make_calculator)
    return AdditiveExternalCoupling(lmp, region, calculator, comm=comm, type_symbols=types,
                                    fix_id="qmdpa_reactive_water")
