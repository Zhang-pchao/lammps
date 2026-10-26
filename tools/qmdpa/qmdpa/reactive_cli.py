#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Native total-potential execution for the finite reactive-water Hamiltonian."""

from .additive_cli import _zero_pair  # noqa: F401 - Unified route reuses this entrypoint.


def _reactive_coupling(lmp, comm, config):
    from .backends import DeepMDBackend
    from .inference import StrictRawCalculator, prepare_raw_inference, strict_runtime_state
    from .lammps import AdditiveExternalCoupling
    from .providers import make_qm_provider
    from .reactive_config import build_mixer
    from .region import FixedRegion
    from .runtime import _root_call

    full = config["full_region"]
    region = FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"]))
    types = tuple(config["type_symbols"])
    _root_call(comm, lambda: AdditiveExternalCoupling._validate_native_assembly(lmp, region, types))
    calculator = None

    def make_calculator():
        nonlocal calculator
        torch = prepare_raw_inference(config["model"])
        low = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"], no_jit=True)
        if low.no_jit is not True:
            raise RuntimeError("The reactive DPA provider did not retain no_jit=True")
        high = make_qm_provider(config["qm"], provider=config["provider"])
        mixer = build_mixer(config, low, high)
        backend = low._potential.deep_eval
        config["inference_runtime"] = {
            **strict_runtime_state(torch), "no_jit": True,
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "shared_full_and_fragment_model": True,
        }
        calculator = StrictRawCalculator(mixer, torch)

    _root_call(comm, make_calculator)
    return AdditiveExternalCoupling(lmp, region, calculator, comm=comm, type_symbols=types,
                                    fix_id="qmdpa_reactive_water")
