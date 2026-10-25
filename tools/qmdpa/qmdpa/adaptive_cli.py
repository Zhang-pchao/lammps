# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite conservative molecular ground-state dynamics with all atoms free."""

import argparse
import sys
from pathlib import Path

from .additive_cli import _zero_pair


def _adaptive_coupling(lmp, comm, config):
    from .adaptive_config import MOLECULAR_ADDITIVE, build_mixer, molecular_environment_charges
    from .backends import DeepMDBackend
    from .inference import StrictRawCalculator, prepare_raw_inference, strict_runtime_state
    from .lammps import AdditiveExternalCoupling
    from .providers import make_qm_provider
    from .region import FixedRegion
    from .runtime import _root_call

    full = config["full_region"]
    region = FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"]))
    types = tuple(config["type_symbols"])
    # Reuse the all-atom total-potential checks before allocating the model.
    _root_call(comm, lambda: AdditiveExternalCoupling._validate_native_assembly(lmp, region, types))
    calculator = None

    def make_calculator():
        nonlocal calculator
        torch = prepare_raw_inference(config["model"])
        additive = config["hamiltonian"] == MOLECULAR_ADDITIVE
        low = DeepMDBackend(config["model"],
                            charge_spin=config["environment_charge_spin"] if additive else config["full_charge_spin"],
                            no_jit=True)
        if low.no_jit is not True:
            raise RuntimeError("The adaptive DPA provider did not retain no_jit=True")
        high = (tuple(make_qm_provider(config["qm"], provider=config["provider"], charges=field)
                      for field in molecular_environment_charges(config)) if additive else
                tuple(make_qm_provider(config["qm"], provider=config["provider"])
                      for _ in config["partitions"]))
        mixer = build_mixer(config, low, high)
        backend = low._potential.deep_eval
        config["inference_runtime"] = {
            **strict_runtime_state(torch),
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "no_jit": True,
             "shared_full_and_region_model": not additive,
            "shared_environment_model": additive,
        }
        calculator = StrictRawCalculator(mixer, torch)

    _root_call(comm, make_calculator)
    return AdditiveExternalCoupling(
        lmp, region, calculator, comm=comm, type_symbols=types, fix_id="qmdpa_adaptive"
    )


def run(config_path, output, *, restart=None):
    """Reuse accepted frames and binary/hash-checked checkpoints without history."""
    from .adaptive_config import load_adaptive_config
    from .cli import _run

    return _run(
        config_path,
        output,
        restart=restart,
        config_loader=load_adaptive_config,
        setup_pair=_zero_pair,
        make_coupling=_adaptive_coupling,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.config, args.output, restart=args.restart)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports backend failures.
        print(f"Adaptive mechanical QM/DPA failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
