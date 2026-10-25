# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Finite reference-subtractive totals using the existing all-atom transport."""

import argparse
import sys
from pathlib import Path

from .additive_cli import _zero_pair


def _reference_coupling(lmp, comm, config):
    from .backends import DeepMDBackend
    from .embedding import FixedEmbeddingCharges
    from .inference import StrictRawCalculator, prepare_raw_inference, strict_runtime_state
    from .lammps import ReferenceExternalCoupling
    from .providers import make_qm_provider
    from .reference import FixedChargeCrossCoulomb, ReferenceSubtractiveCalculator
    from .region import FixedRegion
    from .runtime import _root_call

    calculator = None

    def build():
        nonlocal calculator
        torch = prepare_raw_inference(config["model"])
        charges = FixedEmbeddingCharges(**config["embedding"])
        full = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"], no_jit=True)
        region = DeepMDBackend(
            config["model"], charge_spin=config["region_charge_spin"], no_jit=True
        )
        provider = make_qm_provider(config["qm"], provider=config["provider"], charges=charges)
        reference = FixedChargeCrossCoulomb(
            FixedEmbeddingCharges(**config["reference"]["qm_charges"]), charges
        )
        config["inference_runtime"] = strict_runtime_state(torch)
        calculator = StrictRawCalculator(
            ReferenceSubtractiveCalculator(
                full,
                region,
                provider,
                reference,
                qm_atom_ids=config["qm_region"]["atom_ids"],
                environment_atom_ids=config["environment_region"]["atom_ids"],
            ),
            torch,
        )

    _root_call(comm, build)
    qm, env = config["qm_region"], config["environment_region"]
    return ReferenceExternalCoupling(
        lmp,
        FixedRegion(tuple(qm["atom_ids"] + env["atom_ids"]), tuple(qm["symbols"] + env["symbols"])),
        calculator,
        comm=comm,
        type_symbols=tuple(config["type_symbols"]),
    )


def run(config_path, output, *, restart=None):
    from .cli import _run
    from .reference_config import load_reference_config

    return _run(
        config_path,
        output,
        restart=restart,
        config_loader=load_reference_config,
        setup_pair=_zero_pair,
        make_coupling=_reference_coupling,
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
        print(f"Reference-subtractive QM/DPA failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
