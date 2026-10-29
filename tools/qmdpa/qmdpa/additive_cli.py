# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite electrostatic additive dynamics through an all-atom LAMMPS callback."""

import argparse
import sys
from pathlib import Path


def _zero_pair(lmp, config):
    # The full potential is supplied by the external callback.
    lmp.commands_list(["pair_style zero 1.0", "pair_coeff * *"])


def _additive_coupling(lmp, comm, config):
    from .additive import AdditiveCalculator, CrossLennardJones
    from .additive_config import cross_parameters
    from .backends import DeepMDBackend
    from .embedding import FixedEmbeddingCharges
    from .inference import StrictRawCalculator, prepare_raw_inference, strict_runtime_state
    from .lammps import AdditiveExternalCoupling
    from .providers import make_qm_provider
    from .region import FixedRegion
    from .runtime import _root_call
    from .state_records import RecordingTDAProvider

    if config["provider"]["state"]["method"] == "uks-hirshfeld-cdft":
        from .cdft_runtime import make_coupling

        return make_coupling(lmp, comm, config)
    calculator = state_observer = None

    def make_calculator():
        nonlocal calculator, state_observer
        torch = prepare_raw_inference(config["model"])
        environment = DeepMDBackend(
            config["model"], charge_spin=config["environment_charge_spin"], no_jit=True
        )
        backend = environment._potential.deep_eval
        config["inference_runtime"] = {
            **strict_runtime_state(torch),
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "no_jit": True,
        }
        provider = make_qm_provider(
            config["qm"],
            provider=config["provider"],
            charges=FixedEmbeddingCharges(**config["embedding"]),
        )
        if isinstance(provider, RecordingTDAProvider):
            state_observer = provider
        calculator = StrictRawCalculator(
            AdditiveCalculator(
                environment,
                provider,
                CrossLennardJones(
                    cross_parameters(config["cross"]), source=config["cross"]["source"]
                ),
                qm_atom_ids=config["qm_region"]["atom_ids"],
                environment_atom_ids=config["environment_region"]["atom_ids"],
            ),
            torch,
        )

    _root_call(comm, make_calculator)
    qm, env = config["qm_region"], config["environment_region"]
    region = FixedRegion(
        tuple(qm["atom_ids"] + env["atom_ids"]), tuple(qm["symbols"] + env["symbols"])
    )
    return AdditiveExternalCoupling(
        lmp,
        region,
        calculator,
        comm=comm,
        type_symbols=tuple(config["type_symbols"]),
        state_observer=state_observer,
    )


def run(config_path, output, *, restart=None):
    """Reuse the accepted-frame/checkpoint lifecycle with a distinct Hamiltonian."""
    from .additive_config import load_additive_config
    from .cli import _run

    return _run(
        config_path,
        output,
        restart=restart,
        config_loader=load_additive_config,
        setup_pair=_zero_pair,
        make_coupling=_additive_coupling,
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
        print(f"Additive QM/DPA failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
