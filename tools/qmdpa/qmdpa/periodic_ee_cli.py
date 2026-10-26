# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Construct the bounded periodic total; reuse the unified native driver."""

from .additive_cli import _zero_pair  # noqa: F401 - Dispatch route entrypoint.


def _periodic_ee_coupling(lmp, comm, config, *, provider_resources=None):
    from .abacus import AbacusBackend
    from .backends import DeepMDBackend
    from .embedding import FixedEmbeddingCharges
    from .inference import StrictRawCalculator, prepare_raw_inference, strict_runtime_state
    from .lammps import PeriodicReferenceExternalCoupling
    from .periodic_ee import PeriodicReferenceSubtractiveCalculator
    from .periodic_field import PeriodicGaussianField, PeriodicGaussianReference
    from .region import FixedRegion
    from .runtime import _root_call

    calculator = None

    def build():
        nonlocal calculator
        if provider_resources is None:
            raise ValueError("Periodic EE requires owned ABACUS resources")
        torch = prepare_raw_inference(config["model"])
        full = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"], no_jit=True)
        region = DeepMDBackend(config["model"], charge_spin=config["region_charge_spin"], no_jit=True)
        options = {k: v for k, v in config["qm"].items() if k != "engine"}
        provider = provider_resources.register(
            AbacusBackend(**options, directory=provider_resources.directory))
        reference = PeriodicGaussianReference(
            FixedEmbeddingCharges(**config["reference"]["qm_charges"]),
            PeriodicGaussianField(**config["qm"]["periodic_field"]))
        config["inference_runtime"] = strict_runtime_state(torch)
        calculator = StrictRawCalculator(PeriodicReferenceSubtractiveCalculator(
            full, region, provider, reference,
            qm_atom_ids=config["qm_region"]["atom_ids"],
            environment_atom_ids=config["environment_region"]["atom_ids"]), torch)

    _root_call(comm, build)
    qm, env = config["qm_region"], config["environment_region"]
    return PeriodicReferenceExternalCoupling(
        lmp, FixedRegion(tuple(qm["atom_ids"] + env["atom_ids"]),
                         tuple(qm["symbols"] + env["symbols"])), calculator,
        comm=comm, type_symbols=tuple(config["type_symbols"]))
