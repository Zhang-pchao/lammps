# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Build a root-owned Ehrenfest model for the shared native driver."""

from .additive_cli import _zero_pair  # noqa: F401 - Dispatch route entrypoint.


def _ehrenfest_coupling(lmp, comm, config):
    from .backends import DeepMDBackend
    from .casscf_pair import CASSCFPairProvider, CASSCFPairReference
    from .ehrenfest_mechanical import EhrenfestStepper, MechanicalPair, _complex_from_record
    from .inference import prepare_raw_inference, strict_runtime_state
    from .lammps import EhrenfestExternalCoupling
    from .region import FixedRegion
    from .runtime import _root_call

    calculator = None

    def build():
        nonlocal calculator
        torch = prepare_raw_inference(config["model"])
        full = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"], no_jit=True)
        region = DeepMDBackend(config["model"], charge_spin=config["region_charge_spin"], no_jit=True)
        provider = CASSCFPairProvider(CASSCFPairReference.from_record(config["pair_reference"]))
        pair = MechanicalPair(full, region, provider, qm_atom_ids=config["qm_region"]["atom_ids"])

        class StrictPair:
            def evaluate_pair(self, geometry):
                strict_runtime_state(torch)
                result = pair.evaluate_pair(geometry)
                strict_runtime_state(torch)
                return result

        calculator = EhrenfestStepper(StrictPair(), timestep_fs=config["timestep_fs"],
                                      initial_amplitudes=_complex_from_record(config["initial_amplitudes"]),
                                      identity_sha256=config["physics_sha256"])
        config["inference_runtime"] = strict_runtime_state(torch)

    _root_call(comm, build)
    full = config["full_region"]
    return EhrenfestExternalCoupling(
        lmp, FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"])), calculator,
        type_symbols=tuple(config["type_symbols"]), comm=comm, fix_id="qmdpa_ehrenfest")
