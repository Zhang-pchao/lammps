# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
"""Build the root-owned selected-family total for the shared native driver."""

from .additive_cli import _zero_pair  # noqa: F401 - Dispatch route entrypoint.


def _selected_adaptive_coupling(lmp, comm, config):
    from .backends import DeepMDBackend
    from .core import SubtractiveCalculator
    from .inference import StrictRawCalculator, prepare_raw_inference, strict_runtime_state
    from .lammps import AdditiveExternalCoupling
    from .molecular_adaptive import QuinticMolecularWeights
    from .region import FixedRegion
    from .runtime import _root_call
    from .selected_adaptive import MolecularSelectedMixer, SelectedMechanicalPartition
    from .selection_config import FixedGroup
    from .state_family import FamilyTDABackend, StateFamilyReference

    full = config["full_region"]
    region = FixedRegion(tuple(full["atom_ids"]), tuple(full["symbols"]))
    types = tuple(config["type_symbols"])
    _root_call(comm, lambda: AdditiveExternalCoupling._validate_native_assembly(lmp, region, types))
    calculator = observer = None

    def build():
        nonlocal calculator, observer
        torch = prepare_raw_inference(config["model"])
        low = DeepMDBackend(config["model"], charge_spin=config["full_charge_spin"], no_jit=True)
        family = StateFamilyReference.from_record(config["family_reference"])
        partitions = tuple(SelectedMechanicalPartition(
            calculator=SubtractiveCalculator(low, FamilyTDABackend(family, p["partition_id"])), **p)
            for p in config["partitions"])
        observer = MolecularSelectedMixer(
            low, groups=tuple(FixedGroup(**g) for g in config["groups"]),
            core_group_ids=config["core_group_ids"], partitions=partitions,
            switch=QuinticMolecularWeights(**{k: v for k, v in config["switch"].items() if k != "method"}),
            family=family)
        calculator = StrictRawCalculator(observer, torch)
        config["inference_runtime"] = {**strict_runtime_state(torch), "no_jit": low.no_jit,
                                       "shared_full_and_region_model": True}

    _root_call(comm, build)
    coupling = AdditiveExternalCoupling(lmp, region, calculator, comm=comm,
                                        type_symbols=types, fix_id="qmdpa_selected_family")
    coupling.selected_family_observer = observer
    return coupling
