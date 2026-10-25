# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Finite gap-anchor candidate with pair zero and a complete all-atom total."""

import argparse
import sys
from pathlib import Path


def _zero_pair(lmp, config):
    lmp.commands_list(["pair_style zero 1.0", "pair_coeff * *"])


def _gap_anchor_coupling(lmp, comm, config):
    from .backends import DeepMDBackend
    from .gap_anchor import make_gpu_rhf_gap_anchor, make_pbe0_gap_anchor, make_rhf_gap_anchor
    from .lammps import GapAnchorExternalCoupling
    from .providers import load_tda_reference
    from .region import FixedRegion
    from .runtime import _root_call
    from .state_records import RecordingTDAProvider, StateRecordGroup

    class FieldRecorder(RecordingTDAProvider):
        @property
        def charges(self):
            # Forward the live field so the Hamiltonian catches backend mutation.
            return self.backend.charges

    qm,env=config["qm_region"],config["environment_region"]
    engine = config["provider"]["engine"]
    if engine == "gpu4pyscf" and (
        config["provider"]["state"]["method"] != "rhf-tda"
        or config.get("hamiltonian") != "finite-electrostatic-gap-anchor"
        or "interaction_delta" in config or "fragment_response" in config
    ):
        raise ValueError("GPU supports only the plain RHF gap-anchor; no fragment fallback")
    reuse_ground = config.get("reuse_ground_terms", False)
    if (type(reuse_ground) is not bool
            or ("reuse_ground_terms" in config and config["provider"]["state"]["method"] != "pbe0-tda")):
        raise ValueError("reuse_ground_terms requires an explicit boolean and the matched PBE0 route")
    from .interaction_delta_config import CORRECTED_GAP_HAMILTONIAN, validate_interaction_delta
    if (config.get("hamiltonian") == CORRECTED_GAP_HAMILTONIAN) != ("interaction_delta" in config):
        raise ValueError("The corrected Hamiltonian and interaction_delta must be selected together")
    if "interaction_delta" in config:
        validate_interaction_delta(config["interaction_delta"], qm, env,
                                   config["provider"]["state"]["method"])
    from .fragment_response_config import FRAGMENT_GAP_HAMILTONIAN, validate_fragment_response
    if (config.get("hamiltonian") == FRAGMENT_GAP_HAMILTONIAN) != ("fragment_response" in config):
        raise ValueError("Fragment-response Hamiltonian and configuration must be selected together")
    fragment_config = None
    if "fragment_response" in config:
        from .embedding import FixedEmbeddingCharges
        fragment_config = validate_fragment_response(
            config["fragment_response"], qm, env, config["provider"],
            FixedEmbeddingCharges(**config["embedding"]))
    if "reduce_fragment_terms" in config:
        if (type(config["reduce_fragment_terms"]) is not bool or fragment_config is None
                or (config["reduce_fragment_terms"] and not reuse_ground)):
            raise ValueError("reduce_fragment_terms requires explicit fragment response and exact ground reuse")
    region=FixedRegion(tuple(qm["atom_ids"]+env["atom_ids"]),tuple(qm["symbols"]+env["symbols"]))
    types=tuple(config["type_symbols"])
    _root_call(comm,lambda: GapAnchorExternalCoupling._validate_native_assembly(lmp,region,types))
    calculator = observer = None

    def build():
        nonlocal calculator, observer
        state = config["provider"]["state"]
        factories = {("pyscf", "rhf-tda"): make_rhf_gap_anchor,
                     ("pyscf", "pbe0-tda"): make_pbe0_gap_anchor,
                     ("gpu4pyscf", "rhf-tda"): make_gpu_rhf_gap_anchor}
        route = (engine, state["method"])
        if route not in factories:
            raise ValueError("Gap-anchor requires an explicit supported engine and RHF/TDA or PBE0/TDA method")
        reference = load_tda_reference(state["method"], state["reference"], engine=engine)
        full=DeepMDBackend(config["model"],charge_spin=config["full_charge_spin"])
        region=DeepMDBackend(config["model"],charge_spin=config["region_charge_spin"])
        execution = {}
        if "matrix_assembly" in config["provider"]:
            execution["matrix_assembly"] = config["provider"]["matrix_assembly"]
        if reuse_ground:
            execution["reuse_ground_terms"] = True
        calculator=factories[route](full,region,reference,
            reuse_identical_geometry=config["provider"].get("reuse_identical_geometry",False),
            **execution)
        observer=FieldRecorder(calculator.embedded_selected_backend)
        calculator.embedded_selected_backend=observer
        ground_by_ids = ({tuple(qm["atom_ids"]): calculator.isolated_ground_backend}
                         if reuse_ground else {})
        def ground_for(ids):
            from .pbe0_tda import PySCFPBE0GroundBackend
            key = tuple(ids)
            if key not in ground_by_ids:
                ground_by_ids[key] = PySCFPBE0GroundBackend(
                    reference.settings, reuse_identical_geometry=True)
            return ground_by_ids[key]
        if "interaction_delta" in config:
            from .interaction_delta import CorrectedCalculator, CrossInteractionDelta
            from .interaction_delta_config import (
                CORRECTED_GAP_HAMILTONIAN,
                validate_interaction_delta,
            )
            from .pbe0_tda import PySCFPBE0GroundBackend
            if config["hamiltonian"] != CORRECTED_GAP_HAMILTONIAN:
                raise ValueError("Interaction correction requires its distinct Hamiltonian")
            delta_config = validate_interaction_delta(
                config["interaction_delta"], qm, env, state["method"])
            high = None if reuse_ground else PySCFPBE0GroundBackend(reference.settings)
            # Explicit neutral-singlet conditions were validated for all three
            # fragments. Their quantum settings match the anchor ground method.
            core_ids = delta_config["core_region"]["atom_ids"]
            fragment_ids = delta_config["fragment_region"]["atom_ids"]
            providers = ((ground_for(core_ids + fragment_ids), ground_for(core_ids),
                          ground_for(fragment_ids)) if reuse_ground else (high, high, high))
            delta = CrossInteractionDelta(*providers, region, region, region,
                core_atom_ids=core_ids, fragment_atom_ids=fragment_ids)
            calculator = CorrectedCalculator(calculator, delta)
        if fragment_config is not None:
            from .fragment_response import (
                ClusterGapDelta,
                OneBodyAlignedPairSet,
                PairInteractionSet,
            )
            from .interaction_delta import CorrectedCalculator, CrossInteractionDelta
            from .pbe0_tda import (
                PySCFPBE0ElectrostaticGroundBackend,
                PySCFPBE0GroundBackend,
                PySCFPBE0TDABackend,
                PySCFPBE0TDAElectrostaticBackend,
            )
            high = None if reuse_ground else PySCFPBE0GroundBackend(reference.settings)
            pairs = []
            for spec in fragment_config["pairs"]:
                core_ids, fragment_ids = spec["core_region"]["atom_ids"], spec["fragment_region"]["atom_ids"]
                providers = ((ground_for(core_ids + fragment_ids), ground_for(core_ids),
                              ground_for(fragment_ids)) if reuse_ground else (high, high, high))
                pairs.append(CrossInteractionDelta(*providers, region, region, region,
                    core_atom_ids=core_ids, fragment_atom_ids=fragment_ids))
            pairs = tuple(pairs)
            choices = fragment_config["alignments"]
            pair_set = (OneBodyAlignedPairSet(pairs, alignments=tuple(
                (a["pair_index"], a["side"]) for a in choices)) if choices else PairInteractionSet(pairs))
            gap = fragment_config["cluster_gap"]
            def selected(name, factory, embedded):
                spec = gap[name + "_provider"]
                ref = load_tda_reference(spec["state"]["method"], spec["state"]["reference"])
                execution = {"reuse_identical_geometry": spec.get("reuse_identical_geometry", False),
                             "matrix_assembly": spec.get("matrix_assembly", "operator")}
                args = (ref.settings, ref, ref.charges) if embedded else (ref.settings, ref)
                return FieldRecorder(factory(*args, **execution))
            pair = selected("pair", PySCFPBE0TDABackend, False)
            field = selected("field", PySCFPBE0TDAElectrostaticBackend, True)
            pair_ground = (ground_for(gap["core_region"]["atom_ids"] + gap["fragment_region"]["atom_ids"])
                           if reuse_ground else PySCFPBE0GroundBackend(pair.reference.settings))
            ground_options = {"reuse_identical_geometry": True} if reuse_ground else {}
            delta = ClusterGapDelta(pair_ground, pair,
                PySCFPBE0ElectrostaticGroundBackend(field.reference.settings, field.charges,
                                                    **ground_options), field,
                core_atom_ids=gap["core_region"]["atom_ids"],
                fragment_atom_ids=gap["fragment_region"]["atom_ids"])
            if config.get("reduce_fragment_terms", False):
                from .fragment_response import ReducedFragmentResponseCalculator
                calculator = ReducedFragmentResponseCalculator(calculator, pair_set, delta)
            else:
                calculator = CorrectedCalculator(CorrectedCalculator(calculator, pair_set), delta)
            observer = StateRecordGroup(observer, {"cluster_pair": pair, "cluster_field": field})

    _root_call(comm,build)
    return GapAnchorExternalCoupling(lmp,region,calculator,comm=comm,
        type_symbols=types,state_observer=observer)


def run(config_path, output, *, restart=None):
    from .cli import _run
    from .gap_anchor_config import load_gap_anchor_config
    return _run(config_path,output,restart=restart,config_loader=load_gap_anchor_config,
                setup_pair=_zero_pair,make_coupling=_gap_anchor_coupling)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--restart",type=Path)
    args=parser.parse_args(argv)
    try:
        run(args.config,args.output,restart=args.restart)
    except Exception as exc:
        print(f"Gap-anchor QM/DPA failed: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
