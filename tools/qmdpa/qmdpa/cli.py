#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Run fixed-cell QM/DPA dynamics and save only accepted states."""

import argparse
import json
import sys
import time
from pathlib import Path

from .backends import DeepMDBackend
from .config import file_digest, load_config
from .core import SubtractiveCalculator
from .lammps import ExternalCoupling
from .providers import make_qm_provider
from .region import FixedRegion
from .runtime import _attach_state, _compare_restart, _frame, _quoted, _root_call, _validate_restart


def _write_manifest(output, config, *, world_size, restart):
    """Record JSON-safe provenance for a fresh run or accepted checkpoint."""
    manifest = {
        "configuration": config,
        "world_size": int(world_size),
        "restart_from": None if restart is None else str(Path(restart).resolve()),
        "status": "started",
        "pressure": "not qualified",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _timed_call(timings, name, function, *args):
    """Accumulate host wall time for a call, including a failed call."""
    started = time.perf_counter()
    try:
        return function(*args)
    finally:
        timings[name] += time.perf_counter() - started


def _run(config_path, output, *, restart, config_loader, setup_pair, make_coupling):
    """Run additional steps; each successful sample has a matching restart pair."""
    if restart is not None:
        try:
            from mpi4py import MPI
        except ImportError:
            # Keep the existing serial-without-mpi4py path and its native check.
            pass
        else:
            from .runtime import preflight_restart

            preflight_restart(
                config_path, restart, config_loader=config_loader, comm=MPI.COMM_WORLD,
                output=output
            )
    from lammps import lammps

    from .execution import native_arguments, register_native_cleanup, verify_native

    try:
        from mpi4py import MPI
    except ImportError:
        launch_comm = None
    else:
        launch_comm = MPI.COMM_WORLD
    initialization_started = time.perf_counter()
    prepared_config = _root_call(launch_comm, lambda: config_loader(config_path))

    timings = {
        "initialization_wall_seconds": None,
        "run_wall_seconds": 0.0,
        "checkpoint_wall_seconds": 0.0,
    }
    from .dynamics import capture_thermostat, integrator_command

    coupling = None
    bias = None
    provider_resources = None

    def metrics():
        return {
            "schema_version": 1,
            "scope": "owner_rank_host_wall",
            "clock": "time.perf_counter",
            **timings,
            "coupling": None if coupling is None else coupling.statistics(),
            **({"qm_provider": provider_resources.statistics()}
               if provider_resources is not None else {}),
        }

    lmp = lammps(cmdargs=native_arguments(prepared_config))
    register_native_cleanup(lmp, prepared_config)
    comm = lmp.get_mpi_comm()
    if comm is None and lmp.extract_setting("world_size") != 1:
        lmp.close()
        raise RuntimeError("An MPI run requires mpi4py linked to the same MPI implementation")
    output_created = False
    try:
        config = prepared_config
        output = Path(output).resolve()
        _root_call(comm, lambda: output.mkdir(parents=True, exist_ok=False))
        output_created = True
        previous = (
            None
            if restart is None
            else _root_call(comm, lambda: _validate_restart(restart, config))
        )
        lmp.commands_list(
            [
                "units metal",
                "atom_style atomic",
                "atom_modify map yes",
                "boundary " + ("p p p" if config["periodic"] else "f f f"),
            ]
        )
        if previous is None:
            lmp.command(f"read_data {_quoted(config['data_file'])}")
        else:
            lmp.command(f"read_restart {_quoted(previous['restart_path'])}")
        setup_pair(lmp, config)
        verify_native(lmp, config)
        lmp.commands_list(
            [
                "neighbor 2.0 bin",
                "neigh_modify every 1 delay 0 check yes",
                f"timestep {config['timestep_fs'] / 1000:.17g}",
                integrator_command(config),
                "thermo_style custom step temp pe ke etotal",
                "thermo_modify norm no",
                "thermo 1",
            ]
        )
        if previous is None and "temperature_K" in config:
            lmp.command(
                f"velocity all create {config['temperature_K']} {config['velocity_seed']} "
                "mom yes rot yes dist gaussian"
            )
        if config.get("provider", {}).get("engine") == "abacus":
            from .runtime import ProviderResources

            provider_resources = ProviderResources(output / "qm-backend")
            coupling = make_coupling(lmp, comm, config, provider_resources=provider_resources)
        else:
            coupling = make_coupling(lmp, comm, config)
        if "bias" in config:
            from .bias import NativeDistanceBias

            bias = NativeDistanceBias(lmp, config)
            coupling.native_bias = bias
        if "sampling" in config:
            from .sampling import NativePlumedBias

            bias = NativePlumedBias(lmp, config, output, comm, previous)
            coupling.native_bias = bias
        _root_call(
            comm,
            lambda: _write_manifest(
                output,
                config,
                world_size=lmp.extract_setting("world_size"),
                restart=restart,
            ),
        )
        timings["initialization_wall_seconds"] = time.perf_counter() - initialization_started
        _timed_call(timings, "run_wall_seconds", coupling.run, 0)
        initial = _frame(lmp, comm, periodic=config["periodic"])
        if bias is not None:
            bias.capture(initial, comm)
        capture_thermostat(lmp, initial, comm, config)
        _root_call(comm, lambda: _attach_state(initial, coupling, config))
        if previous is not None:
            _root_call(comm, lambda: _compare_restart(initial, previous, config))
        _root_call(comm, lambda: (output / "frames.jsonl").write_text(json.dumps(initial) + "\n"))
        remaining = config["steps"]
        while remaining:
            count = min(remaining, config["sample_interval"])
            if "sampling" in config:
                _timed_call(timings, "run_wall_seconds", bias.advance, coupling, count)
            else:
                _timed_call(timings, "run_wall_seconds", coupling.run, count)
            if "sampling" in config:
                _timed_call(timings, "run_wall_seconds", bias.refresh, coupling)
            frame = _frame(lmp, comm, periodic=config["periodic"])
            if bias is not None:
                bias.capture(frame, comm)
            capture_thermostat(lmp, frame, comm, config)
            _root_call(comm, lambda frame=frame: _attach_state(frame, coupling, config))
            step = int(lmp.extract_global("ntimestep"))
            restart_path = output / f"restart-{step:012d}.bin"
            _timed_call(
                timings,
                "checkpoint_wall_seconds",
                lmp.command,
                f"write_restart {_quoted(restart_path)}",
            )

            def save_accepted(restart_path=restart_path, frame=frame, step=step):
                checkpoint = {
                    "physics_sha256": config["physics_sha256"],
                    "restart_file": restart_path.name,
                    "restart_sha256": file_digest(restart_path),
                    "frame": frame,
                }
                if "sampling" in config:
                    bias.save_history(checkpoint, output)
                checkpoint_path = output / f"checkpoint-{step:012d}.json"
                temporary = checkpoint_path.with_suffix(".json.part")
                temporary.write_text(json.dumps(checkpoint, indent=2) + "\n")
                temporary.replace(checkpoint_path)
                with (output / "frames.jsonl").open("a") as stream:
                    stream.write(json.dumps(frame) + "\n")

            _timed_call(timings, "checkpoint_wall_seconds", _root_call, comm, save_accepted)
            remaining -= count
        if provider_resources is not None:
            cleanup_errors = _root_call(comm, provider_resources.close)
            if cleanup_errors:
                raise RuntimeError("QM provider cleanup failed: " + "; ".join(cleanup_errors))
        _root_call(
            comm,
            lambda: (output / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "steps": config["steps"],
                        "final_step": int(lmp.extract_global("ntimestep")),
                        "scientific_acceptance": "not assessed",
                        "metrics": metrics(),
                    },
                    indent=2,
                )
                + "\n"
            ),
        )
    except Exception as exc:
        if timings["initialization_wall_seconds"] is None:
            timings["initialization_wall_seconds"] = time.perf_counter() - initialization_started
        failure_message = f"{type(exc).__name__}: {exc}"
        if provider_resources is not None:
            # Close every registered owner resource while preserving the original
            # calculation exception. The returned errors enter failure metrics.
            _root_call(comm, provider_resources.close)
        if output_created:
            try:
                _root_call(
                    comm,
                    lambda: (output / "failure.json").write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "error": failure_message,
                                "recovery": (
                                    "Use the last accepted checkpoint; discard in-memory state."
                                ),
                                "metrics": metrics(),
                            },
                            indent=2,
                        )
                        + "\n"
                    ),
                )
            except RuntimeError:
                pass  # Preserve the original error if diagnostic I/O also failed.
        raise
    finally:
        try:
            if provider_resources is not None:
                provider_resources.close()
        finally:
            lmp.close()


def _mechanical_pair(lmp, config):
    condition = config.get("full_charge_spin")
    suffix = "" if condition is None else " charge_spin " + " ".join(map(str, condition))
    from .execution import pair_style

    lmp.command(f"pair_style {pair_style(config)} {_quoted(config['model'])}{suffix}")
    lmp.command("pair_coeff * * " + " ".join(config["type_symbols"]))


def _mechanical_coupling(lmp, comm, config, *, provider_resources=None):
    from .state_records import RecordingTDAProvider

    calculator = state_observer = None
    boundary = ("periodic-same-cell" if config["qm"].get("boundary") == "periodic-same-cell"
                else "isolated")

    def make_calculator():
        nonlocal calculator, state_observer
        if config["provider"]["engine"] == "abacus":
            if provider_resources is None:
                raise ValueError("ABACUS coupling requires an owned runtime directory")
            provider = provider_resources.register(make_qm_provider(
                config["qm"], provider=config["provider"], directory=provider_resources.directory
            ))
        else:
            provider = make_qm_provider(config["qm"], provider=config["provider"])
        if isinstance(provider, RecordingTDAProvider):
            state_observer = provider
        calculator = SubtractiveCalculator(
            DeepMDBackend(config["model"], charge_spin=config.get("region_charge_spin")), provider,
            region_boundary=boundary,
        )

    _root_call(comm, make_calculator)
    region = FixedRegion(tuple(config["region"]["atom_ids"]), tuple(config["region"]["symbols"]))
    coupling = ExternalCoupling(
        lmp,
        region,
        calculator,
        comm=comm,
        type_symbols=tuple(config["type_symbols"]),
        state_observer=state_observer,
        region_boundary=boundary,
    )
    return coupling


def run(config_path, output, *, restart=None):
    """Run the established mechanical Hamiltonian with accepted-state checkpoints."""
    return _run(
        config_path,
        output,
        restart=restart,
        config_loader=load_config,
        setup_pair=_mechanical_pair,
        make_coupling=_mechanical_coupling,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Run fixed-cell dynamics with real backends")
    runner.add_argument("config", type=Path)
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--restart", type=Path)
    validator = commands.add_parser("validate", help="Validate a configuration without evaluation")
    validator.add_argument("config", type=Path)
    validator.add_argument("--restart", type=Path, help="Also verify restart input integrity")
    for command, help_text in (
        ("prepare-state", "Prepare a reference for a supported selected state"),
        ("prepare-states", "Prepare several CPU RHF/PBE0 or GPU RHF references with one common solve"),
    ):
        state = commands.add_parser(command, help=help_text)
        state.add_argument("request", type=Path)
        state.add_argument("--output", required=True, type=Path)
    reference = commands.add_parser("prepare-water-reference",
                                    help="Prepare a fixed isolated-water buffer reference")
    reference.add_argument("request", type=Path)
    reference.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from .dispatch import describe_run, load_run_config
        from .dispatch import run as run_configured

        if args.command == "prepare-water-reference":
            from .water_reference import prepare_reference

            print(json.dumps(prepare_reference(args.request, args.output), indent=2))
        elif args.command in ("prepare-state", "prepare-states"):
            from .state_inputs import prepare_state, prepare_states

            prepare = prepare_state if args.command == "prepare-state" else prepare_states
            print(json.dumps(prepare(args.request, args.output), indent=2))
        elif args.command == "validate":
            config = load_run_config(args.config)
            description = describe_run(config)
            if args.restart is not None:
                _validate_restart(args.restart, config)
                description["restart"] = {
                    "path": str(args.restart.resolve()),
                    "status": "input-valid",
                    "native_recomputed_state": "not evaluated",
                }
            print(json.dumps(description, indent=2))
        else:
            run_configured(args.config, args.output, restart=args.restart)
    except Exception as exc:  # noqa: BLE001 - Preserve collective or top-level failure reporting.
        print(f"QM/DPA failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
