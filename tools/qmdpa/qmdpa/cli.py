# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Run fixed-cell QM/DPA dynamics and save only accepted states."""

import argparse
import json
import sys
import time
from pathlib import Path

from .backends import DeepMDBackend, PySCFBackend
from .config import file_digest, load_config
from .core import SubtractiveCalculator
from .lammps import ExternalCoupling
from .region import FixedRegion
from .runtime import _compare_restart, _frame, _quoted, _root_call, _validate_restart


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
    from lammps import lammps

    initialization_started = time.perf_counter()
    timings = {
        "initialization_wall_seconds": None,
        "run_wall_seconds": 0.0,
        "checkpoint_wall_seconds": 0.0,
    }
    coupling = None

    def metrics():
        return {
            "schema_version": 1,
            "scope": "owner_rank_host_wall",
            "clock": "time.perf_counter",
            **timings,
            "coupling": None if coupling is None else coupling.statistics(),
        }

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    comm = lmp.get_mpi_comm()
    if comm is None and lmp.extract_setting("world_size") != 1:
        lmp.close()
        raise RuntimeError("An MPI run requires mpi4py linked to the same MPI implementation")
    output_created = False
    try:
        config = _root_call(comm, lambda: config_loader(config_path))
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
        lmp.commands_list(
            [
                "neighbor 2.0 bin",
                "neigh_modify every 1 delay 0 check yes",
                f"timestep {config['timestep_fs'] / 1000:.17g}",
                "fix qmdpa_integrator all nve",
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
        coupling = make_coupling(lmp, comm, config)
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
        initial = _frame(lmp, comm)
        if previous is not None:
            _root_call(comm, lambda: _compare_restart(initial, previous))
        _root_call(comm, lambda: (output / "frames.jsonl").write_text(json.dumps(initial) + "\n"))
        remaining = config["steps"]
        while remaining:
            count = min(remaining, config["sample_interval"])
            _timed_call(timings, "run_wall_seconds", coupling.run, count)
            frame = _frame(lmp, comm)
            step = int(lmp.extract_global("ntimestep"))
            restart_path = output / f"restart-{step:012d}.bin"
            _timed_call(
                timings,
                "checkpoint_wall_seconds",
                lmp.command,
                f"write_restart {_quoted(restart_path)}",
            )

            def save_accepted():
                checkpoint = {
                    "physics_sha256": config["physics_sha256"],
                    "restart_file": restart_path.name,
                    "restart_sha256": file_digest(restart_path),
                    "frame": frame,
                }
                checkpoint_path = output / f"checkpoint-{step:012d}.json"
                temporary = checkpoint_path.with_suffix(".json.part")
                temporary.write_text(json.dumps(checkpoint, indent=2) + "\n")
                temporary.replace(checkpoint_path)
                with (output / "frames.jsonl").open("a") as stream:
                    stream.write(json.dumps(frame) + "\n")

            _timed_call(timings, "checkpoint_wall_seconds", _root_call, comm, save_accepted)
            remaining -= count
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
        if output_created:
            try:
                _root_call(
                    comm,
                    lambda: (output / "failure.json").write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "error": failure_message,
                                "recovery": "Use the last accepted checkpoint; discard in-memory state.",
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
        lmp.close()


def _mechanical_pair(lmp, config):
    condition = config.get("full_charge_spin")
    suffix = "" if condition is None else " charge_spin " + " ".join(map(str, condition))
    lmp.command(f"pair_style deepmd {_quoted(config['model'])}{suffix}")
    lmp.command("pair_coeff * * " + " ".join(config["type_symbols"]))


def _mechanical_coupling(lmp, comm, config):
    calculator = None

    def make_calculator():
        nonlocal calculator
        calculator = SubtractiveCalculator(
            DeepMDBackend(config["model"], charge_spin=config.get("region_charge_spin")),
            PySCFBackend(**config["qm"]),
        )

    _root_call(comm, make_calculator)
    region = FixedRegion(tuple(config["region"]["atom_ids"]), tuple(config["region"]["symbols"]))
    coupling = ExternalCoupling(
        lmp, region, calculator, comm=comm, type_symbols=tuple(config["type_symbols"])
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
    args = parser.parse_args(argv)
    try:
        run(args.config, args.output, restart=args.restart)
    except Exception as exc:
        print(f"QM/DPA failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
