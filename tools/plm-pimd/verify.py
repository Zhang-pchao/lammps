#!/usr/bin/env python3
"""Run a short model/PIMD/PLUMED functional check after sourcing env.sh."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def quoted(path):
    text = str(path.resolve(strict=True))
    if any(c in text for c in ('"', "$", "\n", "\r")):
        raise ValueError("The input path contains LAMMPS substitution characters")
    return '"' + text + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "plumed-input", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--type-map", nargs="+", required=True)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timestep", type=float, default=0.0001)
    parser.add_argument("--contraction", type=float, default=0.5)
    parser.add_argument("--quantum-region", choices=("all", "sphere"), default="all")
    parser.add_argument("--center", type=float, nargs=3)
    parser.add_argument("--core", type=float)
    parser.add_argument("--outer", type=float)
    parser.add_argument("--mpi-command", default="mpirun --bind-to none")
    args = parser.parse_args()
    if args.beads < 1 or args.beads & (args.beads - 1):
        parser.error("--beads must be a positive power of two")
    if (
        min(args.steps, args.gpus) < 1
        or args.timestep <= 0
        or not 0 <= args.contraction <= 1
    ):
        parser.error(
            "Use positive steps, GPU count and timestep, and contraction in [0,1]"
        )
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name) for name in args.type_map):
        parser.error("Invalid element labels")
    data = args.data.read_text()
    count = re.search(r"^\s*(\d+)\s+atoms\s*$", data, re.MULTILINE)
    if count is None:
        parser.error("--data must be an atomic LAMMPS data file")
    limits = [
        re.search(
            r"^\s*(\S+)\s+(\S+)\s+" + axis + r"lo\s+" + axis + r"hi\s*$",
            data,
            re.MULTILINE,
        )
        for axis in "xyz"
    ]
    if not all(limits):
        parser.error("The input cell must be orthorhombic")
    center = args.center or [0.5 * (float(m[1]) + float(m[2])) for m in limits]
    half_box = min(float(m[2]) - float(m[1]) for m in limits) / 2
    core = args.core if args.core is not None else half_box / 4
    outer = args.outer if args.outer is not None else half_box / 2
    if not 0 <= core < outer < half_box:
        parser.error("Require 0 <= core < outer < each periodic half-box")
    driver = Path(os.environ["PLM_PIMD_DRIVER"]).resolve(strict=True)
    plugin = Path(os.environ["PLM_PIMD_PLUGIN"]).resolve(strict=True)
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / "plumed.dat").write_text(args.plumed_input.read_text())
    quantum = " quantum all" if args.quantum_region == "all" else ""
    text = f"""variable bead world {" ".join(map(str, range(args.beads)))}
plugin load {quoted(plugin)}
units metal
atom_style atomic
boundary p p p
atom_modify map array sort 0 0
processors 1 1 1
read_data {quoted(args.data)}
variable shift equal 0.001*(v_bead-0.5*({args.beads}-1))
displace_atoms all move ${{shift}} 0 0 units box
pair_style deepmd/kk {quoted(args.model)} pimd_reuse pi full
pair_coeff * * {" ".join(args.type_map)}
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
timestep {args.timestep}
velocity all set 0 0 0 units box
fix pi all pimd/adaptive temp 300 center {" ".join(map(str, center))} core {core} outer {outer} initialize auxiliary{quantum} thermostat pile 0.1 19771
fix pl all plumed plumedfile plumed.dat outfile plumed.log path_integral cv_mean pimd_fix pi path_contraction {args.contraction}
thermo 1
thermo_style custom step pe f_pi[1] f_pi[2] f_pi[4] f_pi[5]
thermo_modify norm no
run {args.steps}
write_dump all custom endpoint-${{bead}}.dump id x y z fx fy fz f_pi[10] modify sort id format float %.17g
"""
    (work / "in.lammps").write_text(text)
    command = [
        *shlex.split(args.mpi_command),
        "-np",
        str(args.beads),
        str(driver),
        "-k",
        "on",
        "g",
        str(args.gpus),
        "-sf",
        "kk",
        "-partition",
        str(args.beads) + "x1",
        "-in",
        "in.lammps",
        "-log",
        "log",
        "-screen",
        "screen",
        "-nocite",
    ]
    with (work / "stdout.log").open("w") as stream:
        result = subprocess.run(
            command, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    receipt = {
        "functional_pass": False,
        "command": command,
        "exit_code": result.returncode,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "plumed_sha256": hashlib.sha256(args.plumed_input.read_bytes()).hexdigest(),
        "beads": args.beads,
        "contraction": args.contraction,
        "quantum_region": args.quantum_region,
        "steps": args.steps,
    }
    (work / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("LAMMPS failed. Inspect " + str(work / "stdout.log"))
    import numpy as np

    for bead in range(args.beads):
        values = np.loadtxt(work / f"endpoint-{bead}.dump", skiprows=9, ndmin=2)
        if values.shape != (int(count[1]), 8) or not np.isfinite(values).all():
            raise RuntimeError("Invalid saved atom state")
        if args.quantum_region == "all" and not np.allclose(values[:, 7], 1):
            raise RuntimeError("The complete-quantum amplitude check failed")
    receipt["functional_pass"] = True
    (work / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PLM_PIMD_FUNCTIONAL_PASS", work)


if __name__ == "__main__":
    main()
