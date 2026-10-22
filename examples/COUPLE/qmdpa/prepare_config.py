#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# Copyright (2026) Pengchao Zhang <pczhang1997@gmail.com>
# SPDX-License-Identifier: GPL-2.0-only
"""Bind the water regression template to an explicitly identified .pt2 model."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Target-GPU exported .pt2 archive")
    parser.add_argument("--sha256", required=True, help="Expected SHA256 of that archive")
    parser.add_argument("--output", required=True, type=Path, help="New configuration file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10, help="Additional integration steps")
    parser.add_argument(
        "--unconditioned", action="store_true",
        help="Omit ML charge/spin conditioning only for models that do not support it",
    )
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if model.suffix != ".pt2" or not model.is_file():
        parser.error("--model must be an existing .pt2 export; a raw .pt checkpoint is not accepted")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        parser.error("--sha256 must be a full SHA256 hex digest")
    if digest(model) != args.sha256.lower():
        parser.error("model SHA256 differs from --sha256")
    if args.threads <= 0 or args.steps < 0:
        parser.error("--threads must be positive and --steps must be nonnegative")
    if output.exists():
        parser.error("--output already exists; choose a new configuration file")
    source = Path(__file__).resolve().parent
    config = json.loads((source / "run.json").read_text())
    config.update({
        "model": str(model),
        "model_sha256": args.sha256.lower(),
        "data_file": str(source / "water-dimer.data"),
        "steps": args.steps,
    })
    config["qm"]["threads"] = args.threads
    if args.unconditioned:
        config["full_charge_spin"] = None
        config["region_charge_spin"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    print(f"Prepared {output}; no inference or dynamics has been performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
