# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Write offline region proposals for a finite JSON geometry or frames."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

_SELECTOR_FLAGS = (
    "centers",
    "oxygen_count",
    "max_oh_distance",
    "shared_distance_gap",
    "score_tolerance",
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--settings", type=Path, help="Version-1 offline region settings JSON")
    cli.add_argument("--centers", type=int, nargs="+")
    cli.add_argument("--oxygen-count", type=int)
    cli.add_argument("--max-oh-distance", type=float)
    cli.add_argument("--shared-distance-gap", type=float)
    cli.add_argument("--score-tolerance", type=float)
    return cli


def _geometry(frame):
    from .core import Geometry

    if not isinstance(frame, dict):
        raise TypeError("Each frame must be a JSON object")
    if frame.get("units") != "angstrom" or frame.get("boundary") != "nonperiodic":
        raise ValueError("Each frame requires units=angstrom and boundary=nonperiodic")
    if frame.get("cell") is not None or frame.get("periodic", False) is not False:
        raise ValueError("Periodic cells and image reconstruction are unsupported")
    return Geometry(tuple(frame["atom_ids"]), tuple(frame["symbols"]), frame["positions"])


def _configured_selector(args):
    from .selection_config import read_json, selection_from_settings

    supplied = {
        name: getattr(args, name) for name in _SELECTOR_FLAGS if getattr(args, name) is not None
    }
    if args.settings is not None:
        if supplied:
            raise ValueError("--settings cannot be combined with any selector flags")
        raw = args.settings.read_bytes()
        selector, resolved = selection_from_settings(read_json(raw))
        provenance = {"origin": "file", "name": args.settings.name, "hash_basis": "file-bytes"}
    else:
        if args.centers is None or args.oxygen_count is None:
            raise ValueError("Use --settings, or supply both --centers and --oxygen-count")
        selector, resolved = selection_from_settings(
            {"schema_version": 1, "mode": "offline", "method": "water-nearest", **supplied}
        )
        raw = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        provenance = {"origin": "legacy-cli", "name": None, "hash_basis": "canonical-json"}
    return selector, {**provenance, "sha256": hashlib.sha256(raw).hexdigest(), "resolved": resolved}


def report(args):
    """Build a report; invalid frames are fatal, ambiguous topology is explicit."""
    from .selection_config import read_json

    selector, settings = _configured_selector(args)
    raw = args.input.read_bytes()
    document = read_json(raw)
    if not isinstance(document, dict):
        raise TypeError("The input must be a geometry object or an object with frames")
    frames = document.get("frames", [document])
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a nonempty list")
    results = []
    identity = None
    for index, frame in enumerate(frames):
        geometry = _geometry(frame)
        current_identity = tuple(sorted(zip(geometry.atom_ids, geometry.symbols)))
        if identity is not None and current_identity != identity:
            raise ValueError("All frames must preserve atom IDs and their element identities")
        identity = current_identity
        results.append({"frame_index": index, **dataclasses.asdict(selector.select(geometry))})
    return {
        "schema_version": 1,
        "purpose": "Offline geometric proposals only; no adaptive energy or dynamics qualification.",
        "structural_labels": "Groups and OH labels do not assign electronic charges or spin states.",
        "source": {"name": args.input.name, "sha256": hashlib.sha256(raw).hexdigest()},
        "settings": settings,
        "selector": dataclasses.asdict(selector),
        "counts": {
            "frames": len(results),
            "selected": sum(item["status"] == "selected" for item in results),
            "unresolved": sum(item["status"] == "unresolved" for item in results),
        },
        "frames": results,
    }


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.output.exists():
        cli.error("The output path already exists; use a new report path")
    try:
        result = report(args)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        cli.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
