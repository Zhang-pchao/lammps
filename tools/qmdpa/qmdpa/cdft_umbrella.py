#!/usr/bin/env python3
# LAMMPS - Large-scale Atomic/Molecular Simulation
# SPDX-License-Identifier: GPL-2.0-only
"""Fixed umbrella over the same-frame constrained-state gap.

The cDFT provider already resolves both constrained surfaces at identical
coordinates and exposes their energy difference as the named scalar ``cdft_gap``
with a complete all-atom Jacobian. This module turns that observation into an
admitted static umbrella restraint through the existing official PLUMED EXTRACV
bridge; it owns no second mixer, no solver and no electronic state.

One configuration declares exactly one umbrella window. A multi-window umbrella
ensemble is realized as several configurations that share one frozen reference
and differ only in the window center, which keeps each run's bias a single
well-defined function of the observed gap.

The umbrella is a *sampling* device on the reactant reference ensemble. It does
not qualify any chemistry, does not open electronic OPES, and does not admit
dynamic electronic partition switching.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .dynamics import extend_identity

FORM = "cdft-gap-umbrella-v1"
GAP_NAME = "cdft_gap"


def enabled(config):
    return config.get("cdft_umbrella") is not None


def full_region(config):
    """The complete finite atom identity covered by the umbrella."""
    from .cdft_runtime import full_region as cdft_full_region

    return cdft_full_region(config)


def schema(config):
    """The single named scalar owned by the umbrella.

    The schema deliberately reuses the cDFT pair schema so that the umbrella
    value and the accepted pair record are provably the same quantity and not
    two independently computed gaps.
    """
    from .cdft_runtime import schema as cdft_schema

    return cdft_schema(config)


def bind(config, raw, directory):
    """Admit one static umbrella over the constrained gap.

    Admission is deliberately narrow: a finite CPU cDFT additive pair, the
    plain NVE surface it already owns, one declared umbrella window and the
    pinned PLUMED libraries. Every unsupported combination is rejected here
    rather than silently degraded at run time.
    """
    from .config import file_digest

    required = {"form", "center_eV", "kappa_per_eV", "kernel", "wrapper"}
    if not isinstance(raw, dict) or set(raw) != required or raw["form"] != FORM:
        raise ValueError("cdft umbrella requires form, center_eV, kappa_per_eV, kernel and wrapper")
    if config.get("provider", {}).get("state", {}).get("method") != "uks-hirshfeld-cdft":
        raise ValueError("The cDFT umbrella requires an explicit constrained-state provider")
    if config.get("hamiltonian") != "finite-electrostatic-additive":
        raise ValueError("The cDFT umbrella requires the additive electrostatic Hamiltonian")
    if config["periodic"]:
        raise ValueError("The cDFT umbrella requires a finite nonperiodic system")
    if config.get("provider", {}).get("device") != "cpu":
        raise ValueError("The cDFT umbrella requires actual CPU inference")
    for key in ("sampling", "bias", "thermostat", "execution"):
        if key in config:
            raise ValueError("The cDFT umbrella cannot be combined with another bias owner")
    if config["sample_interval"] != 1:
        raise ValueError("The cDFT umbrella requires every step to be sampled")

    center, kappa = raw["center_eV"], raw["kappa_per_eV"]
    if (type(center) not in (int, float) or not np.isfinite(center)
            or type(kappa) not in (int, float) or not np.isfinite(kappa) or kappa <= 0):
        raise ValueError("The cDFT umbrella requires a finite center and a positive kappa")

    paths = {}
    for name in ("kernel", "wrapper"):
        item = raw[name]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Umbrella libraries require path and SHA256")
        path = (Path(directory)/item["path"]).resolve()
        if not path.is_file() or file_digest(path) != item["sha256"]:
            raise ValueError("Umbrella PLUMED library checksum differs")
        paths[name] = str(path)

    value = {"form": FORM,
             "center_eV": float(center),
             "kappa_per_eV": float(kappa),
             "kernel": {"sha256": raw["kernel"]["sha256"]},
             "wrapper": {"sha256": raw["wrapper"]["sha256"]}}
    # The schema is part of the admitted identity: a different dependency set
    # would change the meaning of the biased energy.
    value["schema_sha256"] = schema(config).sha256
    extend_identity(config, "cdft_umbrella", value)
    config["cdft_umbrella_libraries"] = paths
    return config


def restraint(config):
    """The single PLUMED restraint parameter map for this configuration."""
    if not enabled(config):
        raise ValueError("No cDFT umbrella is configured")
    umb = config["cdft_umbrella"]
    return {GAP_NAME: {"center": umb["center_eV"],
                       "kappa": umb["kappa_per_eV"],
                       "unit": "eV"}}


def bias_energy(value, config):
    """Harmonic umbrella energy and gap derivative for the admitted window.

    The restraint is one half of kappa times the squared displacement of the
    gap from the window center. The derivative is with respect to the gap.
    """
    umb = config["cdft_umbrella"]
    delta = float(value)-float(umb["center_eV"])
    derivative = float(umb["kappa_per_eV"])*delta
    return 0.5*derivative*delta, derivative
