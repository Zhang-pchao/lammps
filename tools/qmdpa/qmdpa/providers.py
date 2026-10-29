# LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
# https://www.lammps.org/, Sandia National Laboratories
# SPDX-License-Identifier: GPL-2.0-only
# Author: Pengchao Zhang <pczhang1997@gmail.com>
"""Explicit supported QM combinations and lazy construction.

A contract records what a route implements. Numerical qualification still
belongs to a source/runtime-specific test record, never to this registry.
"""

from __future__ import annotations


def _tda_types(method, *, engine="pyscf"):
    """Resolve only the explicitly requested method, without importing PySCF."""
    if engine == "gpu4pyscf" and method == "rhf-tda":
        from .gpu_excited import (
            GPU4PySCFTDABackend,
            GPU4PySCFTDAElectrostaticBackend,
            GPUTDAReference,
            GPUTDASettings,
        )

        return (GPUTDASettings, GPUTDAReference, GPU4PySCFTDABackend,
                GPU4PySCFTDAElectrostaticBackend)
    if engine != "pyscf":
        raise ValueError("Unsupported selected-state engine/method combination")
    if method == "rhf-tda":
        from .excited import (
            PySCFTDABackend,
            PySCFTDAElectrostaticBackend,
            TDAReference,
            TDASettings,
        )
        return TDASettings, TDAReference, PySCFTDABackend, PySCFTDAElectrostaticBackend
    if method == "pbe0-tda":
        from .pbe0_tda import (
            PBE0TDAReference,
            PBE0TDASettings,
            PySCFPBE0TDABackend,
            PySCFPBE0TDAElectrostaticBackend,
        )
        return (PBE0TDASettings, PBE0TDAReference, PySCFPBE0TDABackend,
                PySCFPBE0TDAElectrostaticBackend)
    raise ValueError("Unsupported selected-state method")


def load_tda_reference(method, record, *, engine="pyscf"):
    """Require the declared method's exact reference schema and runtime pins."""
    return _tda_types(method, engine=engine)[1].from_record(record)


def _reference_identity(reference):
    identities = {
        "cpu-rhf-tda-singlet-fixed-reference": ("pyscf", "rhf-tda"),
        "cpu-pbe0-tda-singlet-fixed-reference": ("pyscf", "pbe0-tda"),
        "gpu-rhf-tda-singlet-fixed-reference": ("gpu4pyscf", "rhf-tda"),
    }
    identity = identities.get(getattr(reference, "_method_id", None))
    if identity is None or type(reference) is not _tda_types(
        identity[1], engine=identity[0]
    )[1]:
        raise TypeError("State recording requires a supported pinned TDAReference")
    return identity


def reference_method(reference):
    """Resolve the exact method; subclasses cannot change its meaning."""
    return _reference_identity(reference)[1]


def reference_engine(reference):
    """Return the independently pinned execution engine of this reference."""
    return _reference_identity(reference)[0]


def tda_settings(qm, method, reference):
    """Validate method settings identically for configuration and construction."""
    from dataclasses import fields

    settings_type = _tda_types(method, engine=reference_engine(reference))[0]
    label = "RHF/TDA" if method == "rhf-tda" else "PBE0/TDA"
    allowed = {item.name for item in fields(settings_type)}
    if not isinstance(qm, dict) or set(qm).difference(allowed):
        raise ValueError(f"qm contains unsupported {label} configuration keys")
    if method == "pbe0-tda" and not {
        "tda_solver", "grid_level", "grid_prune", "grid_response"
    }.issubset(qm):
        raise ValueError("PBE0/TDA qm requires explicit solver and grid response settings")
    settings = settings_type(**qm)
    if settings != reference.settings:
        raise ValueError(f"qm settings differ from the pinned {label} reference")
    return settings


def validate_provider(spec=None, *, electrostatic=False):
    """Normalize the supported combination before importing a quantum engine."""
    if type(electrostatic) is not bool:
        raise TypeError("electrostatic must be true or false")
    if spec is None:
        spec = {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}}
    required = {"engine", "device", "state"}
    if (not isinstance(spec, dict) or not required.issubset(spec)
            or set(spec) - required - {"reuse_identical_geometry", "matrix_assembly"}):
        raise ValueError("provider requires engine, device and state; optional execution controls are reuse_identical_geometry and matrix_assembly")
    reuse = spec.get("reuse_identical_geometry", False)
    if type(reuse) is not bool:
        raise ValueError("provider.reuse_identical_geometry must be true or false")
    state = spec["state"]
    if not isinstance(state, dict) or "method" not in state:
        raise ValueError("provider.state requires the supported method declaration")
    if "matrix_assembly" in spec and (
            state["method"] != "pbe0-tda"
            or not isinstance(spec["matrix_assembly"], str)
            or spec["matrix_assembly"] not in ("operator", "official-get-ab-896")):
        raise ValueError("provider.matrix_assembly requires PBE0/TDA and operator or official-get-ab-896")
    if state["method"] == "uks-hirshfeld-cdft":
        from .cdft import validate_reference

        if (set(spec) != required or set(state) != {"method", "reference"}
                or (spec["engine"], spec["device"]) != ("pyscf", "cpu")):
            raise ValueError("cDFT requires explicit CPU PySCF and a fixed reference, without reuse controls")
        return {"engine": "pyscf", "device": "cpu",
                "state": {"method": state["method"], "reference": validate_reference(
                    state["reference"], electrostatic=electrostatic)}}
    combination = (spec["engine"], spec["device"], state["method"])
    supported = (("pyscf", "cpu", "ground"), ("pyscf", "cpu", "rhf-tda"),
                 ("pyscf", "cpu", "pbe0-tda"), ("abacus", "cpu", "ground"),
                 ("gpu4pyscf", "cuda", "ground"),
                 ("gpu4pyscf", "cuda", "rhf-tda"))
    if combination not in supported or (spec["engine"] == "abacus" and electrostatic):
        field = "fixed-charge" if electrostatic else "isolated"
        raise ValueError(
            f"No implemented provider combination for {combination!r} with {field} QM. "
            "Use pyscf/cpu ground or dense rhf-tda/pbe0-tda, or gpu4pyscf/cuda "
            "ground or dense rhf-tda; abacus/cpu ground supports mechanical embedding only."
        )
    if state["method"] == "ground":
        if "reuse_identical_geometry" in spec:
            raise ValueError("Exact geometry reuse is available only for RHF/TDA or PBE0/TDA selected states")
        if set(state) != {"method"}:
            raise ValueError("Ground provider.state requires only method")
        return {"engine": spec["engine"], "device": spec["device"],
                "state": {"method": "ground"}}
    if set(state) != {"method", "reference"}:
        raise ValueError("TDA provider.state requires method and a complete inline reference")
    reference = load_tda_reference(state["method"], state["reference"], engine=spec["engine"])
    if reference.settings.tda_solver != "dense":
        raise ValueError("Native TDA requires an explicitly dense reference")
    if (reference.charges is not None) != electrostatic:
        raise ValueError("TDA reference environment differs from the requested provider")
    if reference.charges is not None and reference.charges.radii_angstrom is not None:
        raise ValueError(
            "Native TDA supports point charges only; Gaussian radii are unsupported"
        )
    return {
        "engine": spec["engine"],
        "device": spec["device"],
        "state": {"method": state["method"], "reference": reference.to_record()},
        **({"reuse_identical_geometry": reuse} if "reuse_identical_geometry" in spec else {}),
        **({"matrix_assembly": spec["matrix_assembly"]} if "matrix_assembly" in spec else {}),
    }


def provider_contract(spec=None, *, electrostatic=False, region_boundary="isolated", qm=None):
    """Describe the actual QM derivative scope, separately from full MD scope."""
    normalized = validate_provider(spec, electrostatic=electrostatic)
    if region_boundary not in ("isolated", "periodic-same-cell"):
        raise ValueError("Unknown region boundary")
    periodic_region = region_boundary == "periodic-same-cell"
    if periodic_region and (normalized["engine"] != "abacus" or electrostatic):
        raise ValueError("Periodic QM is supported only by ABACUS ground mechanics")
    state = normalized["state"]
    excited = state["method"] != "ground"
    energy = {"ground": "total-electronic-and-nuclear",
              "rhf-tda": "total-rhf-plus-tda-excitation",
              "pbe0-tda": "total-pbe0-plus-tda-excitation",
              "uks-hirshfeld-cdft": "physical-uks-constrained-density"}[state["method"]]
    if excited:
        state = {"method": state["method"], "reference_sha256": state["reference"]["sha256"]}
    abacus_energy = "pseudopotential-pbe-total"
    abacus_model = "pw-pbe-gamma-fixed-occupation"
    electronic_controls = {}
    if normalized["engine"] == "abacus" and qm is not None:
        keys = ("kmesh", "smearing_method", "smearing_sigma", "nbands", "pw_diag_thr_floor")
        electronic_controls = {key: qm[key] for key in keys if key in qm}
        if qm.get("kmesh", [1, 1, 1]) != [1, 1, 1]:
            abacus_model = "pw-pbe-kmesh-fixed-occupation"
        if qm.get("smearing_method") == "fd":
            abacus_energy = "pseudopotential-pbe-electronic-free-energy"
            abacus_model = "pw-pbe-fermi-dirac"
    return {
        "schema_version": 1,
        "engine": normalized["engine"],
        "device": normalized["device"],
        "state": state,
        "qm_boundary": ("periodic-same-cell" if periodic_region else "finite-periodic-supercell"
                        if normalized["engine"] == "abacus" else "isolated-molecule"),
        "environment": "fixed-charge" if electrostatic else "none",
        "energy": abacus_energy if normalized["engine"] == "abacus" else energy,
        "force_coordinates": ["qm", "environment"] if electrostatic else ["qm"],
        "stress": "unsupported",
        "history": "none",
        **({"execution": "cold-fileio-serial",
            "coordinate_mapping": "periodic-wrap" if periodic_region else "arithmetic-centroid",
            "electronic_model": abacus_model,
            **({"electronic_controls": electronic_controls} if electronic_controls else {})}
           if normalized["engine"] == "abacus" else {}),
        **({"charge_model": "point"}
           if electrostatic and normalized["engine"] == "gpu4pyscf" else {}),
        **({"execution": "single-visible-cuda-device", "cpu_fallback": "forbidden"}
           if normalized["engine"] == "gpu4pyscf" else {}),
        **({"result_reuse": "last-successful-exact-input"}
           if normalized.get("reuse_identical_geometry", False) else {}),
        **({"matrix_assembly": normalized["matrix_assembly"]}
           if "matrix_assembly" in normalized else {}),
        "qualification": "requires-matched-runtime-record",
    }


def _validate_reference_charges(reference, charges):
    from .embedding import FixedEmbeddingCharges

    if not isinstance(charges, FixedEmbeddingCharges):
        raise TypeError("charges must be an explicit FixedEmbeddingCharges object")
    expected = reference.charges
    if (
        expected is None
        or charges.atom_ids != expected.atom_ids
        or charges.charges_e.tolist() != expected.charges_e.tolist()
        or charges.source != expected.source
        or (charges.radii_angstrom is None) != (expected.radii_angstrom is None)
        or (
            charges.radii_angstrom is not None
            and charges.radii_angstrom.tolist() != expected.radii_angstrom.tolist()
        )
    ):
        raise ValueError("TDA fixed charges, radii or source differ from the pinned reference")


def validate_provider_regions(spec, qm_region, *, periodic, environment_region=None, charges=None,
                              region_boundary="isolated"):
    """Check fixed state identity while allowing every atom position to evolve.

    Regions have already passed the driver whole-molecule validation. Reference
    coordinates define AO state overlaps; they are not initial-position locks.
    """
    normalized = validate_provider(spec, electrostatic=environment_region is not None)
    if region_boundary not in ("isolated", "periodic-same-cell"):
        raise ValueError("Unknown region boundary")
    if region_boundary == "periodic-same-cell":
        if normalized["engine"] != "abacus" or periodic is not True:
            raise ValueError("Same-cell periodic QM requires ABACUS and a periodic full system")
    elif normalized["engine"] == "abacus" and periodic is not False:
        raise ValueError("Native finite-supercell ABACUS requires a nonperiodic full system")
    if normalized["engine"] == "gpu4pyscf":
        if periodic is not False and not (
            periodic is True
            and normalized["state"]["method"] == "ground"
            and environment_region is None
            and charges is None
        ):
            raise ValueError(
                "Native GPU4PySCF requires a nonperiodic full system except for "
                "ground mechanical embedding with a finite molecular QM region"
            )
        if environment_region is not None:
            from .embedding import FixedEmbeddingCharges

            if not isinstance(charges, FixedEmbeddingCharges):
                raise TypeError("GPU embedding requires explicit FixedEmbeddingCharges")
            if charges.radii_angstrom is not None:
                raise ValueError("GPU4PySCF embedding currently supports point charges only")
    if normalized["state"]["method"] == "ground":
        return
    if normalized["state"]["method"] == "uks-hirshfeld-cdft":
        reference = normalized["state"]["reference"]
        if periodic is not False or region_boundary != "isolated":
            raise ValueError("cDFT requires finite fixed molecular boundaries")
        for region, key in ((qm_region, "qm_geometry"), (environment_region, "environment_geometry")):
            expected = reference[key]
            if expected is None:
                if region is not None:
                    raise ValueError("cDFT environment presence changed")
            elif region is None or any(region[name] != expected[name] for name in ("atom_ids", "symbols")):
                raise ValueError("cDFT region IDs, symbols and order must match the reference")
        if charges is not None:
            actual = {"atom_ids": list(charges.atom_ids), "charges_e": charges.charges_e.tolist(),
                      "source": charges.source}
            if actual != reference["charges"] or charges.radii_angstrom is not None:
                raise ValueError("cDFT point charge identity changed")
        return
    reference = load_tda_reference(
        normalized["state"]["method"], normalized["state"]["reference"], engine=normalized["engine"])
    if periodic is not False:
        raise ValueError("Native TDA requires a nonperiodic full system")
    if (
        tuple(qm_region["atom_ids"]) != reference.qm_geometry.atom_ids
        or tuple(qm_region["symbols"]) != reference.qm_geometry.symbols
    ):
        raise ValueError("TDA QM region IDs, symbols and order must match the reference")
    if environment_region is not None:
        geometry = reference.environment_geometry
        if (
            tuple(environment_region["atom_ids"]) != geometry.atom_ids
            or tuple(environment_region["symbols"]) != geometry.symbols
        ):
            raise ValueError(
                "TDA environment region IDs, symbols and order must match the reference"
            )
        _validate_reference_charges(reference, charges)
    elif charges is not None:
        raise ValueError("An isolated TDA region cannot declare embedding charges")


def make_qm_provider(qm, *, provider=None, charges=None, directory=None):
    """Construct only a validated route; heavy engine imports remain optional."""
    normalized = validate_provider(provider, electrostatic=charges is not None)
    if not isinstance(qm, dict):
        raise TypeError("qm settings must be a mapping")
    if normalized["engine"] == "abacus":
        from pathlib import Path

        from .abacus import AbacusBackend, validate_mechanical_abacus_settings

        if directory is None:
            raise ValueError("ABACUS requires a fresh runtime directory")
        options = validate_mechanical_abacus_settings(qm, Path.cwd())
        options.pop("engine")
        return AbacusBackend(**options, directory=directory)
    if directory is not None:
        raise ValueError("A provider runtime directory is supported only for ABACUS")
    if normalized["engine"] == "gpu4pyscf" and normalized["state"]["method"] == "ground":
        from .gpu_backends import GPU4PySCFBackend, GPU4PySCFElectrostaticBackend

        if charges is None:
            return GPU4PySCFBackend(**qm)
        from .embedding import FixedEmbeddingCharges

        if not isinstance(charges, FixedEmbeddingCharges):
            raise TypeError("charges must be an explicit FixedEmbeddingCharges object")
        return GPU4PySCFElectrostaticBackend(charges, **qm)
    if normalized["state"]["method"] == "uks-hirshfeld-cdft":
        from .cdft import CDFTProvider

        return CDFTProvider(qm, normalized["state"]["reference"], charges)
    if normalized["state"]["method"] != "ground":
        from .state_records import RecordingTDAProvider

        method = normalized["state"]["method"]
        _, _, isolated_type, embedded_type = _tda_types(method, engine=normalized["engine"])
        reference = load_tda_reference(method, normalized["state"]["reference"],
                                       engine=normalized["engine"])
        settings = tda_settings(qm, method, reference)
        execution = ({"reuse_identical_geometry": normalized["reuse_identical_geometry"]}
                     if "reuse_identical_geometry" in normalized else {})
        if "matrix_assembly" in normalized:
            execution["matrix_assembly"] = normalized["matrix_assembly"]
        if charges is None:
            backend = isolated_type(settings, reference, **execution)
        else:
            _validate_reference_charges(reference, charges)
            backend = embedded_type(settings, reference, charges, **execution)
        return RecordingTDAProvider(backend)
    if charges is None:
        from .backends import PySCFBackend

        return PySCFBackend(**qm)
    from .embedding import FixedEmbeddingCharges, PySCFElectrostaticBackend

    if not isinstance(charges, FixedEmbeddingCharges):
        raise TypeError("charges must be an explicit FixedEmbeddingCharges object")
    return PySCFElectrostaticBackend(charges, **qm)
