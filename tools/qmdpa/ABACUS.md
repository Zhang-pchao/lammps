# Ground mechanical coupling with ABACUS

The unified provider accepts engine abacus, device cpu, state method ground.
Use coupling.surface ground and coupling.embedding mechanical. The energy is
still full native DPA plus the regional ABACUS total minus regional DPA. This
route does not interpret DPA atomic energies as quantum-region energies.

The default finite-supercell run interface admits a neutral singlet PW/PBE, Gamma-point, fixed-
occupation ABACUS calculation in an explicit three-dimensional periodic vacuum
supercell for a finite molecular fragment. The
complete LAMMPS system is nonperiodic. The quantum fragment is the complete
fixed molecule set declared in region; no external field or charge embedding
is supplied. This finite-supercell approximation differs from an isolated
Gaussian-basis calculation. Cell/cutoff and chemical accuracy need their own
validation; different methods need not give identical total energies.

The top-level periodic flag controls the full LAMMPS system, not the ABACUS
electronic boundary. Gamma-point sampling does not remove periodic electronic
images. The low-level fragment is finite while the QM solve uses the declared
periodic supercell; this mixed boundary approximation does not qualify a
same-boundary periodic replacement, bulk/slab coupling or the isolated limit.

```json
"provider": {"engine": "abacus", "device": "cpu", "state": {"method": "ground"}},
"qm": {
  "executable": "software/abacus",
  "executable_sha256": "<binary SHA256>",
  "pseudopotentials": {
    "H": {"path": "potentials/H.upf", "sha256": "<H SHA256>"}
  },
  "supercell_angstrom": [12.0, 12.0, 12.0],
  "coordinate_offset_angstrom": [6.0, 6.0, 6.0],
  "coordinate_gauge": "arithmetic-centroid",
  "execution_mode": "fileio",
  "ecutwfc": 60.0,
  "scf_thr": 1e-12,
  "scf_nmax": 100,
  "ks_solver": "dav_subspace",
  "timeout_seconds": 600.0
}
```

This illustrates input structure, not an accuracy prescription. Every region
element needs its declared pseudopotential. File paths resolve relative to the
configuration. The optional qm.engine, if supplied, must also be abacus.
The executable, pseudopotentials, resolved controls and ASE/abacuslite versions
and Python source hashes enter restart identity. Use the identified compatible
Python interface; matching its version string alone is insufficient.

Cold FileIO creates a fresh worker, official ASE calculator and native process
for each evaluation. No preceding density/wavefunction or implicit fallback is
used. The runtime supplies its fresh qm-backend directory; directory is not a
physics input. The existing request deadline includes launch, calculation and
shutdown. It is separate from the scheduler allocation walltime.

The explicit coordinate mapping is X_i = R_i - mean(R) + offset. The mean is
an arithmetic mean over every supplied QM atom, not a mass-weighted or per-
molecule shift. With no external field, ABACUS's native equal-weight force
projection corresponds to this mapping's chain rule when the native force is
the derivative of its energy. The adapter applies no extra force projection.
Changing the coordinate mapping defines another finite-grid energy function;
the current unified route rejects fixed-offset and persistent socket modes.
Their historical standalone implementation remains available without new
numerical or native qualification.

Owner-rank providers are registered before later DPA/coupling construction.
Every exit closes them; a successful run is recorded only after checked shutdown.
A calculation error remains the primary failure if cleanup also fails. The
failure record includes cleanup diagnostics. Complete backend request time and
counts enter result metrics. LAMMPS MPI distributes atoms; this ABACUS executable
route is serial and is not a parallel or GPU electronic-structure implementation.

Selected-state, EE, adaptive and stress/NPT ABACUS routes are rejected.
Full-system periodicity requires the explicit same-cell route below. The other PySCF/GPU provider modes keep their own contracts.
Native and numerical qualification belongs to a matching source/runtime record,
not to this configuration example or a successful process-contract test.

## Declared execution dependencies

An executable checksum identifies that entrypoint only. For a launcher script,
it does not identify the program that the script starts; a native executable
hash also does not identify its shared libraries. Declare additional files with
optional qm.runtime_files, for example:

```json
"runtime_files": {
  "native_binary": {"path": "software/bin/abacus", "sha256": "<SHA256>"},
  "blas": {"path": "software/lib/libopenblas.so.0", "sha256": "<SHA256>"}
}
```

Paths resolve relative to the configuration and join the existing restart
physics identity. Names must be unique nonempty ASCII labels; duplicate resolved
paths, missing files and incorrect hashes are rejected. When supplied, the
complete declared set is checked at configuration/worker initialization, before
each evaluation, and again before accepting its result. An identity change
invalidates the provider and follows the existing collective failure/cleanup
path. Validation time is included in request time and its existing deadline.

An omitted runtime_files key preserves the legacy configuration shape and its
original scope. The option pins only the declared files; it does not discover
transitive dependencies, select loader paths or prove which library was loaded.
Keep the runtime's independent library/source manifests and actual native tests.
Do not change an old checkpoint identity after modifying a dependency. Shared
installations and previous qualification records must remain unchanged.

## Explicit same-cell periodic ground ME

Set `periodic: true` and add `qm.boundary: "periodic-same-cell"`,
`qm.coordinate_gauge: "periodic-wrap"`, and a zero coordinate offset.
`qm.supercell_angstrom` must equal the three orthorhombic LAMMPS box lengths.
The box must remain fixed and periodic in all three directions. Partial PBC,
triclinic boxes and changed cells are rejected. The low-level region and QM
provider receive the same cell and fixed atom IDs. The total is

    U = DPA_periodic(AB, h) - DPA_periodic(A, h) + ABACUS_periodic(A, h).

Native `pair_style deepmd` supplies the full-system term; `fix external` adds
only the difference. Individual atoms are wrapped into their equivalent cell
for ABACUS without centroid recentering. Forces map directly by atom ID. This
requires a neutral-singlet periodic fragment with sensible complete-molecule
membership; it does not make arbitrary covalent cuts or charged fragments valid.
The all-QM cancellation limit additionally requires identical low-level model,
conditions, types and periodic geometry for both low-level evaluations.

The implementation retains cold serial PW/PBE/Gamma/fixed-occupation settings,
explicit file identities, bounded requests, permanent failures and checked
cleanup. Boundary/cell/mapping settings enter restart identity. Existing finite
requests retain their prior schema and use finite low-level subtraction.
There is no automatic conversion between these two Hamiltonians.

Qualification is tied to frozen real-backend evidence. A tiny periodic molecular
cell checks E/F, images and native transport; it does not establish bulk/slab
accuracy, cutoff/k-point convergence, isolated-limit convergence, stress, NPT,
periodic EE, selected states or ABACUS GPU capability.
