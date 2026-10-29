# Practical float32 DPA4C execution

This optional profile includes finite, fixed-region, neutral-singlet CPU PySCF
ground ME with an explicitly selected native DPA pair. Its admission and
numerical qualification are independent of the existing strict profiles.

Add this object to an otherwise complete mechanical configuration:

```json
"execution": {
  "pair_style": "deepmd/kk",
  "numerical_profile": "float32-practical-v1"
}
```

Use `deepmd` for the ordinary native pair with the same numerical policy.
The driver verifies the actual pair and rejects silent fallback. Pair choice
is part of restart identity; switching implementations requires a new run.

The model must be a checksummed official DPA4C `.pt2` archive with float32
descriptor and fitting network, no AMP, and `graph` or `dpa4c_canonical`
inputs. CUDA architecture, Torch, DeePMD, compiler and native library versions
belong to its deployment manifest. An archive from one GPU architecture is
not automatically deployable on another. DPA4C denotes the compact model;
separate table compression and Kokkos are separate optimizations.

The initial profile covers NVE without a bias or an NVT thermostat.
It does not qualify periodic systems, charged models, selected states,
adaptive partitions, GPU QM or stress. Those existing routes retain their
own independent contracts.

## Native Kokkos setup

Build official LAMMPS KOKKOS with CUDA, the target architecture, MPI,
EXTRA-FIX and the official DeePMD built-in adapter. The companion recipe
`environment/build-kokkos.sh` uses a fresh out-of-source build and install.
It does not modify a shared environment or implement a second DPA pair.

The driver requests one GPU per node-visible allocation and full neighbors
with Newton pair off. GPU-aware MPI is explicitly off. Preserve scheduler
GPU visibility; two local ranks may share the one allocated GPU. This scope
does not include multi-node or multi-GPU scaling. Atom IDs are mapped each
callback. CPU QM requires host coordinates and force exchange; the complete
hybrid trajectory is not entirely resident on the GPU.

## Precision and restart

The prospective practical screen uses absolute discrepancies of 1e-5 eV/atom,
1e-4 eV for fixed-composition relative energies, and force-component RMS/max
of 1e-4/1e-3 eV/Angstrom. These are software precision budgets, not chemical
accuracy bounds for a new system.

The explicit profile allows restart recomputation differences up to 1e-4 eV
and 1e-3 eV/Angstrom. IDs, image flags, geometry, velocities, physical settings,
model bytes and accepted-history identity remain strict. Configurations
without this object retain their original 1e-6 eV and 1e-5 eV/Angstrom restart
limits. The old stricter deployment failures are not relabeled.

A useful force finite difference must resolve the archive's coordinate-dependent
energy quantization. Retain raw energies at multiple displacements and require
consecutive Richardson estimates to satisfy the unchanged force budget.
A smaller displacement can increase float32 cancellation error; never select
a single favorable displacement after seeing the result.

Qualification requires actual native E/F, all-QM cancellation, complete ME
derivatives, ownership migration, serial/MPI, split and cross-rank restart,
short multi-step-size NVE and real backend failure. Cost comparisons must time
identical complete workloads, including initialization, QM callbacks, transfers
and checkpoint I/O. Kokkos registration alone proves neither these gates nor
an end-to-end speedup. Exact results belong to the frozen task review.

The driver keeps Kokkos alive across sequential LAMMPS instances and registers
the public Kokkos-only finalizer once for process exit, before CUDA unloading.
It does not finalize an application-owned MPI or Python runtime. Each run
still closes its LAMMPS instance in success and failure paths.


## Reactive additive environment domains

The separate finite reactive additive route admits a checksummed DPA4C archive
with `inference_profile: dpa4c-native-practical-v1` and:

```json
"execution": {
  "pair_style": "deepmd/kk",
  "numerical_profile": "float32-practical-v1",
  "domain": "environment"
}
```

Its outer native pair remains `zero`. An owned one-rank LAMMPS instance
evaluates only each partition's neutral DPA environment, using `MPI.COMM_SELF`.
Global atom IDs are mapped to dense local IDs and mapped back after every
evaluation. Changing domain size or proton ownership rebuilds atoms and
neighbors while retaining the loaded model. Ordinary `deepmd` is an explicit
alternative with a separately prepared reference and restart identity.

Prepare a fresh isolated-water reference with the same compressed profile
and `low_backend: deepmd/kk` (or `deepmd`). Schema 2 binds that backend,
model, QM settings and source; raw schema 1 references cannot be substituted.
The complete additive mixer retains embedding feedback, cross interactions
and weight derivatives. NVT and fixed-pool OPES/Voronoi are admitted through
their existing contracts. No full-system mechanical pair is added.

Resources are registered before mixer construction and closed on success or
failure. Statistics report actual environment pair, communicator, evaluated
domains and cleanup. The outer driver does not initialize Kokkos; process-exit
cleanup belongs to the inner provider. Short controlled switching tests are
engineering evidence, not spontaneous ionization or an efficiency benchmark.

## Named CV increment

See [NAMED_CV.md](NAMED_CV.md) for the scalar/Jacobian and official external
PLUMED restraint API. This API does not admit a new electronic provider or
sampled state. Existing native water OPES remains the history owner for its
configuration; named width maps normalize to its historical positional order.
