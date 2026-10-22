# Fixed-region QM/DPA coupling

This tool adds a molecular QM-region correction to a DeePMD potential in
LAMMPS: `E_hybrid = E_DP(full) + E_QM(region) - E_DP(region)`. It uses the public
LAMMPS library and `fix external` interfaces. The same subtraction implementation
is used for standalone validation and LAMMPS dynamics.

## Installation

Use Python 3.10 or newer, NumPy, a shared LAMMPS build with EXTRA-FIX and the
DeePMD pair style, and matching DeePMD and QM-backend installations. PySCF is the
default QM backend. The experimental ABACUS backend additionally requires ASE,
the official `abacuslite.AbacusSocketIO` interface, and a compatible serial
ABACUS executable with socket support. MPI runs also need
mpi4py built against the MPI library used by LAMMPS. From this directory:

```bash
python -m pip install --no-deps .
python -m pytest tests
```

Heavy dependencies are imported only when their backend is constructed. Pure
unit tests use controlled fixtures for bookkeeping; they do not establish model
accuracy. Real provider tests are opt-in with `QMDPA_REAL_TESTS=1`, `QMDPA_MODEL`,
and, where required, `QMDPA_CHARGE_SPIN='0,1'`.

## API

`Geometry` records positive stable atom IDs, element symbols, Angstrom positions,
and an optional cell (row lattice vectors, three-dimensional periodicity).
`cell=None` means isolated. `Evaluation` returns eV and eV/Angstrom.
`SubtractiveCalculator(low, high, full_low=...)` is the sole subtraction code.
The full and region providers can have different declared electronic states.
`DeepMDBackend` accepts `charge_spin` separately from `fparam` and `aparam`;
`PySCFBackend` converts multiplicity to `spin = N_alpha - N_beta`. Its
`grid_response=True` default includes the response of the numerical DFT
integration grid in the nuclear gradient. Set it explicitly to `False` only
when comparing that approximation; tightening SCF alone does not replace the
grid response or establish force convergence.

`ExternalCoupling(lmp, region, calculator, type_symbols=...)` adds region
corrections only, maps current local atom tags on every callback, and uses the
LAMMPS communicator. Supply the calculator on rank zero. Call its `run()` method
so failures caught inside a ctypes callback are re-raised by the driver.

## Command line

```bash
qmdpa run run.json --output new-run-directory
qmdpa run continuation.json --output new-continuation --restart previous/checkpoint-000000000010.json
```

The JSON schema requires `schema_version: 1`, `data_file` (atomic-style LAMMPS
data), `model`, `model_sha256`, `type_symbols` (LAMMPS types in order), `periodic`
(true or false), `region`, `qm`, `timestep_fs`, and `steps`. Paths are relative to
the configuration. `region` contains `atom_ids`, corresponding `symbols`, and
`molecules` (lists partitioning those IDs into complete molecules).
`qm.engine` selects `pyscf` (also the default when omitted) or `abacus`. Unknown
engines and settings are rejected. For PySCF, `qm` contains settings such as
`xc`, `basis`, `charge`, `multiplicity`, `conv_tol`,
`conv_tol_grad` (positive or `null`), `grid_level`, `grid_response` (a JSON
boolean, default `true`), and `threads`. The optional orbital-gradient threshold
controls SCF independently of the energy-change threshold. `null` preserves the
PySCF default, normally `sqrt(conv_tol)`; energy convergence alone does not bound
nuclear-force error. Use an explicit value for a qualified dynamics protocol and
compare forces against a tighter reference on identical geometries.
The resolved grid-response setting is saved in the manifest and restart physics
identity. Changing it requires a fresh run. Numerical results obtained before
this explicit policy must be revalidated with finite differences and NVE; the
new default does not by itself qualify a grid, force, or trajectory.
Conditional models require separate `full_charge_spin` and
`region_charge_spin`. Optional `temperature_K` plus `velocity_seed` initializes
velocities on a fresh run only. Otherwise the input data velocities are used.
`sample_interval` defaults to one. `steps` always means additional steps.

Each successful segment writes an accepted frame, a native LAMMPS restart, and a
checksummed companion checkpoint. A restart verifies the model, input identity,
physical settings, coupling source/dependency identity, step, recomputed energy/forces, and restored positions/images
and velocities. Output directories must be new. Generated artifacts are not
source files and should not be committed.

## Experimental ABACUS backend

ABACUS uses a persistent official socket calculator for a neutral singlet region
(`charge: 0`, `multiplicity: 1`). Nonperiodic region coordinates use an explicit
mapping and are evaluated in an explicitly declared finite periodic supercell.
The default mapping retains the same fixed offset on every call.
This is a molecular supercell approximation, with fixed
PW/PBE, Gamma sampling and fixed occupations. It does not establish convergence
with vacuum size, cutoff or SCF settings. ABACUS force, dynamics and restart
qualification remains pending; process-contract tests do not supply that evidence.

An example `qm` object is shown below. Paths are relative to the run configuration;
replace each SHA256 placeholder with the full digest of the actual binary or
pseudopotential. These values illustrate the schema, not a qualified protocol.

```json
{
  "engine": "abacus",
  "executable": "software/abacus",
  "executable_sha256": "<binary-sha256>",
  "pseudopotentials": {
    "H": {"path": "pseudopotentials/H.upf", "sha256": "<H-upf-sha256>"},
    "O": {"path": "pseudopotentials/O.upf", "sha256": "<O-upf-sha256>"}
  },
  "supercell_angstrom": [12.0, 12.0, 12.0],
  "coordinate_offset_angstrom": [6.0, 6.0, 6.0],
  "coordinate_gauge": "fixed-offset",
  "ecutwfc": 60.0,
  "scf_thr": 1e-9,
  "scf_nmax": 100,
  "ks_solver": "cg",
  "timeout_seconds": 120.0
}
```

`coordinate_gauge` accepts exactly `fixed-offset` (default) or
`arithmetic-centroid`. The default evaluates `X_i = R_i + offset`. The explicit
centroid mode evaluates `X_i = R_i - mean(R) + offset` on every call, with an
equal-weight mean over every atom supplied to the provider. The offset is then
the target centroid, often the cell center. This operation uses no atomic masses,
wrapping, or separate per-molecule shifts, and leaves the input coordinates intact.

These mappings define different finite-grid energy functions. Native ABACUS
removes the equal-weight mean force when external fields are absent. For
`E_g(R) = E_native(R - mean(R) + offset)`, the chain rule gives
`F_i = F_raw_i - mean(F_raw)`, matching that native projection when the raw force
is the energy derivative. The adapter passes the native energy and forces
through unchanged; it adds no force projection. The centroid contract is limited
to the complete fixed neutral singlet molecular set evaluated without external
fields or charge embedding. It does not establish periodic embedding or vacuum
convergence. Changing the mode changes restart identity and requires fresh full
finite-difference, translation, and dynamics qualification; choosing the mode
does not itself qualify a calculation.

`ks_solver` explicitly selects the official PW eigensolver: `cg` (default),
`dav`, or `dav_subspace`. There is no automatic solver fallback. A different
solver changes restart identity and requires fresh numerical qualification;
support in this interface does not guarantee convergence of a given input.

The executable and pseudopotential digests, cell, offset, coordinate gauge, resolved settings and
ASE/abacuslite source identities enter the restart identity. Atom IDs and element
order stay fixed within a session. The driver supplies a fresh output directory;
`directory` is not a QM setting. Native ABACUS must resolve its own runtime
libraries: its launch removes inherited `LD_PRELOAD` and `LD_LIBRARY_PATH` and
sets OpenMP, OpenBLAS and MKL thread counts to one without changing the caller.

On POSIX systems, the worker and native ABACUS share a private process group.
Requests have a deadline, including first-call initialization. A timeout,
unconverged SCF or invalid response permanently invalidates that backend and
terminates the group; previous forces are never returned. A separate watchdog
also detects loss of the parent while initialization, evaluation or close is
blocked, removes the session's unique socket and terminates its own group.
Close is idempotent and cleanup errors cannot replace the original evaluation
error. Failure diagnostics and native logs remain in the output directory.

## Scope and failure semantics

- Fixed complete-molecule membership is an input contract. No covalent cuts,
  adaptive membership, or automatic assignment of a spatial QM region is made.
- Initial image flags must describe the intended intact QM molecules and their
  relative positions. Images are preserved continuously; do not reset them
  during a run. Fixed-cell triclinic image mapping is supported.
- Only `units metal`, fixed cells and NVE are implemented by the CLI. Correction
  virial and full hybrid pressure are not qualified; do not use NPT.
- QM failures and invalid model output never reuse previous forces. Callback
  errors stop the run and invalidate its in-memory failed segment. A ctypes
  callback cannot unwind LAMMPS mid-step safely. Use only the last accepted
  checkpoint; do not attach automatic LAMMPS dumps/restarts to such segments.
- Exact full-QM cancellation requires identical full and region low-level inputs,
  including periodicity, electronic state, precision, and model. A periodic full
  model minus an isolated-region model need not cancel even when IDs coincide.
- Passing numerical tests or completing a trajectory does not establish physical
  suitability of a pretrained model for water or a reaction boundary.
