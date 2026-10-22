# Fixed-region QM/DPA coupling

This tool adds an isolated molecular QM correction to a DeePMD potential in
LAMMPS: `E_hybrid = E_DP(full) + E_QM(region) - E_DP(region)`. It uses the public
LAMMPS library and `fix external` interfaces. The same subtraction implementation
is used for standalone validation and LAMMPS dynamics.

## Installation

Use Python 3.10 or newer, NumPy, a shared LAMMPS build with EXTRA-FIX and the
DeePMD pair style, and matching DeePMD/PySCF installations. MPI runs also need
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
`PySCFBackend` converts multiplicity to `spin = N_alpha - N_beta`.

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
`molecules` (lists partitioning those IDs into complete molecules). `qm` contains
PySCF settings such as `xc`, `basis`, `charge`, `multiplicity`, `conv_tol`, and
`threads`. Conditional models require separate `full_charge_spin` and
`region_charge_spin`. Optional `temperature_K` plus `velocity_seed` initializes
velocities on a fresh run only. Otherwise the input data velocities are used.
`sample_interval` defaults to one. `steps` always means additional steps.

Each successful segment writes an accepted frame, a native LAMMPS restart, and a
checksummed companion checkpoint. A restart verifies the model, input identity,
physical settings, step, recomputed energy/forces, and restored positions/images
and velocities. Output directories must be new. Generated artifacts are not
source files and should not be committed.

## Scope and failure semantics

- Fixed complete-molecule membership is an input contract. No covalent cuts,
  adaptive membership, or automatic assignment of a spatial QM region is made.
- Initial image flags must describe the intended intact QM molecules and their
  relative positions. Images are preserved continuously; do not reset them
  during a run. Fixed-cell triclinic image mapping is supported.
- Only `units metal`, fixed cells and NVE are implemented by the CLI. Correction
  virial and full hybrid pressure are not qualified; do not use NPT.
- PySCF failures and invalid model output never reuse previous forces. Callback
  errors stop the run and invalidate its in-memory failed segment. A ctypes
  callback cannot unwind LAMMPS mid-step safely. Use only the last accepted
  checkpoint; do not attach automatic LAMMPS dumps/restarts to such segments.
- Exact full-QM cancellation requires identical full and region low-level inputs,
  including periodicity, electronic state, precision, and model. A periodic full
  model minus an isolated-region model need not cancel even when IDs coincide.
- Passing numerical tests or completing a trajectory does not establish physical
  suitability of a pretrained model for water or a reaction boundary.
