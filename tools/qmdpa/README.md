# QM/DPA coupling

This package couples LAMMPS, DeePMD and explicit quantum providers through the
public LAMMPS library and fix external interfaces. Fixed-region mechanical
embedding uses E_DPA(full) + E_QM(region) - E_DPA(region). Additive EE and
gap-anchor have their own complete potentials and assembly rules.

The unified configuration also supports finite complete-molecule ground-state
partition families. Read [CAPABILITIES.md](CAPABILITIES.md) for the current
backend/method matrix, [CONFIGURATION.md](CONFIGURATION.md) for selectors, and
[DEPLOYMENT.md](DEPLOYMENT.md) for isolated package installation and runtime
requirements. Qualification remains specific to source, method and inputs.

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
`PySCFBackend` converts multiplicity to `spin = N_alpha - N_beta`. Its
`grid_response=True` default includes the response of the numerical DFT
integration grid in the nuclear gradient. Set it explicitly to `False` only
when comparing that approximation; tightening SCF alone does not replace the
grid response or establish force convergence. `nlc_grid_level=None` preserves
PySCF's nonlocal-correlation grid default. A nonnegative integer selects that
grid independently of `grid_level` for ordinary and electrostatic providers.

`ExternalCoupling(lmp, region, calculator, type_symbols=...)` adds region
corrections only, maps current local atom tags on every callback, and uses the
LAMMPS communicator. Supply the calculator on rank zero. Call its `run()` method
so failures caught inside a ctypes callback are re-raised by the driver.

## Selected electronic states

The CPU RHF/TDA route and its fixed-reference state contract are described in
[EXCITED.md](EXCITED.md). A separate experimental PBE0/def2-SVP/grid4 API is
described in [PBE0_TDA.md](PBE0_TDA.md). PBE0 has an explicit experimental factory/configuration
route; complete hybrid-Hamiltonian and native qualification remain separate.
Its analytic quadrature response has its own reference identity and diagnostics.
The shared CPU RHF/PBE0 and GPU RHF [state-set API](MULTISTATE.md) provides
several fixed-reference energies and complete gradients at one geometry using
one common SCF/TDA solve.
The unified prepare-states command prepares its named references and portable
files; see [STATE_INPUTS.md](STATE_INPUTS.md).

The bounded CPU ground mechanical ABACUS route uses the existing official
ASE/abacuslite worker. See [ABACUS.md](ABACUS.md) for its finite-supercell and explicit same-cell periodic routes,
source identity and checked cleanup requirements.

## Standalone electrostatic embedding

`PySCFElectrostaticBackend` evaluates a finite QM system in caller-supplied fixed
external charges using official PySCF `qmmm`. Given nonperiodic `Geometry` objects
`qm` and `environment`, a charge array `fixed_charges_e` in environment atom-ID
order, and a nonempty `charge_source` describing its provenance:

```python
from qmdpa.embedding import FixedEmbeddingCharges, PySCFElectrostaticBackend

charges = FixedEmbeddingCharges(
    environment.atom_ids, fixed_charges_e, source=charge_source
)
provider = PySCFElectrostaticBackend(
    charges,
    xc="pbe",
    basis="def2-svp",
    conv_tol=1e-12,
    conv_tol_grad=1e-8,
    grid_level=4,
    nlc_grid_level=4,
    grid_response=True,
)
result = provider.evaluate(qm, environment)
energy_ev = result.energy_ev
qm_forces = result.qm_forces_ev_per_angstrom
environment_forces = result.environment_forces_ev_per_angstrom
```

The energy includes QM internal energy and QM interactions with the external
charges. Both force arrays are in eV/Angstrom, with orders recorded in
`result.qm_atom_ids` and `result.environment_atom_ids`. Pass both current
geometries on every call; membership, symbols and order remain fixed. SCF failure
raises an error without reusing a previous result. Charges are in electron units;
optional `radii_angstrom` sets Gaussian charge widths with exponent
`zeta = 1 / radius**2`. Charges and widths must be independent of coordinates.

`nlc_grid_level=None` preserves PySCF's nonlocal-correlation grid default;
ordinary and nonlocal grids are controlled separately. `dict(result.diagnostics)`
records the method, convergence settings, grids and charge provenance.
`include_qm_dipole=True` additionally records the QM-only dipole, excluding
external charges, about the zero Bohr origin; charged-QM dipoles depend on origin.
The settings above are example inputs, not a qualified dynamics protocol.

This provider is separate from the CLI and `SubtractiveCalculator`. It supplies
neither environment internal energy nor QM/environment short-range interactions,
and has no DPA cross-interaction coupling. Inserting its energy into the isolated
QM subtraction can double-count interactions and requires a separate energy
model. Periodic electrostatics, coordinate-dependent charges, virtual sites and
adaptive regions are outside its scope. Finite molecular tests do not qualify
oxide interfaces, OMat24 model suitability, or hybrid dynamics. Real embedding
force tests are opt-in with `QMDPA_REAL_EMBEDDING_TESTS=1` for
`tests/test_embedding.py`.

## Finite additive electrostatic potential

`qmdpa.additive` composes a separately declared finite Hamiltonian:

```text
E_total = E_DPA(environment) + E_QM_embedded(QM; fixed charges)
          + E_cross_LJ(QM, environment)
```

DeePMD receives only the environment. The embedded term includes QM internal
energy and its interaction with external charges, including reaction forces on
those charge sites. No full-system DPA or extra environment Coulomb term is
added. A caller must justify the charge and cross-parameter model separately.

```python
from qmdpa.additive import AdditiveCalculator, CrossLennardJones

cross = CrossLennardJones(cross_parameters, source=parameter_source)
calculator = AdditiveCalculator(
    environment_dpa, provider, cross,
    qm_atom_ids=qm.atom_ids, environment_atom_ids=environment.atom_ids,
)
terms = calculator.evaluate_terms(full_geometry)
energy_ev = terms.total.energy
forces = terms.total.forces
```

`cross_parameters` maps each `(QM_symbol, environment_symbol)` to
`(epsilon_eV, sigma_Angstrom)`. Every occurring pair needs explicit values,
including deliberately zero epsilon; there are no mixing rules, cutoffs or
within-region LJ interactions. Parameters and their source are immutable.
The potential is `4*epsilon*((sigma/r)**12-(sigma/r)**6)` for each cross pair.
Coincident sites and invalid numerical output are rejected.

The two fixed, nonempty regions must be disjoint and cover every input atom.
Full input order may change; providers always receive their configured atom
orders and the total forces follow the full input order. Element identity
cannot change. `terms.environment`, `terms.embedded` and `terms.cross` expose
the independent contributions, while `evaluate()` returns only `terms.total`.
The caller owns provider lifetimes. Periodic geometries are unsupported.

For raw PyTorch checkpoints, `DeepMDBackend(..., no_jit=True)` explicitly
passes the official no-scripting option and rejects compiled archive inputs.
Its default is false, preserving existing callers. This option alone does not
make inference deterministic: the application must declare and verify its
PyTorch/CUDA runtime policy before loading a model. An already compiled AOT
archive cannot be made deterministic by this option or by a runtime flag.

The additive native adapter described below supplies this complete energy and
all forces with a zero native pair baseline. Its configuration and qualification
are distinct from the mechanical subtraction route. Creating a standalone
calculator alone does not establish native communication or restart validity.

## Command line

```bash
qmdpa validate run.json
qmdpa run run.json --output new-run-directory
qmdpa run continuation.json --output new-continuation --restart previous/checkpoint-000000000010.json
```

The shared run command selects an existing implementation using the
configuration's explicit Hamiltonian. Fixed-region mechanical, additive and
gap-anchor routes coexist with the declared molecular ME and additive EE
families and explicit fragment corrections listed in CONFIGURATION.md. A legacy schema-1 input without this field
continues to select mechanical embedding only. No electrostatic model is
selected by fallback. See [configuration routing](CONFIGURATION.md) for the
assembly table, preflight output and compatibility boundaries.

Validation loads the same configuration and checks file identities, settings
and declared provider capabilities without constructing an evaluator or running
LAMMPS. Its success is configuration validity, not numerical or scientific
acceptance. Configuration loading during a native run remains collective, with
only rank zero reading the input.

The mechanical JSON schema requires `schema_version: 1`, `data_file` (atomic-style LAMMPS
data), `model`, `model_sha256`, `type_symbols` (LAMMPS types in order), `periodic`
(true or false), `region`, `qm`, `timestep_fs`, and `steps`. Paths are relative to
the configuration. `region` contains `atom_ids`, corresponding `symbols`, and
`molecules` (lists partitioning those IDs into complete molecules). `qm` contains
PySCF settings such as `xc`, `basis`, `charge`, `multiplicity`, `conv_tol`,
`conv_tol_grad` (positive or `null`), `grid_level`, `nlc_grid_level` (a nonnegative
integer or `null`), `grid_response` (a JSON boolean, default `true`), and `threads`.
The optional orbital-gradient threshold controls SCF independently of the
energy-change threshold. `null` preserves the
PySCF default, normally `sqrt(conv_tol)`; energy convergence alone does not bound
nuclear-force error. Use an explicit value for a qualified dynamics protocol and
compare forces against a tighter reference on identical geometries.
Omitting `nlc_grid_level` or setting it to `null` preserves the ordinary ground
provider's default and omits the key from its resolved settings. An explicit
integer is recorded in the manifest and restart physics identity.
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

## Scope and failure semantics

- Fixed-region modes retain their declared complete-molecule membership.
  Molecular modes use an explicitly declared family and weight policy; see
  MOLECULAR_ADAPTIVE.md and MOLECULAR_ADDITIVE.md. Covalent cuts and automatic
  chemical regrouping are unsupported.
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


## Finite electrostatic additive dynamics

The experimental additive mode is available through the shared run command
with its explicit Hamiltonian, or through the preserved dedicated entry point:

```bash
python -m qmdpa.additive_cli CONFIG.json --output FRESH_DIRECTORY
python -m qmdpa.additive_cli CONFIG.json --output NEW_DIRECTORY --restart CHECKPOINT.json
```

It supplies the entire potential
`E_DPA(M) + E_QM_emb(Q; q_M) + E_LJ(Q, M)` through an all-atom
`fix external` callback with `pair_style zero`. The original mechanical
command and energy definition are unchanged. Both commands use the same
accepted-frame, collective failure, and checksum-verified restart code.

The configuration declares `hamiltonian: finite-electrostatic-additive`,
nonperiodic whole-molecule `qm_region` and `environment_region`, all fixed
environment charges (and optional Gaussian radii) with a source, the
environment charge/spin, and every directional cross LJ parameter with a
source. Cross mixing and cutoffs are unsupported. The fixed-charge sum must
match the environment model charge. No charge response or additional
environment Coulomb term is included.

Only the explicit `torch-raw-deterministic-v1` inference profile is currently
supported by this driver. It requires an original `.pt` model, `no_jit=True`,
strict Torch deterministic algorithms, TF32 off, and disabled DeePMD compiled
or custom inference paths. Conflicting environment flags fail before backend
loading; policy changes fail before accepting the force call. There is no AOT
fallback. These inputs, model/data hashes and software identity participate in
restart checks. Restart qualification is limited to the same externally frozen
native library stack; the configuration does not fingerprint every linked
native dependency. The native box, atom IDs/types, zero native interactions and
all force shapes are checked at the callback.

A finite three-water raw-model regression passes native energy/all-atom-force
assembly, serial/two-rank SCF-failure rejection, and 20 versus 10+10 restarted
steps in both modes. This is a bounded 5 fs integration screen, not timestep
convergence, long-time dynamics or physical water qualification. Targeted native
sorting/migration evidence, periodic embedding, oxide transfer and mutual
polarization remain separate requirements. Portable inputs and detailed gates
are maintained in the companion qm-dpa repository's benchmarks/water directory.

## Execution measurements

Both CLI commands add a `metrics` snapshot to `result.json` on completion and
to `failure.json` when the output directory has been created. The schema version
is 1. Measurements restart from zero for each new CLI invocation, including a
continuation. They do not change the Hamiltonian, checkpoint schema or force
evaluation policy.

`scope: owner_rank_host_wall` means the recorded intervals use
`time.perf_counter` on rank zero, without an MPI reduction or additional
synchronization. They are neither CPU usage nor GPU kernel timings, and the
intervals are not a complete job-time decomposition:

- `initialization_wall_seconds`: native LAMMPS construction, configuration,
  restart verification, provider construction and setup through manifest writing,
  before the initial `run 0`; includes a failed initialization.
- `run_wall_seconds`: all coupling `run` calls, including `run 0`, setup,
  force callbacks and failed calls.
- `checkpoint_wall_seconds`: native restart writes and the existing collective
  companion checksum/JSON/frame writes, including failures. Separate frame
  gathering, final report writing and teardown are excluded.

The nested `coupling` snapshot reports `rank`, local `callback_calls`, and
root-only `calculator_attempts`, `calculator_successes`,
`calculator_failures`, `calculator_wall_seconds` and
`consecutive_repeated_geometry_attempts`. The calculator interval wraps the
adapter evaluation after geometry assembly. A successful return is not yet an
accepted callback or checkpoint: subsequent force mapping or native integration
can still fail. These counters do not separate DeePMD, SCF or gradient work.

A repeated input has exactly equal assembled IDs, symbols, coordinates and cell
to the immediately preceding calculator attempt, including failed attempts.
There is no tolerance, memoization or reuse of returned forces. Callbacks rejected
before the calculator do not replace that previous attempted input. The bridge's
`statistics()` returns a detached JSON-safe snapshot; `reset_statistics()`
clears only measurements and the previous-input reference, never a failure,
accepted state or simulation state.

## Offline and experimental diagnostics

[Offline water selection](SELECTION.md) proposes complete OH groups with explicit
ambiguity diagnostics. It does not change dynamics. [Reference-subtractive
embedding](REFERENCE.md) compares a declared fixed-charge Coulomb counterterm
with an embedded QM provider; it does not establish a decomposition of pretrained
DPA electrostatics. Neither tool changes the existing dynamics CLI. The companion
validation project supplies real-provider scans and separate acceptance records.

[Fixed-charge configuration](CHARGES.md) documents explicit water charge templates,
per-atom/type/file inputs, overrides, provenance and additive schema 2.
It does not assign a universal element charge or qualify a new physical model.

## Experimental gap-anchor candidate

See [GAP_ANCHOR.md](GAP_ANCHOR.md) for the separate standalone ground-ME plus
embedded-excitation-gap model, its five-term ledger and qualification limits.

The next development version adds an optional [static distance bias](BIAS.md)
through the unified entry point. Its qualification is separate from the rc2 candidate.
