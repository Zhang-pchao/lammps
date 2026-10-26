# Configuration routing

Use the installed qmdpa command or its equivalent python module:

    qmdpa validate CONFIG.json
    qmdpa run CONFIG.json --output FRESH_DIRECTORY
    qmdpa run CONFIG.json --output FRESH_DIRECTORY --restart CHECKPOINT.json
    python -m qmdpa.cli validate CONFIG.json

The Hamiltonian field selects an existing implementation. Routing introduces no
new potential, electronic-state algorithm, integrator or checkpoint format.

| Hamiltonian identifier | Native pair | External energy and forces | Owning configuration |
| --- | --- | --- | --- |
| fixed-region-mechanical | Full-system DeePMD | QM minus region-DPA correction | Mechanical schema in README.md |
| finite-electrostatic-additive | Zero | Complete additive total | Additive schema in README.md |
| finite-electrostatic-reference-subtractive | Zero | Full DPA - region DPA + embedded QM - reference cross Coulomb | REFERENCE.md |
| finite-electrostatic-gap-anchor | Zero | Complete gap-anchor total | GAP_ANCHOR.md |
| finite-molecular-mechanical | Zero | Full DPA plus weighted QM corrections and switching forces | MOLECULAR_ADAPTIVE.md |
| finite-water-buffer-mechanical | Zero | Full DPA plus calibrated variable-water corrections and every switching force | WATER_BUFFER.md |
| finite-molecular-additive | Zero | Weighted complete additive totals and switching forces | MOLECULAR_ADDITIVE.md |
| finite-electrostatic-gap-anchor-cross-delta | Zero | Gap-anchor plus explicit cross difference | INTERACTION_DELTA.md |
| finite-electrostatic-gap-anchor-fragment-response | Zero | Explicit fragment-response selected total | CLUSTER_RESPONSE.md |
| finite-mechanical-fragment-response | Zero | Explicit ground fragment total | CLUSTER_RESPONSE.md |

The identifier is a JSON string in the top-level hamiltonian field. Explicit
mechanical inputs and legacy schema-1 mechanical inputs without the field
normalize identically. For electrostatic modes, declare the identifier or the equivalent explicit selector below.
Unsupported identifiers, duplicate keys, nonfinite JSON numbers, changed inputs
during validation and incompatible fields are rejected without choosing another
model. Existing dedicated Python run APIs remain tied to their own Hamiltonian.

## Inspect before running

The validate subcommand prints the resolved Hamiltonian, native assembly,
region IDs/symbols/molecules, provider contract, QM settings, model/data identities,
source/dependency identity, restart physics hash and requested dynamics. It does
not print a full state transition-density matrix or construct a backend. Optional
scientific dependencies must still satisfy the selected configuration contract.

Validation does not test model suitability, provider convergence, numerical
derivatives, trajectories, MPI communication or scientific accuracy. In
particular, finite RHF/TDA selected-state support does not imply general TDDFT,
adaptive states, nonadiabatic dynamics or a periodic electrostatic method.
Finite CPU ground reference subtraction uses the explicit schema in REFERENCE.md. The historical two-partition
mechanical driver retains its dedicated entrypoint; complete molecular ME and
additive EE families use the shared entrypoint and their own strict schemas.

## State, assembly and restarts

Mechanical embedding retains full-system native DPA plus the differential
correction. Additive, reference-subtractive and gap-anchor modes return their complete
totals with a zero native pair. Never add another full-system DPA term to a
returned EE total.

The common driver retains owner-only collective configuration loading, accepted
frame/checkpoint publication, failure propagation and source-bound restarts.
Output directories must be fresh. The normalized Hamiltonian participates in the
physics identity. A changed source digest also invalidates checkpoints from an
older snapshot: do not remove identity fields or rewrite a saved checkpoint to
force continuation. Source compatibility and scientific qualification are
separate gates.

For selected states, use the state reference and safeguards documented in
EXCITED.md. Gap-anchor and fixed-charge reference-subtractive EE are different
Hamiltonians and cannot inherit one another's evidence.

## Surface and embedding selector

The shared `validate` and `run` entrypoints also accept:

```json
"coupling": {
  "surface": "selected",
  "embedding": "electrostatic",
  "ee_model": "gap-anchor"
}
```

This optional front end consumes `coupling` before the existing model loader
resolves files and computes the physical identity. All model-specific input fields,
charges, LJ parameters, QM settings and selected-state references remain required.
Relative paths retain the original configuration directory. Inputs are not rewritten.
Dedicated mechanical/additive/gap-anchor APIs retain their original schemas.

| Surface | Embedding | EE model | Result |
| --- | --- | --- | --- |
| ground | mechanical | absent | GS-ME |
| selected | mechanical | absent | selected-state ME |
| ground | electrostatic | additive | GS additive EE |
| ground | electrostatic | reference-subtractive | Finite CPU ground reference subtraction |
| selected | electrostatic | additive | selected-state additive EE |
| selected | electrostatic | gap-anchor | selected-state gap-anchor EE |

A selected surface requires an explicit supported `provider.state` with its full
reference. A ground surface cannot use a selected-state provider. The selector
must agree with any explicit Hamiltonian. EE models are never inferred. Ground
gap-anchor is rejected because its defined limit is GS-ME, not a distinct GS-EE
model. Reference-subtractive is admitted for the finite CPU ground contract in REFERENCE.md. ME rejects `ee_model`.

Validation prints the resolved choices and signed total-energy terms, their force
coordinates, the native baseline and the external return contract. This is a
formula description, not an evaluated energy decomposition or a new electronic
calculation. LJ remains the additive cross model; no D3 or repulsion correction
is implied. Black-box DPA terms do not identify unique physical interaction terms.

Equivalent old and selector inputs on the same source produce identical normalized
configurations and restart hashes. Source/dependency identity guards remain active;
this does not make checkpoints from older source snapshots compatible.

## Restart input preflight

`qmdpa validate config.json --restart checkpoint.json` verifies the existing
physics, selected-state and binary-integrity contracts without evaluating a
backend. It does not qualify the recomputed native state. With mpi4py available,
all fixed-region native drivers also perform this check collectively before
constructing LAMMPS. They retain the later validation immediately before restart
loading and the full recomputed-state comparison. The serial path without
mpi4py retains its existing validation after native initialization.

A rejected run preflight writes only `failure.json` in a fresh output directory;
it never overwrites existing evidence. The native integration observer records `input-rejected` separately from native identity
observations; it cannot treat that record as a successful native run.

### Experimental interaction replacement

`coupling.ee_model = "gap-anchor-cross-delta"` selects the distinct finite PBE0
candidate described in [INTERACTION_DELTA.md](INTERACTION_DELTA.md). It requires
an explicit `interaction_delta` block and selected PBE0 state. Existing
`gap-anchor` inputs do not enable it. Configuration validation is separate from
numerical, native, scientific and complete-workload cost qualification.

The optional finite ground fragment model is selected by

```json
{"coupling": {"surface": "ground", "embedding": "mechanical", "me_model": "fragment-response"}}
```

Its ground-only inputs and complete-force zero-pair assembly are documented in
[CLUSTER_RESPONSE.md](CLUSTER_RESPONSE.md). Existing mechanical configurations
without me_model retain their original full-DPA native baseline.

## Optional ground-state density fitting

Ground ME and additive EE inputs may explicitly set
`qm.density_fit = {"auxbasis": "def2-universal-jkfit"}`. Omission or null retains
conventional integrals; enabling fitting or changing the auxiliary basis changes
the restart physics identity. Selected-state and matched fragment/anchor methods
reject this option. See [PROVIDERS.md](PROVIDERS.md#optional-ground-state-density-fitting)
for the full derivative contract, limitations and small-system example.

## Preparing selected states

Use `qmdpa prepare-state REQUEST.json --output FRESH_DIRECTORY` to create an
immutable reference with an existing supported quantum provider. Fixed-region
loaders accept either the existing inline record or an explicit hash-checked
`provider.state.reference_file`. See [STATE_INPUTS.md](STATE_INPUTS.md) for the
request schema, portable file layout and unchanged restart checks.

## GPU ground mechanical configuration

A ground/mechanical selector may declare `provider.engine="gpu4pyscf"` and
`provider.device="cuda"`. Conventional RHF or named-density-fitted PBE/PBE0 and complete response
are required as specified in [GPU_BACKENDS.md](GPU_BACKENDS.md). The QM region is finite; an explicitly periodic full DPA system follows
the separate molecular-region boundary contract described below. The selector and legacy ME form resolve to the same Hamiltonian and
restart identity on the same runtime. GPU execution does not change the
full-DPA-plus-local-difference assembly.

Finite ground/additive EE also accepts `gpu4pyscf/cuda/ground` with explicit
point charges and the existing cross-term fields. Its native pair remains zero
and its callback supplies the complete total. Gaussian radii and dipole diagnostics are rejected. Selected GPU RHF/TDA
uses its separate source-pinned provider; see [GPU_BACKENDS.md](GPU_BACKENDS.md).

## Finite complete-molecule partitioning

Select ground/mechanical with me_model=molecular for the full-DPA baseline
plus weighted corrections. Select ground/electrostatic with
ee_model=molecular-additive for weighted complete additive potentials.
Both use pair zero and complete callback totals. The fixed core, occupancy,
candidate family and all weight derivatives are explicit; the EE route also
requires one full atom-ID charge table and explicit cross LJ parameters.
See MOLECULAR_ADAPTIVE.md and MOLECULAR_ADDITIVE.md. Selected molecular surfaces,
variable occupancy and reactive regrouping are rejected.

The ground mechanical me_model = water-buffer selector routes to the protected
finite water-buffer schema in [WATER_BUFFER.md](WATER_BUFFER.md). It has its own
reference, complete subset family and boundary guard; no other mode inherits it.

## Two-state coherent dynamics

The exact selector `{surface: two-state, embedding: mechanical, me_model: ehrenfest}`
selects the separate finite CPU SA-CASSCF(2,2) route. It requires a prepared signed
pair reference, normalized complex amplitudes and explicit full/QM molecular
regions. Only plain native NVE and strict raw DPA inference are admitted.
See [EHRENFEST.md](EHRENFEST.md) for the complete potential, coherent force,
source patch and restart contract. Ordinary selected-state inputs retain their
single-surface meaning.
