# Preparing a selected-state input

The installed command exposes CPU RHF/TDA, bounded CPU PBE0/TDA and
GPU RHF/TDA reference builders:

    qmdpa prepare-state request.json --output fresh-state-directory

This command performs a quantum calculation. Run it under the same runtime and
resource allocation required for the requested provider. RHF and PBE0 runtime
pins remain distinct; the command does not install or change a quantum engine.
It does not require a DPA model or start LAMMPS.

A minimal isolated RHF request is:

```json
{
  "schema_version": 1,
  "method": "rhf-tda",
  "qm": {"basis": "def2-svp", "tda_solver": "dense", "threads": 2},
  "qm_geometry": {
    "atom_ids": [1, 2, 3],
    "symbols": ["O", "H", "H"],
    "positions_angstrom": [[0, 0, 0], [0.963, 0, 0], [-0.244, 0.928, 0.035]]
  },
  "state": {"initial_root": 1, "label": "Declared water interface state"}
}
```

Use `method: pbe0-tda` with the explicit solver and grid controls required by
[PBE0_TDA.md](PBE0_TDA.md). Unsupported methods, missing controls and density
fitting in selected-state settings are rejected. This is state initialization,
not general TDDFT support or validation of an excitation's physical accuracy.

For supported fixed-charge embedding also supply `environment_geometry` in the same
geometry schema and `charges` with `atom_ids`, `charges_e`, `source` and explicit
`radii_angstrom: null`. IDs and their order must match the environment geometry.
The native selected-state route accepts fixed point charges only. Geometry
coordinates establish the reference for state matching; they do not fix MD
coordinates in place.

A successful fresh output directory contains:

- `reference.json`: complete immutable method-specific state record;
- `provider-inline.json`: the existing full provider input;
- `provider-file.json`: a compact provider input with a byte-level file hash;
- `qm.json`: complete resolved settings to use in the run configuration;
- `request.json`, `manifest.json`, `result.json`: input and completion identities.

A failed preparation writes `failure.json` and a failed manifest. Existing
output directories are never overwritten. No gradient, native trajectory or
chemical accuracy qualification follows from preparing a reference.

## Reading a reference from a file

Fixed-region ME, additive and gap-anchor loaders accept the optional file form:

```json
"provider": {
  "engine": "pyscf",
  "device": "cpu",
  "state": {
    "method": "rhf-tda",
    "reference_file": {"path": "reference.json", "sha256": "<file SHA256>"}
  }
}
```

Copy the generated reference alongside the run configuration and use the
`provider-file.json` contents as its provider block; use the `qm.json` contents
as its qm block. Paths resolve relative to the run configuration. Keep this
relative layout when moving inputs to another machine, or explicitly change
the path. The hash identifies file bytes; the record's separate semantic hash
identifies the state. A file and an inline reference cannot both be supplied.

The loader verifies file bytes and expands them to the same full inline record
before the existing method, field, atom-order, runtime and restart checks.
Equivalent inline/file inputs on the same source have identical normalized
configuration and restart physics identity. Files cannot bypass a source or
method mismatch. A reference from one quantum runtime does not automatically
qualify another runtime or host.

The fragment cluster_gap pair_provider and field_provider blocks also accept
this same hashed reference_file form, resolved relative to the run configuration.
The normalized content is identical to inline input. All three state hashes,
matched physical settings/runtime, region/field identity and declared state
correspondence are still checked before any evaluator is created.

## Selected GPU RHF/TDA

The optional engine gpu4pyscf, device cuda, method rhf-tda route supports
source-pinned singlet conventional TDA references with an explicit dense solver.
It accepts fixed finite ME, additive point-charge EE and plain RHF gap-anchor.
State preparation may
declare engine and device at the top level; omission preserves pyscf/cpu.
Embedded preparation also requires environment_geometry and charges. Prepared
GPU files retain their separate reference/runtime identity.

ME uses full native DPA plus the local selected-total-minus-low correction.
Additive EE uses pair zero and returns the complete total, including environment
DPA and explicit cross terms. Accepted EE state records additionally require the
checked total relaxed density and electron-count diagnostics.
GPU fragment, selected PBE0/DFT, density fitting, adaptive partitions
and periodic full systems remain rejected. Actual numerical/native qualification
requires a matching source and runtime record.


## Preparing several states together

Use the same common method, geometry, field and numerical controls as above,
but replace the singular state object with a states mapping:

```json
"states": {"state_a": 1, "state_b": 2}
```

Then run:

    qmdpa prepare-states request.json --output fresh-state-set

This route supports CPU RHF/TDA, bounded CPU PBE0/TDA and GPU RHF/TDA with
an explicit dense solver and at least two distinct initial roots within
qm.nstates. For GPU RHF, set engine to gpu4pyscf and device to cuda, using
the same controls as the singular GPU command. PBE0 retains its explicit grid
controls and separate CPU runtime. The command calls the requested shared-state
builder once and preserves engine-specific reference and runtime identities.
GPU PBE0, density-fitted selected states and iterative solvers remain rejected;
ordinary prepare-state keeps its existing supported methods.

The collection contains request.json, manifest.json, result.json and numbered
states/state-0001, states/state-0002, ... directories in request order. Names are
labels only and never become filesystem paths. Each state directory contains
the same reference.json, provider-inline.json, provider-file.json, qm.json and
result.json formats as the singular command. Keep the completed parent manifest
with the collection when moving it to an identical supported runtime.

The parent result.json maps each requested name to its directory, semantic
reference hash and a provider block whose file path is relative to the collection
root. Put a run configuration at that root and use the chosen provider block plus
its qm.json settings. Alternatively use a state's provider-file.json with a
configuration inside that state directory. The existing ME/additive/gap-anchor
loaders remain responsible for field, region, Hamiltonian and restart checks.
Several prepared references do not request multi-state native dynamics.

All states are validated before their files are published together. Input or
source changes, quantum failures and invalid references mark the parent manifest
failed. Incomplete file output may remain under .states-pending for diagnosis;
it is not a completed collection. Existing output directories are never replaced.
The command prepares references only and does not calculate state gradients.
The [state-set API](MULTISTATE.md) performs shared energy/full-force evaluation.

Common-core selected molecular families use the separate
`qmdpa prepare-state-family REQUEST.json --output FAMILY.json` command.
See [SELECTED_ADAPTIVE.md](SELECTED_ADAPTIVE.md); a family of cross-partition
references is not a multi-root electronic-propagation state.
