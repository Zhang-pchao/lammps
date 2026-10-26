# Fixed-region periodic Gaussian embedding

Development 0.4.0.dev6 implements a distinct ground-state periodic
reference-subtractive Hamiltonian:

    U = DPA(full,h) + ABACUS(Q,h;field) - DPA(Q,h) - V_ref(Q,M,h).

Use the unified `qmdpa validate` and `qmdpa run` commands with
`hamiltonian: "periodic-electrostatic-reference-subtractive"`, or
`coupling: {"surface":"ground", "embedding":"electrostatic",
"ee_model":"periodic-reference-subtractive"}`. Native `pair_style zero`
receives the complete energy and all forces. Do not add native full DPA again.

## Bounded configuration

The provider is CPU ABACUS ground state. Full DPA and QM-fragment DPA use the
same fixed orthorhombic three-dimensional cell as the PW/PBE/Gamma/NCPP solve.
Require neutral singlet QM and full DPA conditions `[0,1]`, fixed occupations,
cold FileIO, periodic wrapping, zero coordinate offset and fixed atom IDs.
Use the executable, pseudopotentials and runtime-file identities in ABACUS.md.
An optional explicit `qm.ecutrho` controls the density/potential cutoff in Ry
independently of `ecutwfc`, and must be at least four times it. Omission retains
the native default. Both settings enter restart identity; do not infer grid or
force convergence from a wavefunction cutoff alone.
The executable must implement the declared Gaussian-field extension; a stock
binary or a matching version string alone does not supply that extension.

Use `qm_region` and `environment_region` with IDs, symbols and explicit groups,
`periodic: true`, and `inference_profile: "torch-raw-deterministic-v1"` with
an identified raw .pt model. Add these explicit settings (values illustrate
syntax, not an application-ready physical model):

```json
"qm": {
  "boundary": "periodic-same-cell",
  "execution_mode": "fileio",
  "coordinate_gauge": "periodic-wrap",
  "coordinate_offset_angstrom": [0, 0, 0],
  "supercell_angstrom": [6, 6, 6],
  "periodic_field": {
    "atom_ids": [11, 12], "charges_e": [0.3, -0.3],
    "sigma_angstrom": [0.8, 0.8], "cell_angstrom": [6, 6, 6],
    "kmax": 6, "source": "Explicit neutral numerical fixture"
  }
},
"reference": {
  "form": "periodic-gaussian-charge-cross-reference",
  "qm_charges": {
    "atom_ids": [1, 2], "charges_e": [0.2, -0.2],
    "source": "Explicit neutral reference model"
  }
}
```

Include the remaining executable/pseudopotential identities and dynamics
settings in a complete configuration. The field order must match environment
IDs, and point-reference order must match QM IDs. Both charge sets are neutral.
The nonempty field API admits zero charges. For empty environment/all-QM,
select the explicit existing same-cell mechanical route, whose low-level terms
cancel under identical arguments. An empty Geometry is not an admitted input.

## Field and energy ownership

`sigma_angstrom` is a physical standard deviation, not a PySCF Gaussian radius
or an adjustable Ewald splitting parameter. The complete signed reciprocal
cube `[-kmax,kmax]^3` excludes zero and includes both signs exactly once.
The truncated cube defines the potential under test; its convergence is a
separate qualification. Every retained mode must exist in the native FFT grid.

G=0 is omitted in a 3D mean-zero/tin-foil convention. Separate neutrality avoids
a compensating background. ABACUS owns QM internal Coulomb/Ewald terms. The
field adds QM-environment terms and their images exactly once, including the
electronic response and ionic interaction. Environment self and pair Coulomb
energies are excluded. No slab, open-boundary or dipole correction is added.
The reference uses the identical kernel and native ABACUS unit constants, with
fixed neutral QM point charges; it is not a decomposition of DPA electrostatics.
Both reference force sets are subtracted. The QM net force must not be removed:
the environment receives its reaction.

All field, reference, cell, model and dependency identities enter restart
validation. Native MPI distributes atoms and broadcasts a root-owned serial
QM evaluation. Failed evaluations invalidate the segment and preserve earlier
accepted checkpoints. A request timeout is distinct from scheduler walltime.

This development route needs matched frozen numerical and native evidence.
It does not qualify metallic occupations, Cu slabs, charged components, QM GPU,
selected states, periodic adaptive partitioning, NPT/stress, mutual polarization,
periodic sampling, long dynamics, scientific accuracy or a performance advantage.
