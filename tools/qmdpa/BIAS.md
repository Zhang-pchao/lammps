# Static distance bias

The unified `qmdpa run` and `qmdpa validate` entry points accept an optional
`bias` object for finite fixed-region mechanical, additive electrostatic and
plain gap-anchor configurations. Ground and supported selected states retain
their existing electronic-state contracts. Legacy mode-specific loaders reject
this new key; use the unified entry point.

```json
"bias": {
  "form": "harmonic-distance",
  "restraints": [
    {"atom_ids": [1, 4], "k_ev_per_angstrom2": 0.2, "distance_angstrom": 2.8}
  ]
}
```

The additional potential is `sum(0.5*k*(r-r0)^2)`, where `r` is the direct
Cartesian distance in Angstrom between the two atom IDs, `k` is in
eV/Angstrom squared, and `r0` is in Angstrom. It is an external sampling bias,
not a correction to the electronic Hamiltonian or a new QM/DPA embedding model.
Both atoms receive the full conservative bias force. The DPA/QM calculator
returns its original coupling quantity, without this additional energy.

The driver creates one single-atom group per restrained ID and uses official
LAMMPS [fix spring couple](https://docs.lammps.org/fix_spring.html) with zero
displacement. It enables `fix_modify energy yes` and creates the springs after
the external coupling. Additive/anchor callbacks still require exactly zero
pre-callback native forces; the spring adds its force afterward. There is no
second production force implementation. At most 31 distinct restrained atoms
are supported because native group IDs are limited.

Every accepted frame keeps the complete biased native potential and forces.
Its `bias` record provides the native bias energy, the potential after
subtracting the bias, each distance, and the native force on the first listed
atom (the other atom receives its negative). The underlying selected-state
provider energy remains the electronic energy. Reweighting and free-energy
estimation are separate tasks; an unbiased energy field does not turn the
trajectory into an unbiased sample.

The bias is part of the physics checksum. Restart recreates groups and fixes
from the same validated configuration because binary restart does not store
the spring fix. A changed spring constant, target distance, atom identity or
missing/tampered bias record is rejected. Native energies and full forces are
recomputed before accepting the restart.

This entry is deliberately finite and static: periodic images, moving targets,
adaptive partition composition, corrected/fragment Hamiltonians, arbitrary
native command strings, metadynamics and thermostats are not enabled here.
Parameters must be finite and positive. Duplicate pairs and missing atom IDs
are rejected. At every force evaluation the driver checks finite coordinates,
zero image flags and a distance strictly above 1e-10 Angstrom, the native
spring's small-distance cutoff. A failed segment publishes no new accepted
checkpoint. Choose a stable timestep for the bias stiffness; no long-time,
free-energy or rate qualification follows from interface tests.
