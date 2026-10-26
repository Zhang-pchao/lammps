# Protected finite QM core with variable whole-water buffers

This ground mechanical model keeps a fixed neutral aqueous QM core and allows
zero to all of a bounded pool of intact waters to enter its QM buffer. All
atoms on the intended local proton-transfer path must remain in the protected
core. Internal core O-H redistribution is allowed; transferring a proton across
the core/buffer/environment boundary is rejected. The core does not migrate.

The existing fixed-k model in [MOLECULAR_ADAPTIVE.md](MOLECULAR_ADAPTIVE.md)
retains its meaning. This separate model reuses its complete molecular mixer,
low-level subtraction, native mapping and accepted-checkpoint lifecycle.

## Potential and reference

For one to six candidate waters, declare every subset S, including the empty
buffer. The complete family has at most 64 partitions. A candidate's radial
quintic preference p_i is one inside the inner radius and zero outside the
outer radius, with zero first and second derivatives at both endpoints.

```text
w_S = product(p_i for i in S) * product(1-p_i for i outside S)
delta_water = E_QM(isolated reference water) - E_DPA(the same reference water)
C_S = E_QM(core+S) - E_DPA(core+S) - size(S)*delta_water
E = E_DPA(full) + sum(w_S*C_S)
F = F_DPA(full) + sum(w_S*F_C_S) - sum(C_S*grad(w_S))
```

All Cartesian derivatives, including the force on the core center, are retained.
Only exactly zero weight and gradient permit skipping a partition. No positive
weight is truncated. Providers are reevaluated when a water re-enters support.

The identified isolated-water reference removes a declared per-water energy
zero from the varying compositions. It is part of this Hamiltonian, not a
thermodynamic chemical potential or proof of solvent/reaction accuracy. It
does not repair an unsuitable electronic method or DPA model. At full occupancy
and with no permanent environment, forces equal full QM; energy equals full
QM minus the declared number of reference differences. Do not compare
unshifted total energies as if they should cancel exactly.

## Preparation and inputs

Use the same explicit model hash, neutral-singlet CPU PySCF ground provider,
QM settings and raw deterministic inference profile for calibration and dynamics:

```bash
qmdpa prepare-water-reference REQUEST.json --output NEW_REFERENCE_DIRECTORY
qmdpa validate CONFIG.json
qmdpa run CONFIG.json --output NEW_RUN_DIRECTORY
qmdpa run CONFIG.json --output NEW_CONTINUATION --restart ACCEPTED_CHECKPOINT.json
```

REQUEST.json contains model, model_sha256, qm, provider, geometry_angstrom
(three rows in O,H,H order), inference_profile and source. The source identifies
the frozen calibration choice. The command records computed high/low energies,
geometry, dependency versions and coupling-source digest in reference.json.
It publishes no reference after backend failure. Embed the complete generated
object as CONFIG.json's buffer_reference. Do not change its energies or relabel
the model, method, inference, geometry or software identity. A changed coupling
runtime requires a newly prepared reference and new run qualification.

Use hamiltonian = finite-water-buffer-mechanical, or the equivalent selector:

```json
"coupling": {
  "surface": "ground",
  "embedding": "mechanical",
  "me_model": "water-buffer"
}
```

The finite adaptive schema in [ADAPTIVE.md](ADAPTIVE.md) supplies data_file,
model, model_sha256, type_symbols, full_region, groups, partitions, qm,
provider, charge/spin, timestep and steps. Add core_group_ids, buffer_reference,
water_guard and the following switch. Do not specify fixed occupancy:

```json
"core_group_ids": ["core-a", "core-b"],
"switch": {
  "method": "quintic-water-buffer",
  "center_atom_id": 1,
  "candidate_anchors": [["a", 7], ["b", 10]],
  "inner_radius_angstrom": 3.0,
  "outer_radius_angstrom": 5.0,
  "max_partitions": 64
},
"water_guard": {
  "min_oh_angstrom": 0.6,
  "max_oh_angstrom": 1.25,
  "min_foreign_oh_angstrom": 1.45
}
```

For these two candidates the partition group sets are core, core+a, core+b
and core+a+b. All partitions declare the same method, ground state,
energy-reference label, charge zero and multiplicity one. Their offsets are
derived as minus buffer count times delta_water. Any supplied offset or offset
source must exactly match this derivation.

Center and candidate anchors must be oxygen atoms. Every noncore group is
one declared H2O; the protected core has total H:O = 2:1. Geometry checks enforce
the declared outer water's O-H interval and minimum distance to foreign
oxygens. A core H approaching an outer O is also rejected. The sample guard
numbers above are explicit software-fixture values, not universal bond rules.
A permanently low-level water entering switching support fails as missing
coverage. No bonds, groups, charges or candidate lists are reassigned silently.

## Native records and limits

The native pair is zero and the existing callback returns the complete total
energy and force. Do not add full-system DPA a second time. Each accepted frame
records reference identity, all partition weights, buffer counts, expected
buffer occupancy and active partitions. Restart recomputes these from saved
coordinates and rejects changed policy/reference or altered occupancy records.
Failed backend or geometry checks preserve only preceding accepted checkpoints.

The initial numerical/native fixture is CPU HF/STO-3G with strict raw FP64
Nano DPA, a two-water core and one or two candidate waters. Small-system
finite differences, changing occupancy, serial/two-rank trajectories, checkpoint
recovery and explicit failure propagation are separate gates from scientific
accuracy. Numerical results belong to the identified qualification records.

The initial route is finite nonperiodic, neutral singlet, ground ME and NVE.
It does not qualify periodic/adaptive EE, selected-state exchange, charged
cores, moving reactive centers, covalent cuts, adaptive PLUMED sampling,
float32/Kokkos adaptive inference, NPT, long trajectories or model convergence.
Worst-case work grows as 2**N candidates; no large-system speedup is claimed.
