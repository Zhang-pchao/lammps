# Finite dynamic exchange of complete molecules

The molecular route extends ground-state ME from two partitions to an explicit
family of complete-molecule choices. A fixed core retains exactly k candidate
molecules. Candidates have the same elemental composition. The input must
declare every k-subset of the candidate pool; missing choices are errors.

For candidate i at anchor distance r_i from the core anchor, f_i is one within
the inner radius, a decreasing quintic through the switching shell, and zero
at or beyond the outer radius. For each subset p with exactly k members,

```text
v_p = product(f_i for i in p)
w_p = v_p / sum(v_q for every declared k-subset q)
Delta_p = E_QM(Q_p) - E_DPA(Q_p) + offset_p
E = E_DPA(full) + sum(w_p * Delta_p)
F = F_DPA(full) + sum(w_p * F_Delta_p) - sum(Delta_p * grad(w_p))
```

Both the radial and normalization derivatives are included, including forces
on the core anchor. The compact quintic has zero first and second derivatives
at its endpoints. A partition is skipped only when its weight and every weight
derivative are exactly zero. Small positive weights are never discarded.
Re-entering candidates are evaluated afresh. At least k candidates must remain
inside the outer radius. Loss of coverage stops the calculation.

This is a fixed-occupancy model, not a variable-size QM region. The molecules
occupying that region can change continuously through the mixture. All
partitions share charge, multiplicity, method and energy-reference identities.
Offsets are explicit and affect switching forces. They are never fitted or
changed during a trajectory. The caller declares groups and anchors; bonds,
protons and electronic states are not reassigned by geometry heuristics.
Finite nonperiodic, neutral singlet CPU PySCF ground states are the initial
native scope. Chemical accuracy, general reaction applications and larger
candidate families require separate evidence.

## Unified configuration

Use qmdpa validate and qmdpa run with an explicit
hamiltonian of finite-molecular-mechanical, or the equivalent selector:

```json
"coupling": {
  "surface": "ground",
  "embedding": "mechanical",
  "me_model": "molecular"
}
```

The existing adaptive configuration fields in [ADAPTIVE.md](ADAPTIVE.md) apply,
with core_group_ids and a different switch. For three candidates and k=1:

```json
"core_group_ids": ["core"],
"switch": {
  "method": "quintic-molecular-occupancy",
  "center_atom_id": 1,
  "candidate_anchors": [["a", 3], ["b", 5], ["c", 7]],
  "occupancy": 1,
  "inner_radius_angstrom": 2.0,
  "outer_radius_angstrom": 3.0,
  "max_partitions": 64
}
```

Each anchor must belong to its named complete group. In this example declare
three partitions, with qm_group_ids [core,a], [core,b] and [core,c].
For k=2 all three pairs of candidates are required instead. The family size
is combinatorial and is checked against the explicit maximum before allocating
providers. This first implementation prioritizes a reproducible complete
family over a large-scale adaptive selection optimization.

The native pair is zero; the callback returns the complete total potential and
all forces. No full-DPA pair energy is added again. The existing MPI mapping,
accepted-frame lifecycle, collective failure handling and binary restart
checks are reused. The physics identity includes the entire candidate family,
core, offsets, radii, occupancy and source/model/provider identities.
Candidate weights are recomputed from restart coordinates, with no hidden
membership history. Legacy two-partition inputs retain their separate model.

The Python API is MolecularMechanicalMixer with QuinticMolecularWeights in
qmdpa.molecular_adaptive. evaluate_terms exposes the weights, all derivatives,
active partition IDs, switching force, baseline, full correction and total.
Inactive partition corrections are None, never fabricated zero evaluations.
The current ledger records numerical and native qualifications for each source
snapshot. No selected-state exchange, variable occupancy, dynamic EE, periodic
electrostatics, stress or reactive regrouping is implied by this ground-ME API.
