# Finite selected-state molecular adaptation

This development model follows one localized RHF/TDA excitation while complete
environment molecules exchange their QM weights. It has an intact fixed
chromophore core, exactly k same-composition candidate molecules and every
possible k-subset. It is a separate model from the ground-state molecular
families, reactive-water reassignment and fixed-region two-state Ehrenfest.

## Energy and force

For partition a, define a complete neutral-singlet correction

    Delta_a = E_selected,a(Q_a) - E_DPA(Q_a)
    U = E_DPA(full) + sum_a w_a Delta_a
    F = F_DPA(full) + sum_a w_a (F_selected,a - F_DPA(Q_a))
                      - sum_a Delta_a grad(w_a).

All Cartesian derivatives of the existing compact quintic fixed-occupancy
weights are retained, including the center anchor and normalization. Full and
fragment DPA use the same raw model and neutral-singlet conditioning. No fitted
energy offsets, truncated positive weights or approximate active-family pruning
are allowed. Exactly zero weight and zero derivative permit skipping a partition.

Native LAMMPS uses pair_style zero and receives this complete potential and
force through the existing all-atom callback. Adding another full-DPA term
would count the low-level baseline twice.

## State correspondence

Preparation computes a core-only transition-density reference. Each complete
QM partition is selected by normalized cross-AO transition-density overlap to
that common excitation and receives its own immutable reference.

Every evaluation must satisfy all three checks:

1. The partition's own fixed reference selects an admitted root.
2. The common core reference selects the same root.
3. For every directed pair of active partitions, the selected transition on the
   source side identifies the selected root among all requested target roots.

Overlap, separation from the next-best overlap and candidate energy gaps are
explicit settings. A root number or textual label alone does not establish a
common excitation. The metric compares transition-density operators across AO
bases. It is invariant to transition phase sign, but is not a signed
many-electron overlap, nonadiabatic coupling or electronic population.

Matching is an admission check; it is not an additional blending weight.
Within an admitted nondegenerate branch, the existing complete state gradient
and the explicit weight derivatives define the force. Loss of core character,
ambiguous roots, small candidate gaps or failed response stops the step.
The gap check covers the requested roots, not all possible electronic states.
Large displacements or character changes may leave this fixed-reference domain.

## Preparation and run configuration

Prepare a fresh reference file in the intended scientific allocation:

    qmdpa prepare-state-family REQUEST.json --output FAMILY.json

The request contains exactly these keys:

- schema_version: integer 1.
- geometry: atom_ids, symbols and positions_angstrom for the full finite system.
- core_atom_ids: all atoms of the intact fixed core.
- partitions: objects with partition_id and the complete QM atom_ids.
- settings: explicit CPU TDASettings controls; native runs require tda_solver=dense.
- initial_root: the one-based core excitation used to seed preparation.
- label: a descriptive state label.

Preparation refuses to overwrite an existing file. Its output identifies both
the file SHA256 and the canonical family SHA256. Derivative and dynamics
qualification remains separate from successful preparation.

The run configuration reuses the groups, core_group_ids, complete partitions,
quintic-molecular-occupancy switch, full_region, paths, model hash, timestep and
step count documented in MOLECULAR_ADAPTIVE.md, with these explicit changes:

    "hamiltonian": "finite-molecular-selected-mechanical"
    "provider": {
      "engine": "pyscf", "device": "cpu",
      "state": {"method": "common-core-rhf-tda"}
    }
    "family_reference": {"path": "FAMILY.json", "sha256": "<file SHA256>"}

Each partition declares electronic_state=common-core-rhf-tda,
method_identity=cpu-rhf-tda:<canonical family SHA256>, and
energy_reference=<canonical family SHA256>. Omit offsets or supply only
energy_offset_ev=0 and offset_source=null. The qm settings must match the
prepared family. full_charge_spin and region_charge_spin are both [0, 1].

The optional equivalent selector is

    "coupling": {
      "surface": "selected", "embedding": "mechanical",
      "me_model": "molecular-common-core"
    }

Use the common commands:

    qmdpa validate CONFIG.json
    qmdpa run CONFIG.json --output FRESH_DIRECTORY
    qmdpa run CONFIG.json --output NEW_DIRECTORY --restart CHECKPOINT.json

Configuration validation imports no electronic engine. Actual preparation and
dynamics require the pinned RHF/TDA runtime, including its CPHF response hook,
and the explicit torch-raw-deterministic-v1 inference policy.

## Accepted frames and restart

Every accepted frame stores a checksummed selected_family record: immutable
reference identities, all partition memberships, current weights, active state
diagnostics, every directed active correspondence, complete potential/force and
switching forces. The validator reconstructs weights from the actual frame and
checks every state-selection and response gate.

A restart first validates stored evidence and native binary identity, then
recomputes the complete potential and state correspondence. Membership and roots
must agree; the common native energy/force and coordinate/velocity tolerances
also apply. This model propagates no electronic amplitudes or adaptive history.
A failed callback cannot publish its stale state or promote a failed step.
Recovery uses only the immutable last accepted checkpoint.

## Scope and evidence

Initial scope is finite, neutral-singlet CPU RHF/TDA, fixed intact core,
complete same-composition fixed-k families, strict raw DPA and plain native NVE.
The family reference admits two to 64 partitions, subject also to the molecular
switch's complete-family limit. No covalent cuts, proton reassignment,
variable occupancy, periodic or electrostatic embedding, GPU QM, enhanced
sampling or adaptive nonadiabatic dynamics are inferred.

Small-system force differences and short native trajectories test software
consistency. They do not establish region-size independence, electronic
accuracy, chemical dynamics, long-time stability or a performance advantage.
The companion qualification ledger owns exact tested source/runtime identities.
