# Finite complete-molecule additive EE

Use the same fixed core, equal-composition candidate groups, complete k-subset
family and compact normalized weights described in MOLECULAR_ADAPTIVE.md.
The electronic state is ground; candidate occupancy stays fixed.

For each partition p:
E_p = U_DPA(M_p) + E_QM(Q_p; q_Mp) + E_LJ(Q_p,M_p) + offset_p.
The dynamic energy is E = sum_p w_p E_p, with complete forces
F = sum_p w_p F_p - sum_p E_p grad(w_p).
No full-system DPA baseline or environment Coulomb energy is added.
Each existing AdditiveCalculator retains both QM and environment derivatives.

One immutable, complete atom-ID charge table supplies the exact ordered
environment subset for every partition. Charges and optional Gaussian radii
do not depend on coordinates. Changing which whole atoms enter a partition
changes its fixed field; weighting the resulting full potentials defines the
model. This is not fractional electronic embedding or mutual polarization.
Parameter provenance and compatible energy references remain the caller's
responsibility. Numerical consistency does not establish chemical accuracy.

Python entry points are AdditivePartition and MolecularAdditiveMixer in
qmdpa.molecular_additive. The shared QuinticMolecularWeights policy is reused.
Every family member and charge subset is validated before evaluation.
Exactly zero weight AND derivative permit skipping; positive weights are
never pruned. Missing coverage or any provider failure returns no total.

For the unified qmdpa command, choose:
- hamiltonian: finite-molecular-additive; or
- coupling: {surface: ground, embedding: electrostatic, ee_model: molecular-additive}.

Use molecular ME's groups/core_group_ids/partitions/switch/full_region fields.
Replace full_charge_spin and region_charge_spin with environment_charge_spin.
Supply embedding with atom_ids, charges_e, source, and optional radii_angstrom
for EVERY full-region atom. Supply the same explicit cross LJ declaration as
the fixed additive driver. Every possible directional element pair is required.
The initial native route supports CPU PySCF, finite fixed box and neutral
singlet QM/environment conditioning. Model, complete field, cross parameters,
partition family, offsets and policy enter the restart identity.

The native pair is zero and the callback supplies the complete total.
This route does not supply periodic EE, NPT/virial, reactive group reassignment,
variable occupancy, excited-state partition correspondence or NAC/SOC.
Small-system qualification and actual runtime costs must be recorded separately
from claims about long trajectories, chemistry or general speedup.
