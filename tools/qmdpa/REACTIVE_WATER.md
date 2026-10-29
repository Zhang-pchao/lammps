# Finite reactive-water ownership and QM adaptation

The ground mechanical and additive electrostatic models declare a small proton-transfer network
before a run. It supports two to four pool waters and one or two distinct
transferable hydrogens with explicit initial donor and acceptor groups. Other
hydrogens keep their original oxygen assignment. The complete family is fixed;
there is no nearest-oxygen hard relabeling, truncation or dynamically generated
partition. This is a bounded software model with independent numerical and
scientific qualification, not a general reactive solvent model.

## Integer states and continuous support

Every binary transfer topology has an exact hydrogen list for every oxygen.
The formal group charge is n_H-2. All nonzero-charge groups and one fixed seed
water are mandatory QM groups. Each QM fragment has neutral water stoichiometry,
charge zero, 10*n_O electrons and multiplicity one. Every DPA-only group remains
a complete neutral water. These formal charges are bookkeeping, not calculated
atomic populations or localized electron counts. The QM solver determines the
electronic distribution; a soft Voronoi CV is never rounded into a charge.

For each channel define z = r(donor,H)-r(acceptor,H). A compact quintic ownership
probability alpha rises from zero to one for -a < z < a. A separate compact
envelope is one for |z| <= a and becomes zero at |z| >= b, where b > a. Both
channel endpoints remain wholly QM whenever ownership is mixed. The envelope
allows an endpoint to leave QM only after the proton ownership has settled.

For each optional neutral group i, combine its radial preference with incident
transfer envelopes through p_i = 1-(1-p_radial_i)*product(1-e_h). Enumerate every
subset of optional groups in every topology. A partition weight is the product
of its topology Bernoulli factors and its subset Bernoulli factors. All O/H
coordinate derivatives are retained, with sum(W)=1 and sum(grad(W))=0. Exactly
zero weight and derivative allow skipping a fragment. No finite weight is cut.
Identical fragments may share an evaluation only within the same geometry call.

## Hamiltonian and reference

```text
delta_water = E_QM(reference water) - E_DPA(the same reference water)
C_p = E_QM(Q_p) - E_DPA(Q_p) - (n_O(Q_p)-1)*delta_water
U = U_DPA(full) + sum(W_p*C_p)
F = F_DPA(full) + sum(W_p*(F_QM(Q_p)-F_DPA(Q_p))) - sum(C_p*grad(W_p))
```

The native pair is zero and the callback returns the complete potential and
all atomic forces. The full DPA energy must not be added again. This assembly
reuses the existing subtraction and conservative molecular mixer. At all-QM
occupancy, force equals full QM and energy equals full QM minus the declared
(n_O(full)-1)*delta_water constant. The reference is part of the model, not a
thermodynamic chemical potential or evidence of buffer convergence.

## Reactive additive electrostatic embedding

Select `hamiltonian: finite-reactive-water-additive` or
`coupling: {surface: ground, embedding: electrostatic, ee_model: reactive-water-additive}`.
The same complete proton topology, weights and domain guards are used:

```text
E_p = E_DPA(M_p) + E_QM_emb(Q_p; q_Mp) + E_cross(Q_p,M_p)
      - (n_O(Q_p)-1)*delta_water
U = sum(W_p*E_p)
F = sum(W_p*F_p) - sum(E_p*grad(W_p))
```

The native pair remains zero. The callback returns the complete potential;
there is no full-system DPA term. Use `environment_charge_spin: [0,1]` instead
of the mechanical full/region conditions. Supply one full-system `embedding`
atom-ID charge table and explicit `cross` parameters in the fixed additive
format. Every environment subset uses the same ordered charge table, and
every reassigned DPA-only water must have zero summed fixed charge. At least
one permanent environment water is required, so this profile has no all-QM
endpoint. All ionic defects and partially transferring endpoints remain QM.

The isolated-water calibration uses the same raw model and common QM settings.
Only the supported embedded ground solver is admitted; the isolated-only
`diis-newton` option is rejected before backend construction. Environment
reaction forces, cross forces and complete weight derivatives are all included.
This route is distinct from reference-subtractive EE and does not inherit its
qualification. A supported combination still requires its recorded tests.

## Explicit input and accepted records

Select `hamiltonian: finite-reactive-water-mechanical` or the unified selector
`coupling: {surface: ground, embedding: mechanical, me_model: reactive-water}`.
Declare `full_region`, initial neutral `groups`, `reactive_region`, `water_guard`,
`buffer_reference`, and `inference_profile: torch-raw-deterministic-v1`. Use the
original raw FP64 `.pt` model and explicit [0,1] full/fragment DPA states. The
initial molecule list must cover every atom exactly. Partitions are generated
from the complete declared topology and subset family, not supplied by users.

`reactive_region` requires `pool_group_ids`, `seed_group_id`, `channels`,
`inner_radius_angstrom`, `outer_radius_angstrom`, `transfer_inner_angstrom` and
`transfer_outer_angstrom`. Each channel has `hydrogen_id`, `donor_group_id` and
`acceptor_group_id`. Optional `max_partitions` caps the complete family at 64;
exceeding it rejects the request instead of dropping partitions.

The guard requires explicit minimum/maximum allowed O-H distances and a minimum
foreign O-H distance. It rejects an undeclared proton acceptor, a hydrogen that
leaves its allowed domain, or a permanent environment water entering the radial
support. Distances are case inputs, not universal bond definitions.

Prepare the isolated-water reference with `qmdpa prepare-water-reference` using
the identical model, QM settings, inference policy, source and dependencies.
Never edit its source hash to reuse an incompatible calibration. Optional CPU
`qm.scf_level_shift_hartree` is an explicit SCF convergence aid using the official
PySCF control. It changes neither the reported physical energy expression nor
its gradient. It is frozen in reference and restart identity; there is no
automatic retry, alternate solver or previous-frame fallback. The PySCF
convergence flag and complete E/F gates are still required. This setting is not
admitted by the GPU or selected-state adapters. Optional `qm.scf_solver:
diis-newton` declares two fixed stages on every call: the ordinary DIIS solver,
then official Newton refinement starting from its orbitals and occupations.
Each stage has the declared `max_cycle` bound. Refinement uses ah_lindep=1e-20,
ah_conv_tol=min(1e-24,(0.1*conv_tol_grad)^2) and
ah_start_tol=min(1e-10,0.1*conv_tol_grad). Final convergence and a freshly computed
orbital-gradient norm no larger than the explicit conv_tol_grad are required.
There is no automatic fallback from the default solver. This refined protocol
is limited to isolated CPU ground calculations.

Accepted frames retain ownership probabilities, channel distances, every
partition weight, integer membership, group formal charges, QM atoms, charge,
electron count and multiplicity. Restart recomputes these from the accepted
geometry and bound reference before native propagation. It rejects changed
policies or corrupted bookkeeping. Mixed ownership is reported as an ensemble
of explicit integer topologies, not one arbitrarily selected hard label.

## Optional NVT and reactive OPES

The unified driver accepts the same explicit native Nose-Hoover chain
thermostat as fixed-region ground ME. With bias, only the official OPES and
classical water Voronoi CV pair are admitted. Select all oxygen/hydrogen atom
IDs in the declared reactive pool, retaining them for the entire trajectory.
Do not rebuild the CV selection from instantaneous QM membership or round
its smooth defects into an electronic charge. Neutral pool water may leave
QM, while every bias derivative on it still acts on the complete system.

The total potential at fixed OPES history is U + B(CV;history), with every
derivative of U, the CVs and B retained. Ordinary NVE energy conservation is
not a valid screen for a time-dependent bias. Use fixed-history complete
finite differences, segmentation/observation-cadence parity and full STATE
continuation together with the reactive integer-state record. Live SCF or
ownership failures must preserve the last accepted native binary, JSON and
STATE, and may not promote a later frame. A new Hamiltonian, source, reference,
thermostat, CV selection or bias setting requires a new restart identity.

See [sampling settings](SAMPLING.md) for units and exact-state requirements.
Distance restraints and WTMetaD on this reactive model are not admitted.
A running implementation is not a combined qualification record; require
matched real-backend, migration, MPI, restart and force evidence.

## Current limits

Both routes retain finite neutral CPU ground QM. ME uses strict raw inference.
Additive EE also admits the explicit native environment-domain profile in
[execution settings](EXECUTION.md), with a separately calibrated compressed
reference. The outer pair stays zero, including under OPES.
Periodic reactive switching, reference-subtractive
reactive EE, charged total systems, unlisted
protons or water networks, general covalent cuts, selected-state mapping and
NPT are outside this model. Each biased sampling combination requires its
own recorded numerical and lifecycle qualification.
Software E/F, native lifecycle and short controlled crossings do not establish
spontaneous reaction kinetics, free-energy convergence, long-term stability,
chemical accuracy or a performance advantage.
