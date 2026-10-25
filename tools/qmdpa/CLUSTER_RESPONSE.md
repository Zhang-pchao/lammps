# Fragment-response candidates

PairInteractionSet sums existing explicit cross-interaction differences over
predeclared non-overlapping atomic-pair coverage. Shared fragment nodes are
allowed; duplicate or partially overlapping edge coverage is rejected.
It does not add switching, periodic interactions or three-body corrections.

ClusterGapDelta defines DeltaOmega = Omega_H(C+D;0) - Omega_H(C;v_D).
Each Omega is formed from separate total selected and ground energies. The
correction contains every pair, core and environment coordinate derivative,
and applies only to the selected surface. It tends to zero when the explicit
fragment and its field are removed consistently. That limiting behavior must
still be verified for the chosen backends and state references.

These standalone components were motivated by observed ground environment-pair
and state-specific embedding residuals. No DPA electrostatic/exchange/induction/
dispersion decomposition is assumed, and no parameter has been fitted.
Caller obligations include matched quantum levels, complete molecules, fixed
charges and independently verified state correspondence for both selected
providers. The native route below records all selected providers. Actual quantum derivatives,
prospective scientific accuracy and dynamics qualification remain separate gates.
The existing native Hamiltonians do not activate these candidates implicitly.


## Explicit one-body alignment

OneBodyAlignedPairSet adds sum_D [E0_H(D)-U_L(D)] for declared complete,
non-overlapping monomers. Each (pair_index, side) selects the exact monomer
already evaluated by that pair, where side is core or fragment. Shared
monomers are selected once; overlapping selections are rejected. The result
retains the original pair ledger and separate high/low one-body terms, returns
complete conservative forces in the input atom order, and has its own model
identity. It makes no additional electronic or DPA calls and holds no
cross-geometry cache.

This is an optional change to the Hamiltonian, not a fitted energy offset:
its internal molecular derivatives are included. Add the same correction to
both electronic surfaces to retain their vertical gap. The caller must ensure
that each selected monomer is still represented at the low level in the
baseline; aligning an already high-level monomer would count its replacement
twice. Neither method compatibility nor whole-molecule membership can be
inferred from atom IDs alone.

For C (core), B (QM buffer), D (environment), the pair-set ground baseline
U_L(C+B+D)-U_L(C+B)+E0_H(C+B)+DeltaI(C,D)+DeltaI(B,D)
with D aligned has ground error I3_L(C,B,D)-I3_H(C,B,D) relative to
the full high-level method when all component identities match. This is an
algebraic diagnostic, not a guarantee that three-body errors are small.
No native, reference-convergence, dynamics or scientific qualification follows
from this standalone API.

## Explicit optional native route

The unified entrypoint accepts surface selected, embedding electrostatic and
ee_model gap-anchor-fragment-response, or the exact Hamiltonian
finite-electrostatic-gap-anchor-fragment-response. It reuses pair zero and the
existing complete-total driver. It currently requires matched PBE0 neutral
singlets, finite boundaries and fixed complete molecules.

The fragment_response object declares model finite-fragment-response-v1,
a nonempty pairs list (each uses the existing interaction_delta schema),
an explicit alignments list (empty means no one-body replacement), and
cluster_gap. Each native alignment selects pair_index and side fragment.
Aligning a QM core or replacing an environment monomer twice is rejected.
No cutoff, automatic fragment selection or fitted offset is introduced.

cluster_gap declares model finite-cluster-minus-field-gap-v1, core_region,
fragment_region, pair_provider and field_provider. Both providers use the
existing inline selected-state schema. The pair uses core followed by fragment
in isolation; the field uses core with the fragment's exact subset of the main
fixed charges. All quantum settings match the anchor except for the explicit
region-size matrix capacity. State correspondence requires source evidence and
anchor_sha256, pair_sha256 and field_sha256. This declaration pins a user's
state assignment; it does not itself prove physical correspondence.

The selected total is the original gap anchor plus declared pair differences,
optional environment one-body differences and the cluster-minus-field gap.
All terms return complete conservative forces through their existing evaluators.
The ordinary electronic_state record remains unchanged. Every accepted frame
and checkpoint also contains auxiliary_electronic_states with cluster_pair and
cluster_field records from the same successful callback. Missing, stale or
foreign records prevent acceptance; restart recomputes and compares each state
identity and each auxiliary provider energy. No report triggers another solve.

This optional route does not activate in existing configurations. It does not
provide mutual polarization or multistate
dynamics or an automatic accuracy guarantee. Numerical, application and complete
workload costs must be reported for the selected configuration.

## Ground-surface native route

The unified selector accepts surface ground, embedding mechanical and
me_model fragment-response, or Hamiltonian finite-mechanical-fragment-response.
This evaluates the same ground subtractive baseline, declared pair differences
and optional environment one-body alignment. It uses pair zero plus the
complete total E/F transport; the full DPA term is already in that total.

Declare qm_region, environment_region, explicit neutral-singlet full/region
conditions and matched PBE0 ground settings (default def2-SVP/grid4 or the
explicit STO-3G/grid1 development pair). The ground-only
fragment_interactions object contains model finite-ground-fragment-interactions-v1,
pairs and alignments with the same coverage rules described above. It does not
accept embedding, cluster_gap or selected-state references. The shared ground
provider pins the existing PBE0 runtime; no TDA solve is performed.

Exact reuse_ground_terms is optional and defaults to false. Reuse requires
identical complete inputs and never returns stale or partial results. Ground
frames contain no electronic-state records; all force, failure, atom-order and
restart checks use the existing driver. This interface exposes an already
defined ground surface and is not a new physical correction or accuracy claim.
Actual-backend and native qualification must be reported for this route.

## Optional reduction of identical signed terms

Set reduce_fragment_terms to true together with reuse_ground_terms for the
selected fragment route. The complete Hamiltonian is unchanged: identical
provider objects on identical ordered regions are collected before evaluation,
and zero-weight ground/low terms are omitted. Settings equivalence alone does
not authorize this reduction. All three selected providers still run and retain
their full state/failure records. Fixed fields, atoms, elements and boundaries
are validated before evaluation. No additional cross-geometry cache is used.

The standalone ReducedFragmentResponseCalculator exposes last_execution counts
after a successful total evaluation. Its evaluate_terms method retains the
expanded diagnostic ledger and may perform otherwise cancelled calculations.
The native callback uses the reduced total. The option defaults to false;
actual complete E/F, native and full-workload cost qualification are separate.

## Small development profile

The explicit STO-3G/grid1 pair is also available for the native ground and
selected fragment routes. It retains complete moving-grid response and every
original convergence, field, region, state and restart check. All three selected
references must carry matched profile/runtime identities; a legacy auxiliary
reference cannot be mixed into a small primary calculation. Prepare each
reference explicitly. Optional exact reuse and signed-term reduction keep their
existing contracts. This is a different approximation for quick interface
checks, with separate numerical evidence, not a chemical accuracy guarantee.
