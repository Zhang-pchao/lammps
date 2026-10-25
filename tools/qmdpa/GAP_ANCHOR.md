# Experimental finite electrostatic gap anchor

The API and native entrypoint evaluate a separate candidate Hamiltonian:

```
U_s = U_low(full) + E_0(Q;0) - U_low(Q) + E_s(Q;v_M) - E_0(Q;v_M).
```

`qmdpa.gap_anchor.GapAnchorCalculator` returns a complete total with Hamiltonian
ID `finite-electrostatic-gap-anchor`. `evaluate_terms(full_geometry)` exposes
all five terms, the ground ME baseline, the embedded excitation gap and total.
The QM and environment gap forces are both retained. Output orders are explicit;
force arrays are immutable. Units are eV and Angstrom.

The two finite, nonempty, disjoint regions cover the full system. IDs and element
symbols stay fixed. Both embedded providers must expose the same fixed point
charges, including their order and declared source. Periodic fields, Gaussian
charges, coordinate-dependent charges and adaptive partitions are unsupported.
Provider failures propagate; they do not trigger a different method or a stale
force. A change of the declared field is rejected.

`make_rhf_gap_anchor(full_low_backend, region_low_backend, reference)` constructs
the existing CPU RHF/TDA route from one frozen embedded `TDAReference`. It matches
basis, charge, spin and SCF settings. The existing ground providers implement
RKS with `xc="hf"`; actual-backend qualification must compare them with official
RHF and the selected provider's ground energy. The selected-state overlap guards
remain active. Exact-geometry reuse is opt-in for the selected provider.

For an explicit ground-state request, pass the same ground-provider object as
both embedded providers. The gap then cancels exactly after one evaluation.
With zero charges and consistent isolated/embedded quantum models, the formula
reduces to selected-state ME. These limits require tests and do not establish
chemical accuracy. Generic callers are responsible for consistent quantum terms.

This model retains the ground ME interaction and adds an environment-dependent
excitation gap. It does not repair the ground ME baseline or supply mutual
polarization, general charge-transfer accuracy, or a universal subtraction of
the electrostatic part of DPA. It is distinct from additive EE (environment DPA
plus explicit cross LJ) and fixed-charge reference-subtractive EE.

The experimental `python -m qmdpa.gap_anchor_cli CONFIG --output DIRECTORY`
entrypoint uses `pair zero` and injects the complete total exactly once. Do not
add this total to a full native DPA pair energy or pass it to a correction-only
callback. `--restart CHECKPOINT` reuses the existing accepted-frame/checkpoint
lifecycle. Selected-state records come from the same embedded force call.

The strict configuration requires `schema_version: 1`, this Hamiltonian ID,
`data_file`, a checksum-pinned compiled `.pt2` `model`, `model_sha256`, explicit
`type_symbols`, `periodic: false`, whole-molecule `qm_region` and
`environment_region`, explicit `embedding`, neutral-singlet `full_charge_spin`
and `region_charge_spin`, matched `qm` and an explicit embedded RHF/TDA or PBE0/TDA `provider`,
`timestep_fs` and `steps`. `sample_interval` and a paired `temperature_K` /
`velocity_seed` are optional. There are no LJ, reference-Coulomb or raw-model
fallback fields. All physical choices, source identities and input bytes enter
the restart identity. Region/electronic/charge-field mismatches are rejected.

Qualification belongs to one frozen source/runtime/model/reference combination.
The finite neutral-singlet RHF/TDA candidate has separate complete-potential
derivative, native serial/MPI/restart and actual state-loss evidence in a small
chromophore/water configuration. Inherited transport code never transfers another
Hamiltonian's qualification. Only point charges and one fixed selected state
are supported; short native trajectories do not establish general stability.

A response gap and a total potential answer different accuracy questions.
The companion response benchmark evaluates both selected and ground totals,
compares their gap with independent full-QM references, and reports relative
total energies and regional force errors separately. A passing excitation-gap
target does not qualify the ground anchor, total force field or chemistry.
Isolated-molecule and molecular net-force/torque diagnostics help distinguish
internal reference-method differences from interaction errors without changing
the Hamiltonian. Unsupported methods, partitions and deployments remain explicit.

## Bounded PBE0 standalone builder

`make_pbe0_gap_anchor` accepts an embedded `PBE0TDAReference` and explicitly
matches its ground/selected XC, basis, grid response, pruning, no DF/NLC and
convergence controls. See PBE0_TDA.md. The RHF builder rejects PBE0 references;
neither method inherits the other's qualification. The native configuration now accepts
an explicit embedded PBE0 reference and selects this same builder. It keeps
pair zero, complete total forces, the fixed-field observer and the existing
accepted-state/restart lifecycle. Method, runtime, quadrature and ground-anchor
identities are included in the resolved configuration. A PBE0 state cannot be
relabelled as RHF or inherit its restart/transport qualification. The bounded
PBE0 complete-Hamiltonian derivative gate has separate evidence; method-specific
native serial/MPI/restart and state-loss tests remain required.

## Optional exact ground-term reuse

PBE0 gap-anchor configurations may explicitly set reuse_ground_terms to true.
The default remains off. This reuses only a provider's last successful complete
energy and all-force result for byte-identical QM/environment coordinates,
atom IDs/order and symbols, with unchanged method, runtime and fixed charges.
Different inputs and every failure discard the saved result. Returned arrays and
diagnostics are detached. Shared declared fragment monomers/pairs use the same
provider; there is no reuse across a changed geometry or incomplete solve.

This is an execution option, not a changed Hamiltonian, SCF warm start or
previous-frame force fallback. All native setup and checkpoint operations are
retained. In particular, write_restart may migrate atoms, so this option does
not substitute run pre no. Performance claims require same-output complete
workloads with the option on and off.

The explicit STO-3G/grid1 development pair also applies to the matched PBE0
ground providers and CPU gap-anchor/fragment routes. All quantum terms share the
same basis, pruning, convergence and complete grid response. The ground terms
retain the selected profile's source pins and optional exact-geometry reuse.
Default def2-SVP/grid4 settings and older references remain unchanged. Every
primary and auxiliary state must match the same explicit profile and source
pins. Small-fixture qualification does not establish
chemical accuracy or an equal-accuracy cost advantage.

## Conventional GPU RHF route

`make_gpu_rhf_gap_anchor(full_low, region_low, reference, ...)` requires an
embedded GPUTDAReference. It reuses the same calculator with the complete
conventional GPU RHF ground and RHF/TDA selected providers. Unified finite
selected/electrostatic ee_model="gap-anchor" accepts gpu4pyscf/cuda/rhf-tda.
The native baseline stays zero; the callback delivers the total, not a correction.
GPU runtime packages and pinned state/field identity participate in restart
validation. Selected-state records retain response and relaxed-density checks.
No corrected/fragment GPU model or ground-term reuse is implied. See GPU_BACKENDS.md.
