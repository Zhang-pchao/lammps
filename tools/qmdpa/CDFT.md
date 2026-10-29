# Finite constrained-state pair (experimental)

`uks-hirshfeld-cdft` is an explicit CPU PySCF provider, separate from the
neutral ground-state and TDA providers. Its first engineering candidate uses
charge +1, multiplicity 2, dense UKS-PBE0/def2-svpd, unpruned molecular grids
with full response, and two fixed signed Hirshfeld populations -1/+1.
Atomic reference densities are neutral spherical atomic HF densities in the
same basis. This use of atomic HF does not change the molecular UKS method.

The provider calls `pyscf.dft.cdft`; PySCF owns the multiplier solve and all
QM/point-charge derivatives. The explicit inner solver is official Newton SCF
with a pinned AH subspace cutoff (`inner_solver: newton`, `inner_lindep: 1e-24`).
The inner AH residual is resolved below the strict internal orbital tolerance;
this derived accuracy control is part of the pinned solver source.
It has no fallback to unconstrained states, finite-difference production forces,
DF, NLC, ECP, periodic or GPU calculations. Missing brackets, unconverged
orbitals, changed source/library identity, electron counts or state guards fail.

`build_reference` freezes the geometry/ID/species mapping, fragment weights,
proatom basis/densities, both density/multiplier initial guesses, spin features,
method settings, actual PySCF sources and native library hashes. R and P are
separate constrained surfaces; `sampled_state` is explicit. Every call resolves
both states anew at exactly the same coordinates. Frozen guesses are not reused
energies or forces. All physical energies exclude the multiplier residual term;
that term and the constraint residual are recorded separately.

Native admission is restricted to plain finite additive NVE via
`coupling: {surface: constrained, embedding: electrostatic, ee_model: additive}`
or the equivalent additive Hamiltonian configuration. The environment DPA and
cross interaction are evaluated once per pair call and assembled by the existing
additive calculator. Native pair zero receives the complete sampled total.
Mechanical drivers, electronic bias, sampling, thermostat and execution
extensions are rejected. The same-frame `cdft_gap` value and all-atom Jacobian
are observations only; cDFT electronic sampling is a later gate.

Accepted frames contain `cdft_states`: both physical QM contributions, complete
additive totals/forces, shared environment/cross terms, constraint and spin
checks, independent Lowdin diagnostics, actual density electron traces and
solver iterations. Convergence and localization guards do not certify a unique
electronic solution or an orbital-stable/global minimum; that evidence is
separate from force and lifecycle qualification.
Their IDs, array shapes, units, geometry/reference hashes
and arithmetic are checked before checkpoint acceptance. Recomputing both
states is required at restart; failures clear temporary pair results and use
the existing collective failure/checkpoint transaction.

Portable preparation and integration checks are in the companion
`benchmarks/water/cdft` and `tests/cdft_*.py`. Qualification is bound to exact
source/reference/model/runtime evidence. This experimental interface does not
establish chemical accuracy, long MD stability, OH-/O2 chemistry, rates, free
energies or performance. Existing ground/TDA/water qualification remains bound
to its original records.
