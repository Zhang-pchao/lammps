# Experimental PBE0/TDA API

`qmdpa.pbe0_tda` defines a separate CPU neutral-singlet PBE0/TDA route.
Use `PBE0TDASettings`, `PBE0TDAReference`, `build_pbe0_tda_reference`,
`PySCFPBE0TDABackend` or `PySCFPBE0TDAElectrostaticBackend` explicitly. The
provider factory and finite ME/additive configurations accept the explicit
`pbe0-tda` method with a complete inline reference. This experimental route
requires separate complete-Hamiltonian and native qualification. The gap-anchor configuration also accepts the explicit PBE0 method with
matched ground terms. Complete-Hamiltonian derivatives and method-specific native
transport have separate qualification records.

The default remains def2-SVP with grid4 and NWChem pruning, three candidate roots,
default dense dimension cap256 and explicit `grid_response=True`. An explicit
`tda_dense_max_dimension=1024` or `2048` enables a separate experimental resource
configuration for larger fixed regions or small full-system force references.
Only these three caps are accepted. The
cap is part of reference and restart identity; the solver still rejects an
actual occupied-virtual dimension above the selected cap before applying the
TDA operator. Existing cap256 and cap1024 records retain their original identity. A larger
cap does not confer derivative, native or scientific qualification on a region.
This route excludes density
fitting, NLC, undeclared functionals/basis-grid pairs, open shells, Gaussian charges,
periodic fields and GPU QM. Solver and residual settings cannot be loosened
beyond the prospective controls. It is not a general TDDFT interface.

The energy is the complete PBE0 ground energy plus the selected TDA gap.
Embedded evaluations include both QM and fixed point-charge reaction forces.
Environment self energy, DPA and hybrid assembly are absent from this API.
State selection, immutable records, source checks and exact-input reuse share
the existing coupling implementation. RHF and PBE0 use distinct references and
source pins; the original RHF reference schema and behavior remain unchanged.

## Explicit small development configuration

For short interface checks, callers may explicitly set basis="sto-3g" and
grid_level=1 together. No default changes and no automatic fallback occur.
The pair still requires PBE0, neutral singlet, three dense roots, NWChem pruning,
complete grid response and the original convergence/residual limits.

Its reference records pin the STO-3G source in addition to the existing method
sources. Legacy def2-SVP/grid4 reference hashes and runtime fields are unchanged.
Changing the basis/grid changes the state and restart identity. This is a
different electronic-structure approximation for development, not an equal-
accuracy optimization or a recommended chemical model. Its own small E/F,
finite-difference and native records determine numerical support.

This pair applies to isolated and finite point-charge selected providers,
ME/additive, matched ground terms, CPU gap-anchor and the existing CPU ground,
cross-delta and fragment-response routes. All primary and auxiliary references
must use the same profile. Existing region, one-body coverage, correspondence
and complete derivative checks remain required; no automatic conversion of
legacy references or numerical/scientific qualification occurs.

## Explicit quadrature response

Official TDRKS omits atomic integration-grid response. A previous uncorrected
candidate passed coordinate finite differences but failed its stricter total
force criterion; that failure remains evidence against the uncorrected route.
The bounded PBE0 API now adds both atomic grid-coordinate and partition-weight
terms using official PySCF quadrature derivatives and the restricted TDA XC
Lagrangian. The orbital-response residual and relaxed-density trace are checked
before the grid term or external forces are used. No force projection is used.
The official unrefined CPHF solver is retained with its explicit tighter input.

The grid-response flag is part of every new reference record. Old PBE0 records
without that explicit field are rejected; they must not be silently promoted
to corrected-force references. Existing RHF records keep their original meaning.
Diagnostics record the grid-response method and its QM gradient contribution.
Missing or failed orbital/quadrature response propagates as an evaluation error.

The analytic term has separate ground-state, atomic quadrature and collective
response gates. Actual corrected gas/embedded API, complete force derivatives,
zero field, state/reload/reuse and native transport remain distinct qualifications.
Passing an isolated helper test does not qualify another geometry or native MD.
The runtime evidence records current numerical and scientific boundaries.

## Matched ground terms for the standalone anchor

`make_pbe0_gap_anchor` in `qmdpa.gap_anchor` constructs the isolated ground,
embedded ground and embedded selected providers from one PBE0 state reference.
The ground adapters reuse the existing analytic energy/force implementation,
while explicitly matching XC, basis, grid level, NWChem pruning, no density
fitting/NLC, grid response and SCF controls. Both field terms use the same fixed
charges. The runtime pins match the selected method.

The native gap-anchor configuration dispatches explicitly to this builder and
records its PBE0 ground method. It uses pair zero and the complete total once.
Native serial/MPI/restart and actual state-loss gates are method-specific.
This builder adds no cross LJ model and does not change the separate
reference-subtractive Hamiltonian.

## Explicit official matrix assembly candidate

The optional provider execution field matrix_assembly accepts operator
(the existing default) or official-get-ab-896 for PBE0/TDA only. The latter
uses PySCF get_ab with all grid points and bounded quadrature blocks. It keeps
the original Hermiticity and operator-residual checks, state reference, complete
gradient and environment response. It is not an iterative fallback, density
fitting approximation or new functional. An explicit field enters configuration
and restart identity; absent fields preserve existing records.

This execution candidate requires its own affected tests, actual complete E/F
parity, native restart and matched complete-process timing before promotion.
Matrix assembly timings alone do not establish an MD speedup.

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
