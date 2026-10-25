# Experimental CPU RHF/TDA provider

This module implements an experimental finite RHF/TDA provider. The explicit
CPU dense-solver combination is now registered by the native mechanical and
additive provider factory. Both routes passed a bounded 2 fs native engineering
gate on a frozen source revision. Native embedding
accepts point charges only; Gaussian references remain standalone experimental
API features pending a separate real derivative gate. Registration does
not establish acceptance for production excited-state dynamics. Source-matched
energy/force qualification, native integration and restart checks, and physical
validation are separate gates. See PROVIDERS.md for the exact input contract.

The separate [PBE0/TDA candidate](PBE0_TDA.md) has its own settings, reference
identity and runtime pins. It is not selected by passing a non-HF functional
to this RHF API.

## Supported electronic problem

The supported problem is a finite, closed-shell RHF molecule with singlet TDA,
a named orbital basis, real orbitals, and fixed atom identity and ordering.
Optional external point charges or Gaussian charges have fixed values and
radii. Both QM and external-charge coordinates can move. The external charges
are applied to RHF before constructing TDA, following the official PySCF route.

The provider returns the total excited energy

    E_excited(R_Q, R_M) = E_RHF(R_Q, R_M) + omega(R_Q, R_M),

in eV, and negative total derivatives in eV/Angstrom. It never returns only the
excitation gap as the potential energy. Environment self energy and
short-range interactions are absent. This provider alone is not a complete
QM/MM or QM/DPA potential. A separate energy-combination model must explicitly
account for those terms and any reference subtraction.

The first implementation rejects non-HF functionals and non-CPU devices.
It does not imply support for density fitting, frozen orbitals, open shells,
triplet excitation, periodic electrostatics, ECPs, adaptive membership,
coordinate-dependent charges, mutual polarization, nonadiabatic transitions,
or excited-state surface hopping. It does not assess RHF orbital stability.

## API

    from qmdpa.excited import (
        TDASettings,
        TDAReference,
        build_tda_reference,
        PySCFTDABackend,
        PySCFTDAElectrostaticBackend,
    )

    settings = TDASettings(
        xc="hf",
        basis="def2-svp",
        nstates=3,
        min_overlap=0.6,
        min_margin=0.1,
        min_gap_ev=0.001,
        tda_solver="dense",
        tda_dense_max_dimension=256,
        threads=1,
    )
    reference = build_tda_reference(
        qm_geometry, settings, initial_root=1, label="declared reference",
    )
    isolated = PySCFTDABackend(settings, reference)
    result = isolated.evaluate(qm_geometry)
    surface = isolated.evaluate_surface(qm_geometry, forces=False)

For an embedded reference, provide environment_geometry and an explicit
FixedEmbeddingCharges object to build_tda_reference, then construct

    embedded = PySCFTDAElectrostaticBackend(settings, reference, charges)
    result = embedded.evaluate(qm_geometry, environment_geometry)
    surface = embedded.evaluate_surface(
        qm_geometry, environment_geometry, forces=False,
    )

The isolated evaluate method returns Evaluation; the embedded method returns
EmbeddedEvaluation, including state diagnostics. evaluate_surface always
returns TDASurfaceEvaluation. An energy-only evaluation returns None for both
force fields. No unavailable derivative is represented by an array of zeros.

Reference records are ordinary JSON data:

    record = reference.to_record()
    restored = TDAReference.from_record(record)

The reference record includes the fixed geometry, transition density, complete
settings, seed-state identity, embedding charges and their provenance, version,
relative source hashes, and canonical SHA256. Loading checks both the checksum
and the supported identity. It does not deserialize executable objects.
Record persistence is the caller's responsibility. A checksum detects
accidental changes; it is not a certificate of physical validity.

## Explicit eigensolver and independent residual guard

The default tda_solver="iterative" retains the official iterative TDA solver
and its default linear-dependence setting. An unconverged root remains an
error. There is no automatic retry, tolerance relaxation, or dense fallback.
Lowering the linear-dependence cutoff is not used as a convergence repair.

The explicit tda_solver="dense" option is limited to small QM regions.
The occupied-virtual dimension must not exceed tda_dense_max_dimension,
which defaults to 256 and cannot exceed the hard resource cap of 1024.
Larger requests need their own source-matched derivative and native qualification;
this resource ceiling does not relax any convergence or state criterion.
Official gen_vind operator actions
on unit vectors assemble matrix columns in batches of at most 16. Shape,
finite values and maximum Hermiticity error are checked before diagonalizing.
A maximum asymmetry of 1e-12 Hartree is permitted for roundoff only, then
(A + A.T)/2 is diagonalized. The lowest requested roots above the official
positive_eig_threshold are retained, and eigenvectors are converted to the
restricted TDA convention X = v/sqrt(2). This changes the eigensolver, not
the RHF/TDA Hamiltonian, basis or physical model.

Both routes independently apply the original operator to every candidate
and require

    ||A X - omega X||_2 / ||X||_2 < tda_conv_tol.

The normalization compensates for restricted X having norm 1/sqrt(2).
The original Hartree residual tolerance is unchanged. Solver flags cannot
override this check. Dense matrix size, cap, batch size, Hermiticity error,
positive-root threshold and per-root residuals are recorded in diagnostics.
The screen requests dense explicitly and its independent reference assembles
A with official get_ab, separately diagonalizes it, and checks its eigenpairs
against gen_vind. A different solver label or missing residual evidence fails
that screen's qualification gate.

## Provider-result energy and derivative contract

TDASurfaceEvaluation keeps its existing constructor fields and does not change
reference schema 1 or native electronic-state record schema 1. The result now
validates its own same-call ledger, including when used outside native dynamics:

- method must be cpu-rhf-tda-singlet and energy_definition must be E_RHF+omega;
  the SCF and candidate-root convergence flags must both be true.
- ground_energy_ev is a finite scalar and excitation_energy_ev is the positive
  candidate energy at the one-based selected_root. Candidate energies must be a
  finite positive vector with at least two entries.
- energy_ev equals ground_energy_ev + excitation_energy_ev within the existing
  1e-7 eV native-record bookkeeping tolerance. This is not a scientific accuracy
  threshold, and the provider constructs the same sum without changing it.
- QM IDs are nonempty, unique positive integers. Environment IDs are empty for
  an isolated result or unique positive integers disjoint from the QM IDs.
- Embedded force evaluations contain both complete QM and environment arrays,
  in the declared atom order. Energy-only evaluations contain neither array.
  Isolated results never contain environment forces. forces_evaluated agrees
  with the actual fields; absent derivatives are never replaced by zeros.

The ground_energy_ev and excitation_energy_ev read-only properties expose the
validated components without requiring a caller to interpret the diagnostics.
The ground component belongs to the same geometry and embedding field; it is
not an isolated ground-state reference or a gap-anchor subtraction term.
Contract-critical energy scalars, candidate energies, atom IDs and force arrays
are detached from caller-owned mutable inputs. Additional diagnostic metadata
retains the existing shallow mapping interface.

This check does not select or relabel a root, alter the immutable-reference
tracking policy, or qualify a Hamiltonian, electronic method, environment model
or derivative. Native state-record validation still checks its separate
reference, geometry, solver and response evidence.

## State selection and reproducibility

For every evaluation, all requested candidate roots must converge with positive
excitation energies. At least two candidate states are required. The transition
matrix for each restricted TDA state is

    T = C_occ X C_vir^T,       2 ||X||^2 = 1.

For a reference A and candidate B, the overlap score uses the cross-AO overlap
S_AB between their geometries:

    numerator = abs Tr[T_A^T S_AB T_B S_AB^T]
    norm_A^2  = Tr[T_A^T S_AA T_A S_AA]
    norm_B^2  = Tr[T_B^T S_BB T_B S_BB]
    score     = numerator / sqrt(norm_A^2 norm_B^2).

The selected state must pass the minimum score, the top-two score margin, and
the nearest excitation-energy gap among the requested candidate roots.
Degeneracy, ambiguity, a lost candidate root or an unconverged state is a hard
error. Increasing nstates may reveal a nearby state outside a smaller
candidate set; the reported gap is not a bound on the entire spectrum.

This criterion follows transition-density character. It is not an exact
many-electron wavefunction overlap or proof of global adiabatic continuity.
It is invariant to the global sign of X and uses moving AO centers. Within a
smooth, unambiguous branch, analytic derivatives refer to the selected total
energy surface; there is no derivative of an overlap weighting function.

Selection always refers to the same immutable reference. There is no hidden
previous-frame state, no scanner state update, and no force reuse after failure.
Reordered calls, finite differences and restarts therefore use identical state
selection inputs. Large displacements can fail the overlap guard. Creating a
new reference changes the declared selection problem and must be an explicit
operation, not an automatic fallback.

## Experimental external-charge derivative

The official RHF/TDA total QM gradient formula is retained. A small local
subclass uses the project fork's instance solve_cphf hook before any
response-dependent gradient intermediates are constructed. The hook's default
implementation delegates to the unchanged official solver. The subclass also
uses extra_force to read the resulting relaxed density and returns its
unchanged base contribution. No global PySCF functions are patched and no
full gradient implementation is copied.

In the inspected source, the one-electron derivative contracts with

    P_relaxed = 2 * oo0 + sym(dmz1doo),
    sym(A) = (A + A^T) / 2.

For displacing an external charge, QM AO centers, overlap and two-electron
integrals do not explicitly move. The candidate MM derivative is therefore

    G_MM = grad_hcore_mm(P_relaxed) + grad_nuc_mm().

Using the ground-state density here would omit excited-state orbital response.
The module checks the relaxed-density shape, finiteness, symmetry, electron
count, and consistency across per-atom hooks. It independently checks the
maximum residual of the CPHF equation

    (epsilon_a - epsilon_i) z_ai + fvind(z)_ai + w_ai = 0.

Both isolated and embedded analytic routes use this response check. The
response_residual_tol is an absolute maximum residual in the atomic-unit
CPHF equation. It is distinct from the iterative solver tolerance.

The inspected Krylov solve uses a linear-dependence floor
max(tol**2, machine_epsilon). Tightening its tolerance alone can leave a
non-negligible equation residual. The instance hook first calls the default
solver, then uses at most two normalized defect corrections if the original
residual guard is not met:

    A(z) = (epsilon_a - epsilon_i) z + fvind(z)
    r = A(z) + w
    s = max(abs(r))
    A(u) = -r/s                 # same default solver, normalized right-hand side
    z_new = z + s*u

Every correction must be finite and strictly decrease the maximum residual.
The original residual tolerance remains unchanged. Failure after two
corrections stops the calculation. The checked response is returned before
z1ao, veff, im0 and dmz1doo are built, so QM and MM derivatives use the same
corrected response. Correcting only the density afterwards would not fix
the QM gradient and is not permitted.

Diagnostics record the residual before and after correction, the residual
history, the number of refinements, and the unchanged tolerance. The
extra_force hook independently checks the final equation residual and verifies
that the gradient used the checked response. The additional response
contraction is performed once per gradient, not once per QM atom.

The implementation is pinned to an inspected PySCF 2.14.0-based fork with the
instance solve_cphf hook, and exact SHA256 values for grad/tdrhf.py,
grad/rhf.py, scf/cphf.py, qmmm/itrf.py, tdscf/rhf.py and tdscf/_lr_eig.py. The stock 2.14.0 source does not
have this new instance hook and intentionally fails the pin. A source
mismatch fails before SCF execution. Requalifying another version requires
source review and numerical tests; removing the pin is not a valid upgrade
path.

DFT TDA gradients are outside this first implementation. In particular, NLC
and numerical-grid response must not be silently omitted, and another
functional must not be substituted to make a calculation run.

## Numerical qualification required

The synthetic tests exercise bookkeeping, overlap metrics, state identity,
serialization, call order, convergence rejection, relaxed-density extraction,
and unit/sign conventions. They also verify defect-correction sign/scale,
bounded iterations, strict residual decrease, failure on nonfinite response,
and use of the corrected response before gradient construction. Dense tests
cover explicit solver selection, the hard dimension and batch limits,
Hermiticity, positive-root availability, independent normalized eigenpair
residuals, and the absence of an automatic fallback. They cannot prove
scientific gradient correctness.

A compute-node qualification should record the package/source/reference hashes,
runtime versions, settings, candidate roots and all selection diagnostics, then:

1. Verify total excited energy against official RHF plus TDA excitation energy.
2. Check every QM Cartesian force by central differences of the same selected
   total-energy surface, with at least two displacement sizes.
3. For embedded calculations, check every external-charge Cartesian force as
   well as every QM force; retain separate maximum/RMS errors and state labels.
4. Check zero-charge reduction, translation-force balance, and point/Gaussian
   charge cases separately.
5. Repeat displaced evaluations in reverse order and after loading the pinned
   reference record. No result should depend on evaluation history.
6. Exercise deliberately ambiguous roots, small candidate energy gaps, failed
   convergence, reference mismatch, and source-pin failure.
7. Only after those gates, consider a bounded excited-state trajectory and
   assess state identity and energy drift separately.

A small water fixture can qualify the interface derivatives for that fixture.
It does not validate photochemistry, proton transfer, an oxide interface,
adaptive partitioning or a pretrained ground-state DPA on excited surfaces.


## Qualified finite native combination

The isolated and point-charge embedded providers passed all-coordinate
finite differences at two displacement sizes with their immutable references.
The corresponding native mechanical and additive water-trimer routes then
passed independent static assembly, six serial/two-rank/restart cases, 86 dynamic
frames, 80 checkpoint/binary pairs and five comparisons per route. Four types
of corrupted electronic-state checkpoint were rejected before accepted output.
Every accepted dynamic frame retained the same-call reference, geometry and
selected-state diagnostics. The shared frozen source passed 620 package tests
and 328 subtests, with 11 explicitly documented skips.

These trajectories cover 20 steps at 0.1 fs, starting from zero velocities.
Maximum total-energy excursions were 3.347e-4 eV (mechanical) and 3.300e-4 eV
(additive), below the unchanged 1e-3 eV screen. This qualifies this short
fixed-region CPU RHF/TDA/dense/def2-SVP integration. It does not establish
long-time stability, timestep convergence, photochemistry or the untested
feature combinations above. Runtime records retain exact source/model hashes,
original failures, independent audits and full commands.
