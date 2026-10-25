# Experimental GPU4PySCF provider APIs

`qmdpa.gpu_backends.GPU4PySCFBackend` adapts official PySCF `to_gpu()` for
isolated molecular, closed-shell conventional RHF or density-fitted PBE/PBE0.
It uses the existing ground-provider construction and full grid/auxiliary-basis
response settings. Inputs use `Geometry`; results use eV and eV/Angstrom.

```python
from qmdpa.gpu_backends import GPU4PySCFBackend

backend = GPU4PySCFBackend(
    xc="pbe", basis="sto-3g", threads=2,
    density_fit={"auxbasis": "def2-universal-jkfit"},
    grid_level=1, grid_response=True,
)
result = backend.evaluate(geometry)
```

Install a mutually compatible GPU4PySCF, CuPy and CUDA stack separately. Imports
are lazy until construction, and exactly one visible GPU is required. CPU
conversion, host orbital coefficients, unsupported gradient classes and failed
SCF/device operations are errors; no CPU fallback or previous force reuse occurs.
Official gradients can return host arrays without changing their GPU execution.
Named density fitting is an explicit integral approximation, not an exact
optimization of conventional integrals. Unsupported functionals, open shells,
NLC and periodic QM are rejected. Embedding and selected states have their own
providers below.

The fixed finite-molecular GS-ME route accepts this explicit provider in the unified
configuration (all existing model, region and molecular settings remain required):

```json
"provider": {
  "engine": "gpu4pyscf", "device": "cuda", "state": {"method": "ground"}
},
"qm": {
  "xc": "pbe", "basis": "sto-3g",
  "density_fit": {"auxbasis": "def2-universal-jkfit"},
  "grid_response": true
}
```

Native assembly remains full-system DPA plus the QM-minus-DPA region correction.
MPI evaluates the QM provider on the callback root and broadcasts the complete
correction; it does not distribute a single SCF over GPUs. The same one visible
GPU can serve DPA and QM. GPU dependency versions join the restart identity.
Ground ME may use a fixed three-dimensional periodic full DPA system while
keeping both the QM calculation and the subtracted DPA region finite. The
existing native image flags define the unwrapped molecular geometry. Input
images must describe the intended complete molecules; the driver neither
infers bonds nor independently reimages fragments. The correction has the
identity coordinate Jacobian at fixed cell. Equivalent wrapped x/image pairs
must preserve the same unwrapped geometry across migration and restart.

For this route, use periodic=true with the ground mechanical selector and the
same finite QM settings. The configuration description records the three
boundaries and image mapping. Its energy is periodic full DPA minus finite
region DPA plus finite molecular QM. Even when every atom is in the QM region,
the periodic-minus-finite low-level residual generally remains; full-QM
cancellation is not guaranteed. This is not periodic electronic structure.
Periodic EE, selected-state PBC, adaptive and matched-fragment routes remain
unsupported. Numerical qualification belongs to the frozen method and fixture.
The plain RHF gap-anchor has the separate matched route described below. Selected ME uses the separate RHF/TDA provider described below. Finite additive EE
uses the separate complete point-charge provider described below.
Runtime libraries and numerical/native qualification still require an exact
source-matched record; registration alone does not qualify a machine or method.
Small-system startup overhead can outweigh GPU work; no speed advantage follows
from successful conversion or an isolated electronic-structure comparison.

## Conventional RHF ground surface

Use the same ground provider with `qm.xc="hf"` and no `density_fit` (or null).
The adapter constructs official `scf.RHF(...).to_gpu()` and its total analytic
gradient. This matches the integral method used by the selected RHF/TDA provider;
matching basis, charge, spin, geometry and embedding field are still required.
Ground energy alone is not a selected-state energy or gradient.

RHF uses no numerical DFT grid or auxiliary basis. `grid_level` and
`grid_response` retain their schema defaults but have no RHF quadrature to
control; embedding diagnostics mark grid quantities as null. Named fitting,
open shells and NLC remain unsupported for this RHF route. The existing
PBE/PBE0 defaults, fitting requirements and response controls are unchanged.
Both isolated ME and finite point-charge additive EE use the existing unified
configuration and complete-force assembly. Numerical and native evidence must
still identify the tested method and runtime; this ground API alone does not
qualify a composed gap-anchor or fragment model.

## Experimental finite point-charge API

`GPU4PySCFElectrostaticBackend(charges, **qm_settings)` accepts explicit
`FixedEmbeddingCharges` and evaluates `(qm_geometry, environment_geometry)`.
It shares the CPU `EmbeddedEvaluation` contract and returns QM internal plus
QM/charge interaction energy, complete QM forces and charge-site reaction forces.
Environment internal energy and cross-region short-range terms are absent;
this provider is not a full additive potential on its own.

The GPU embedding API accepts point charges and either conventional singlet RHF
or PBE/PBE0 with named fitting and complete grid/auxiliary-basis response. Gaussian radii,
QM dipole diagnostics, selected states and periodic boundaries are rejected.
The density stays on GPU for electronic charge-site derivatives. Only complete
energy/force outputs are transferred to host. To select finite additive EE,
use the same gpu4pyscf/cuda/ground request with an explicit ground/electrostatic
coupling and ee_model="additive". Declare all fixed point charges and cross
parameters as before. The native pair is zero; the external callback returns
complete total energy and forces, including environment DPA and cross terms.
Registration does not transfer numerical qualification from another source or
runtime. Configuration rejects Gaussian radii and dipole requests before any
quantum engine is constructed.

## Isolated selected-state RHF/TDA

GPUTDASettings, GPUTDAReference, build_gpu_tda_reference and
GPU4PySCFTDABackend in qmdpa.gpu_excited provide a separate experimental
singlet RHF/TDA surface. Conventional integrals are explicit; density fitting,
DFT selected gradients and periodic cells are not supported. The separate
finite-field Python API below handles point charges.
Total energy is E_RHF + omega; all QM forces use the official total gradient.
The fixed AO transition-density selector, candidate-root residuals and dense
operator assembly are shared with the existing CPU implementation. Dense
diagonalization and the operator execute on GPU; host transfers serve immutable
state tracking and independent residual validation. Iterative failure is not
retried with a different solver.

The GPU reference carries its own method ID, GPU4PySCF/CuPy versions and source
hashes as well as PySCF identity. CPU state files cannot be relabeled as GPU.
The inspected official gradient directly calls its CPHF solver. An instance-local,
source-checked function binding checks that response before the gradient consumes
it. If needed, the existing bounded normalized defect correction refines the
response using the same official GPU solver and unchanged residual tolerance.
It does not modify package globals or copy gradient equations. It removes the temporary method binding after each call.
The unified provider now accepts gpu4pyscf/cuda with state.method=rhf-tda for
fixed finite ME or additive point-charge EE with an explicitly dense GPU reference. prepare-state accepts
explicit engine=gpu4pyscf and device=cuda; CPU defaults and older state records
are preserved. Accepted records include GPU response identity and residual history.
Native numerical qualification is separate from this software registration.

## Selected finite point-charge Python API

GPU4PySCFTDAElectrostaticBackend and build_gpu_tda_reference accept explicit
environment_geometry and FixedEmbeddingCharges. This API retains the existing
selected E_RHF+omega contract and returns all QM and environment derivatives.
Reference records add the inspected field/density implementation hashes; isolated
reference records keep their original identity. Gaussian radii are rejected.

The source-pinned official base gradient exposes its response-density intermediate
through the with_solvent bookkeeping branch. The adapter verifies that the base
solvent_response function is identically zero and no solvent object is present.
This exports the relaxed density; it introduces no continuum or polarization
model. The density stays on GPU for official charge-site derivative contraction,
with electron-count and finite-value checks. Unified selected additive EE now
uses this provider with pair zero and complete total energy/forces. Accepted
state records check the relaxed-density identity and electron-count residual.
Fragment corrections still reject GPU selected providers. Registration alone
does not establish native numerical or scientific qualification.

## Plain RHF gap-anchor

`make_gpu_rhf_gap_anchor` in `qmdpa.gap_anchor` composes the existing isolated
RHF ground, embedded RHF ground and embedded RHF/TDA selected providers from
one GPU reference. The shared GapAnchorCalculator supplies all terms and forces;
no coupling formula or gradient is duplicated. Ground and selected terms use
the same basis, charge, spin, convergence and finite point-charge field.

The unified selected/electrostatic configuration accepts ee_model="gap-anchor"
with gpu4pyscf/cuda/rhf-tda. Assembly is pair zero plus the complete total energy
and all forces, including the full-system DPA already inside the total.
Accepted selected-state records retain the GPU response/density identity.
GPU dependency versions enter the restart identity. CPU references cannot be
relabeled as GPU and there is no CPU fallback. Only selected exact-geometry
reuse is available here; reuse_ground_terms remains a CPU PBE0 option.

Corrected/fragment gap-anchor, selected GPU DFT/DF, periodic and adaptive GPU
routes remain unsupported. Qualification must cover the composed total energy
and complete QM/environment derivatives in its own frozen runtime record.
