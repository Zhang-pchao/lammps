# Implemented capabilities and explicit limits

The entries below describe implemented routes, not unrestricted accuracy or
performance. Scientific suitability of the DPA model, charges, cross parameters,
QM method and partition must be established for the intended application.

| Provider | Ground fixed ME / additive EE | Selected fixed ME / additive EE | Plain selected gap-anchor | Molecular dynamic family |
| --- | --- | --- | --- | --- |
| CPU PySCF | HF/DFT under declared settings; optional ground density fitting | Source-pinned RHF/TDA and explicit PBE0/TDA profiles | Matched RHF or PBE0 | Ground ME and additive EE; separate selected common-core RHF/TDA ME family |
| GPU4PySCF | Conventional RHF or named-density-fitted PBE/PBE0; ground ME also admits a periodic full DPA system with a finite molecular region | Dense conventional RHF/TDA only | Matched RHF only | Not admitted |
| ABACUS CPU | Ground ME: finite-supercell approximation or explicit same-cell orthorhombic periodic replacement; CPU PW/PBE/Gamma; qualification remains configuration-specific | Not admitted | Not admitted | Not admitted |

CPU RHF selected gradients require the inspected PySCF instance-level CPHF
hook described in EXCITED.md. Stock PySCF with the same version label is not
equivalent. PBE0 and GPU methods carry different source pins. Prepared state
files cannot be relabeled between methods or CPU/GPU providers.

The shared named-state API supplies complete per-state hybrid totals and
forces for CPU RHF/PBE0 and GPU RHF. It does not propagate simultaneous
electronic populations or provide nonadiabatic couplings. Native selected
dynamics follows one declared fixed-reference surface.

Development 0.4.0.dev7 separately admits finite fixed-QM CPU two-singlet
SA-CASSCF(2,2) Ehrenfest ME with strict raw DPA inference and plain native NVE.
Signed full-wavefunction correspondence, complete coherent forces and
immutable electronic history are defined in EHRENFEST.md. This is not a
TDA/NAD, hopping, periodic or adaptive-state capability.

Development 0.4.0.dev8 adds a separate finite selected-state molecular ME
family: CPU RHF/TDA, a common intact chromophore core and complete fixed-k
same-composition candidates. Own/core and bilateral cross-AO transition-density
correspondence guard every active state; all switching derivatives are retained.
See SELECTED_ADAPTIVE.md. This does not combine adaptive regions with Ehrenfest,
EE, reactive reassignment, periodic boundaries or enhanced sampling.

Ground dynamic complete-molecule ME and additive EE retain a fixed core, exactly k
same-composition candidates and every possible k-subset. They retain all weight
derivatives and reject missing coverage. They do not infer bonds, move protons
between groups, change occupancy, cut covalent bonds, map excited states across
orbital spaces, or implement periodic EE and mutual polarization.

Development 0.4.0.dev3 adds a distinct protected aqueous core with a variable
whole-water buffer, using every subset of one to six declared candidate waters.
It uses a bound isolated-water energy reference and all switching derivatives.
Core-internal proton movement is permitted; outer water identity loss, a proton
crossing the protected boundary and undeclared water coverage fail explicitly.
See WATER_BUFFER.md. It does not change the existing fixed-k ME/EE model or
qualify general reactive regrouping. CPU ground ME, finite boundaries, raw
deterministic inference and NVE define the initial validation scope.

The default native integrator remains fixed-cell LAMMPS NVE. Development
0.4.0.dev1 adds native NHC NVT and official PLUMED sampling for finite,
fixed-region CPU PySCF ground ME: distance or distance-difference harmonic
restraints, and a bounded well-tempered metadynamics route with immutable
accepted history. The low-level DPA device is separately identified. See
SAMPLING.md for admitted inputs, history and analysis contracts. This does
not change the frozen 0.3.0 release or qualify adaptive/selected-state sampling.

Development 0.4.0.dev2 adds an explicit practical float32 DPA4C execution
profile and the official CUDA deepmd/kk pair for finite fixed-region,
neutral-singlet CPU ground ME. It uses an independent precision budget and
retains strict geometry/identity checks. NVE, one GPU and ordinary or Kokkos
inference are the initial scope; see EXECUTION.md. Historical strict
float32 failures and other configurations retain their original status.

MPI distributes atoms
and communicates callback results; it does not parallelize a quantum solve
across ranks or GPUs. Source-bound restart and failure checks remain mandatory.
Pressure/virial and NPT are not qualified. Small-system serial/MPI/restart and
force tests are software evidence, not long-time dynamics or chemical accuracy.

Explicit cross-interaction/fragment differences are separate models documented
in INTERACTION_DELTA.md and CLUSTER_RESPONSE.md. Fixed-charge reference-
subtractive EE has a finite CPU ground-state native route with explicit point
charges and deterministic raw inference; see REFERENCE.md. Its numerical
qualification does not establish a physically unique DPA electrostatic split. None of these
decomposes a black-box DPA into electrostatic, exchange, induction or dispersion.

CP2K and SHARC integration are deferred. ABACUS GPU/selected, GPU selected
DFT, general reactive partitioning, nonadiabatic dynamics and automatic D3/D4 additions
are not advertised as implemented. Future backends must provide the same
complete energy/force/state contracts through their official interfaces.

Development 0.4.0.dev4 adds official OPES_METAD with classical reactive
soft-Voronoi water CVs and exact accepted history; see SAMPLING.md.
Development 0.4.0.dev5 adds a bounded finite ground reactive-water Hamiltonian
with explicit proton reassignment, integer electron allocation, optional
neutral-water subsets and every weight derivative. It also admits the bounded
OPES combination. See REACTIVE_WATER.md; this is separate from fixed-k mixing.

Development 0.4.0.dev6 adds fixed-region, neutral, same-cell periodic Gaussian
reference-subtractive EE with CPU ABACUS ground PW/PBE. The shared reference
assembler returns complete totals through zero-pair transport. The field,
reference and ownership conventions are defined in PERIODIC_EE.md; this is
neither additive EE nor a qualification of all ABACUS electrostatic combinations.
All development entries require source/runtime-specific evidence and retain
separate scientific, long-time and performance gates.
