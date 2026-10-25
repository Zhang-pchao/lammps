# Implemented capabilities and explicit limits

The entries below describe implemented routes, not unrestricted accuracy or
performance. Scientific suitability of the DPA model, charges, cross parameters,
QM method and partition must be established for the intended application.

| Provider | Ground fixed ME / additive EE | Selected fixed ME / additive EE | Plain selected gap-anchor | Molecular dynamic family |
| --- | --- | --- | --- | --- |
| CPU PySCF | HF/DFT under declared settings; optional ground density fitting | Source-pinned RHF/TDA and explicit PBE0/TDA profiles | Matched RHF or PBE0 | Ground ME and additive EE, finite neutral singlet |
| GPU4PySCF | Conventional RHF or named-density-fitted PBE/PBE0; ground ME also admits a periodic full DPA system with a finite molecular region | Dense conventional RHF/TDA only | Matched RHF only | Not admitted |
| ABACUS CPU | Ground ME only: finite-supercell approximation or explicit same-cell orthorhombic periodic replacement; CPU PW/PBE/Gamma; qualification remains configuration-specific | Not admitted | Not admitted | Not admitted |

CPU RHF selected gradients require the inspected PySCF instance-level CPHF
hook described in EXCITED.md. Stock PySCF with the same version label is not
equivalent. PBE0 and GPU methods carry different source pins. Prepared state
files cannot be relabeled between methods or CPU/GPU providers.

The shared named-state API supplies complete per-state hybrid totals and
forces for CPU RHF/PBE0 and GPU RHF. It does not propagate simultaneous
electronic populations or provide nonadiabatic couplings. Native selected
dynamics follows one declared fixed-reference surface.

Dynamic complete-molecule ME and additive EE retain a fixed core, exactly k
same-composition candidates and every possible k-subset. They retain all weight
derivatives and reject missing coverage. They do not infer bonds, move protons
between groups, change occupancy, cut covalent bonds, map excited states across
orbital spaces, or implement periodic EE and mutual polarization.

The native integrator is existing fixed-cell LAMMPS NVE. MPI distributes atoms
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

CP2K and SHARC integration are deferred. ABACUS GPU/selected/EE, GPU selected
DFT, reactive partitioning, nonadiabatic dynamics and automatic D3/D4 additions
are not advertised as implemented. Future backends must provide the same
complete energy/force/state contracts through their official interfaces.
