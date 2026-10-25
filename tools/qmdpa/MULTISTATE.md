# Shared fixed-reference state evaluation

The CPU RHF/TDA and bounded PBE0/TDA interfaces can return several named selected
states at the same geometry. It shares one SCF and dense TDA solution and uses
the existing single-state response for every requested state's complete forces.

```python
from qmdpa.excited import TDASettings
from qmdpa.multistate import build_tda_references, PySCFTDAStateSet

settings = TDASettings(basis="sto-3g", tda_solver="dense", nstates=3)
references = build_tda_references(
    qm, settings, states={"state_a": 1, "state_b": 2},
    environment_geometry=environment, charges=charges,
)
provider = PySCFTDAStateSet(references)
surfaces = provider.evaluate_surfaces(qm, environment)
a, b = surfaces["state_a"], surfaces["state_b"]
gap_ev = b.energy_ev - a.energy_ev
gap_qm_force = b.qm_forces_ev_per_angstrom - a.qm_forces_ev_per_angstrom
```

Omit environment and charges for an isolated molecule. The finite-charge API
returns each state's complete environment forces as well. For a gap, subtract
these in the same state order; these force differences are the negative gradient
of the energy difference. For energy-only requests use forces=False; unavailable
forces are None, never zero. The small basis above is a development choice, not
a chemical-accuracy recommendation.

Names identify immutable AO transition-density references. Each surface retains
the method-specific total energy (E_RHF + omega for RHF), reference hash, selected
root, candidate overlaps, residuals and response diagnostics. References can be
saved and loaded with the existing TDAReference JSON format. Preparation returns
an immutable mapping of those references; it does not qualify their forces.

References must share method/settings, atom IDs and ordering, runtime pins and
fixed field. Selection checks every requested state before evaluating gradients.
If two references select the same candidate root, a state is ambiguous, or any
quantum/response calculation fails, no partial state set is returned. The next
call starts a new solve; results and wavefunctions from a preceding call are not
reused. Input mappings and force arrays cannot mutate the provider's references.

Both routes require explicit CPU singlet dense TDA. Existing bounded matrix
size, residual, response, overlap and gap guards remain unchanged. Iterative solvers and open-shell state sets are not admitted by these CPU APIs.
The separate bounded GPU RHF API is described below.
Existing single-state native ME/additive/gap-anchor configuration is unchanged.
A quantum state set contains surfaces, not assembled hybrid total energies.
For complete named totals use the assembly API described below.
NAC/SOC, electronic populations, hopping and native simultaneous-state dynamics
are outside this data interface. Shared quantum work is not a demonstrated
complete-MD performance advantage.


For file-based preparation, the unified command qmdpa prepare-states accepts
named initial roots and writes portable per-state references. See
[STATE_INPUTS.md](STATE_INPUTS.md). The preparation command uses this same builder
once; it does not evaluate gradients or start a trajectory.


## Bounded CPU PBE0 state sets

Use build_pbe0_tda_references and PySCFPBE0TDAStateSet from qmdpa.multistate
with explicit PBE0TDASettings from qmdpa.pbe0_tda. The call shapes match the RHF
example. Each state's total energy is E_PBE0 + omega; its original PBE0 orbital,
quadrature and environment responses are evaluated separately on the shared
SCF/TDA solution. This is a different method and runtime identity from RHF.
Mixed RHF/PBE0 references, inconsistent grids/settings and foreign runtimes are
rejected before a state-set calculation. The RHF entry points still reject PBE0.

The existing allowed basis/grid pairs and three candidate roots are unchanged.
For a small development example use PBE0TDASettings(basis="sto-3g", grid_level=1,
grid_prune="nwchem_prune", grid_response=True, tda_solver="dense", threads=2).
The state-set route uses the existing operator matrix assembly; it does not
introduce another eigensolver, density fitting or an implicit fallback. Broader
chemical accuracy and native simultaneous-state trajectories remain separate.
The same prepare-states command accepts method pbe0-tda with explicit solver
and grid controls and writes the existing PBE0 per-state file format.


## Bounded GPU RHF state sets

Use build_gpu_tda_references and GPU4PySCFTDAStateSet from qmdpa.multistate
with explicit GPUTDASettings from qmdpa.gpu_excited. Their call shapes match
the CPU example. References carry the separate GPU source/runtime identity;
CPU references cannot be relabeled or mixed into a GPU state set.

The existing conventional GPU RHF SCF and dense TDA solve is shared. Each
requested state's complete existing gradient and finite point-charge reaction
forces remain separate, with the same orbital-response and state-tracking
checks. No gradient equation is copied, no CPU fallback is allowed, and a
collision or failed state cannot return a partial collection. Energy-only
calls omit every force and start a new common solve, as on CPU.

This Python API does not enable selected GPU DFT/density fitting, electronic
propagation or simultaneous-state native dynamics. The prepare-states command
also accepts the explicit GPU RHF request and writes the existing portable
per-state files; see [STATE_INPUTS.md](STATE_INPUTS.md). Numerical qualification
and cost claims require a matched runtime record.

## Complete hybrid totals for named states

Use qmdpa.multistate_coupling to assemble a complete hybrid energy and all-atom
forces for every name. These evaluators reuse the existing single-state
Hamiltonians; a quantum surface alone is not a hybrid total.

```python
from qmdpa.multistate_coupling import (
    MechanicalStateSet, AdditiveStateSet, GapAnchorStateSet,
)

me = MechanicalStateSet(region_low, isolated_states, full_low_backend=full_low)
ee = AdditiveStateSet(environment_low, embedded_states, cross_interaction)
anchor = GapAnchorStateSet(full_low, region_low, embedded_states)
results = ee.evaluate(full_geometry)
total_energy_ev = results["state_a"].total.energy
all_atom_forces = results["state_a"].total.forces
quantum = results["state_a"].quantum
```

All three routes accept the bounded CPU RHF, CPU PBE0 and GPU RHF state-set
types described above. ME requires isolated references; additive and plain
gap-anchor require embedded references. Fixed finite regions, fixed point
charges and each route's existing method limits apply. Complete forces are
required. The result mapping, quantum data and total arrays are immutable.
Total forces follow the full input atom order; each quantum surface retains
its own QM/environment ID order, reference hash and state diagnostics.

Each evaluate call solves the quantum state set once. State-independent low,
cross and anchor-ground evaluations are shared only when the backend object
and its exact ordered geometry inputs are identical within that call. Distinct
backend instances are never merged. The next call starts fresh, even after a
failure. Backend-internal caching remains the caller's separate responsibility.
Any missing state, mismatched identity, absent force or backend failure raises;
no partial collection is returned.

These APIs return complete totals. They must not be added to a native full-DPA
pair result as a correction. Existing native single-state configuration and
restart behavior are unchanged. There is no simultaneous-state native driver,
electronic propagation or cross-partition state correspondence in this module.
