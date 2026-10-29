# Finite ground/TDA gap restraint

The optional `electronic_gap` extension of the standard `qmdpa run` dispatcher
adds a static external restraint to an explicitly chosen complete ground or
selected surface. It currently admits finite fixed-QM CPU RHF/dense-TDA ME and
fixed-charge additive EE, raw CPU DPA inference, NVE and `sample_interval=1`.
The selected reference and its original overlap/SCF/response guards are required
also when the sampled surface is ground. Legacy ground-only water sampling
admission is unchanged. This is an engineering interface, not cDFT or ET sampling.

The extension requires these fields:

- `form`: `ground-tda-gap-restraint-v1`.
- `name`: a valid scalar CV identifier.
- `order`: `["ground", "selected"]` or its reverse; `x=U_order[1]-U_order[0]`.
- `sampled_surface`: `ground` or `selected`, independent of the gap order.
- `full_region`: complete global atom IDs, symbols and molecule groups.
- `restraint`: `center`, positive `kappa` and `unit` (`eV`). Kappa uses 1/eV.
- `kernel` and `wrapper`: each an explicit PLUMED library `path` and `sha256`.

For each same-geometry pair, `J=F_first-F_second`. The official PLUMED C bridge
receives `x` through `setExtraCV` and returns `-dB/dx` through
`setExtraCVForce`. The complete force is `F_sampled - (dB/dx)*J`, including
every environment row. ME environment rows are zero in J; additive EE retains
the embedded ground and selected field-source responses. Existing single-state
calculators assemble the two surfaces. Every call checks their ground RHF energy
parity at 1e-7 eV. No electronic solver or mixer is duplicated.

Native transport uses pair zero and the complete biased total once. Force calls,
run-zero queries and finite differences do not commit history. Accepted native
steps validate state, geometry, gap and full native forces before one history
commit. A checkpoint is published only after the native restart is written.
Restart pins physics, source, model, state order, partition, units and PLUMED
library identity, then recomputes both state forces and the CV. The initial
restart query retains the original accepted history after numerical comparison;
it does not count as another accepted step. Failure invalidates both state and
gap records, clears native external force/energy, latches the callback failure,
and leaves earlier checkpoint files unchanged. Discard the failed in-memory run.

Core tests cover identity, signed derivatives, exact call-local sharing and
failure clearing. The companion `benchmarks/formaldehyde/electronic-gap` owns
portable real-backend all-coordinate finite differences, serial/MPI trajectories,
bidirectional restart and controlled real-state failure. Numerical evidence
applies only to the explicitly tested configuration; chemical accuracy,
converged sampling, production stability and performance require separate tests.
