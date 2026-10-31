# cDFT gap umbrella (finite CPU constrained pair)

`cdft_umbrella` is the V2-05a sampling device: a static umbrella restraint over
the same-frame constrained-state gap that the already qualified `uks-hirshfeld-cdft`
pair produces. It owns no mixer, no solver and no electronic state. The pair
resolves both constrained surfaces at identical coordinates and exposes their
energy difference as the named scalar `cdft_gap` with a complete all-atom
Jacobian; the umbrella turns that observation into a bias through the existing
official PLUMED `EXTRACV` bridge.

One configuration declares exactly one umbrella window. A multi-window ensemble
is realized as several configurations that share one frozen reference and differ
only in the window center, so that each run's bias stays a single well-defined
function of the observed gap.

## Admission

`cdft_umbrella` is bound only when the configuration declares an explicit
constrained-state provider on the finite additive electrostatic Hamiltonian, a
nonperiodic system, actual CPU inference, a sample interval of one, and no
other bias owner. The window requires a finite center in eV and a positive
kappa in 1/eV. The PLUMED kernel and wrapper are checksum-pinned at bind time
and their resolved paths are kept out of the hashed identity.

The umbrella reuses the cDFT pair schema rather than declaring a second one.
The umbrella value and the accepted pair record are therefore provably the same
quantity, not two independently computed gaps.

## Force and energy assembly

The restraint is one half of `kappa` times the squared displacement of the gap
from the window center. Its derivative is with respect to the gap; the
Cartesian force is the pullback of that derivative through the recorded gap
Jacobian, `-dB/dx = -dB/dxi * dxi/dx`. The bias energy and force are added to
the sampled surface total exactly once. The umbrella force is a function of the
gap only, so it does not require the two surfaces to be differentiated against
each other beyond what the pair already provides.

## Accepted-frame record

Every accepted frame carries a `cdft_umbrella` record with the schema and CV
identities, the observed gap, the admitted window center and kappa, the bias
energy, the named derivative, the Cartesian pullback, the complete CV record
and the accepted bridge history. The record is rejected when the gap, the
window, the bias energy, the derivative or the pullback is altered, including
when the alteration is rehashed. The record is also bound to the accepted
history identity, so a changed kernel, wrapper, timestep or source identity is
rejected at restart.

A restart must reproduce the umbrella record of its parent step: the same gap,
the same bias energy and the same pullback. A pure restart query does not
advance the umbrella history.

## Scope

This module qualifies a sampling device on the already qualified finite
constrained pair. It does not qualify any chemistry, does not open electronic
OPES, does not admit dynamic electronic partition switching, and does not
establish long MD stability, free energies, rates or performance. The
configuration-probability decomposition of Eq. (V2-11) is a separate
statistical gate that reads the accepted frames of this device.

Validation is split between `tests/test_cdft_umbrella.py` in the coupling
source, the companion `tests/cdft_umbrella_native.py` native acceptance and the
companion `tests/cdft_umbrella_analysis.py` scientific recomputation. The
scientific analysis is recomputed from the raw accepted frames rather than
read from a stored summary.
