# Named scalar CV interface (version 1)

`named_cv.NamedCVSchema` declares an ordered, unique set of named scalar CVs,
explicit scalar shape `[]`, canonical unit (`eV`, `angstrom`, `dimensionless`)
and checksummed provider/state/partition dependencies. Tensors and implicit
units are rejected. The result for each name includes a same-geometry digest,
the same dependencies, its value and a complete Jacobian by stable global atom
ID. Zero rows must be explicit, including environment atoms. Input Hartree and
Bohr conversion scales value and Jacobian together; coordinate derivatives are
always with respect to Angstrom.

`CVFrame.pullback` implements `F_bias = -J.T @ dB/dx`. The bias is added to the
already complete sampled potential exactly once. No new mixer or electronic
solver is implemented. A state-energy-difference producer is a later admitted
combination; the finite CPU ground-state sampling guards remain in force.

`external_cv.ExternalCVBias` sends actual scalar buffers to the official
PLUMED `setExtraCV` interface and receives `-dB/dx` via `setExtraCVForce`.
It uses the documented nonthrowing C wrapper, checksum-pinned wrapper/kernel
libraries, and matching explicit numerical units on the input and MD sides.
Historical PLUMED versions use a slightly different constant for the named eV
unit; the bridge explicitly avoids mixing that constant with the MD conversion.

The first bridge admits static harmonic restraints. Each parameter map binds
`center`, positive `kappa` (eV per squared CV unit) and the declared `unit` by
CV name. It does not install a native `fix plumed`; callers must not also apply
the same bias through another owner. The returned record contains bias energy,
named derivatives, full Cartesian forces and the exact CV frame identity.

Pure `evaluate` calls use `performCalcNoUpdate`; finite differences and repeated
queries do not accept a step. `accept` advances consecutive steps once;
identical duplicate acceptance is idempotent and changed same-step data fails.
Checkpoints bind schema/order/units/dependencies, bias parameters, kernel and
wrapper hashes, timestep, accepted frame and count. Static bias has no evolving
OPES state. Electronic OPES, selected-state CV production, cDFT and adaptive
state tracking are not admitted by this module.

`collective_evaluate` gives the root sole ownership of PLUMED, checks identical
full frames and disjoint complete global-ID ownership, propagates failures to
all ranks, and returns the local owned Cartesian forces. Empty ranks are legal;
duplicate owners, ghosts as owners, missing atoms and differing steps fail.

Existing water OPES accepts either positional widths in the historical
log-distance/ionization order or maps named `logdistance` and `ionization`.
Both normalize to the identical old schema and PLUMED text. Its native exact
STATE lifecycle and ground-state admission are unchanged. Legacy source or
checkpoint hashes are never edited to claim compatibility with changed code.

Validation is split between `tests/test_named_cv.py` and the companion real
PLUMED analytic probe plus the portable water native/restart regression.
Analytic potentials establish interface derivatives, not electronic chemistry.
