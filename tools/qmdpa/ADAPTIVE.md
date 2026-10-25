# Experimental two-partition mechanical dynamics

The independent kernel defines one continuous ground-state mechanical energy
for two fixed complete-group partitions. The experimental native adapter now
uses that total energy and its all-atom force in LAMMPS, with the existing
accepted-frame and restart lifecycle. It does not turn the discrete offline
selector into a general adaptive dynamics algorithm. This route has no live
chemical regrouping, proton reassignment, excited-state mixing, or virial.
Native dynamics qualification is separate from the finite-geometry controls.

## Energy and force contract

The same full low-level baseline is used for both partitions:

```text
Delta_p(R) = E_QM(Q_p) - E_DPA(Q_p) + C_p
E_p(R) = E_DPA(full) + Delta_p(R)
E(R) = sum_p w_p(R) E_p(R)
     = E_DPA(full) + sum_p w_p(R) Delta_p(R)
F(R) = F_DPA(full) + sum_p w_p(R) F_Delta_p(R)
       - sum_p Delta_p(R) grad(w_p(R))
```

Here `F_Delta_p` is the isolated correction force scattered to the full atom
array, with zeros on that partition's environment. `C_p` is a declared constant
energy offset. Both weights sum to one and their gradients sum to zero. The
implementation evaluates the full baseline once and computes the switching
force as `-(Delta_B - Delta_A) * grad(w_B)`. This is algebraically identical to
the complete energy derivative and avoids subtracting the common absolute
baseline through two large weighted energies. The correction energy similarly
uses `Delta_A + w_B * (Delta_B - Delta_A)`.

Dropping the switching-force term changes the force law and is unsupported.
It can be substantial even with smooth weights. Reporting its magnitude is a
diagnostic, not a reason to omit it. A conservative mixed potential does not
prove that its switching forces, solvation structure, or reaction energetics
are physically accurate.

## Explicit groups, states, and energy references

Reuse `FixedGroup` from `selection_config`. Groups must be disjoint and cover
all input atom IDs and element identities exactly. No chemical grouping is
inferred from distances. The center and two candidate anchors must lie in three
distinct groups. Both QM partitions retain the center group and may differ only
by exchanging their respective candidate groups. Additional common groups are
allowed. Candidate and center anchors must be distinct in ID and position.

The two QM regions must have the same elemental composition, charge,
multiplicity, declared method identity, and energy-reference identity. They
therefore have the same declared electron count. The high providers must expose
`charge` and `multiplicity`; these are checked when constructing a partition and
again on every evaluation. A non-None low-region `charge_spin` is checked against
the same state. High-provider `xc` and `basis`, when present, must match between
partitions. No state is inferred from a water geometry label. The common full
baseline keeps the caller's explicit full-system conditioning; no environment
charge or spin is derived by the mixer.

The caller must bind `method_identity` and `energy_reference` to qualified
provider settings, model hashes, numerical modes and energy conventions.
Those strings are declarations, not substitutes for runtime identity checks.
Low providers without exposed charge conditioning remain the caller's explicitly
chosen unconditioned model; the mixer does not manufacture a charge condition.
Only `electronic_state="ground"` is accepted. Matching charge/multiplicity does
not independently establish ground-state convergence or identify an excited
state across different QM spaces.

No energy offset is fitted, inferred, or adjusted automatically. The default is
`C_p=0`. A nonzero `energy_offset_ev` requires an explicit `offset_source`.
Offsets enter both the weighted energy and switching force. An arbitrary offset
on one partition is consequently a different potential, even though it would
not change that fixed partition's own force. A common additive energy shift
changes the mixed energy by that constant and leaves all forces unchanged.
Unequal QM compositions or charge states require a separately justified energy
and electron-reference construction and are rejected here.

## Distance-based quintic exchange

For explicit anchors C, A and B, define

```text
s = |R_A - R_C| - |R_B - R_C|
t = (s + h) / (2 h), h = half_width_angstrom > 0
S(t) = 6 t^5 - 15 t^4 + 10 t^3
w_A = 1 - S(t), w_B = S(t), for -h < s < h
```

Outside this band, the nearer candidate's partition has unit weight. The first
and second weight derivatives vanish at the band endpoints. `h` is an explicit
geometric width, not a universal chemical constant. All derivatives with
respect to C, A and B are included; other atoms have zero weight derivatives.
Thus translating or rotating the whole geometry preserves the distance policy.
Equal distances are a regular half-weight point, not a discontinuous tie break.
Coincident anchors are rejected. Periodicity and nearest-image reconstruction
are unsupported.

## API

```python
from qmdpa.adaptive import (
    ConservativeMechanicalMixer,
    MechanicalPartition,
    QuinticDistanceExchange,
)
from qmdpa.core import SubtractiveCalculator
from qmdpa.selection_config import FixedGroup

water_groups = (
    FixedGroup("core", (1, 2, 3), ("O", "H", "H")),
    FixedGroup("a", (4, 5, 6), ("O", "H", "H")),
    FixedGroup("b", (7, 8, 9), ("O", "H", "H")),
)
partitions = (
    MechanicalPartition(
        "A", ("core", "a"), SubtractiveCalculator(region_low_a, qm_a),
        qm_charge=0, qm_multiplicity=1,
        method_identity="PBE/def2-SVP; declared ground-state settings",
        energy_reference="same pinned DPA model and QM reference for both regions",
    ),
    MechanicalPartition(
        "B", ("core", "b"), SubtractiveCalculator(region_low_b, qm_b),
        qm_charge=0, qm_multiplicity=1,
        method_identity="PBE/def2-SVP; declared ground-state settings",
        energy_reference="same pinned DPA model and QM reference for both regions",
    ),
)
calculator = ConservativeMechanicalMixer(
    full_low,
    groups=water_groups,
    partitions=partitions,
    switch=QuinticDistanceExchange(1, 4, 7, half_width_angstrom=0.4),
)
terms = calculator.evaluate_terms(full_geometry)
energy_ev = terms.total.energy
forces_ev_per_angstrom = terms.total.forces
```

The caller constructs and owns all scientific providers. The module imports no
PySCF, DeePMD, LAMMPS, or optional electronic-structure engine. Each isolated
correction reuses `SubtractiveCalculator.correction`; its `full_low` member is
not called. The common `full_low` receives sorted global atom IDs. Each isolated
provider receives sorted QM IDs. Per-atom provider parameters must be bound to
these documented orders. Output forces and weight gradients follow the original
input atom order. Reordering the input does not change atom identity.

Both partitions are evaluated on every call, including zero-weight endpoints.
There is no cutoff pruning, previous-result fallback, SCF history, or cached
successful force. Any invalid identity, state mismatch, provider failure, or
nonfinite result stops the calculation. Both provider states should be qualified
before treating even a pure endpoint as accepted.

`evaluate()` returns the existing `Evaluation` total. `evaluate_terms()` returns
an immutable `AdaptiveTerms` ledger with partition IDs/QM IDs, coordinate,
weights and gradients, both partition energies and offsets, the shared baseline,
both globally scattered corrections, weighted correction forces, switching
forces and their largest per-atom norm. `correction` includes the switching
force; `total` adds the baseline exactly once. The experimental driver below
publishes this total through the existing all-atom callback; it never adds the
full baseline a second time as a native DPA pair interaction.

## Experimental native driver

Run the adapter as a module; it has no run subcommand:

```bash
python -m qmdpa.adaptive_cli adaptive.json --output fresh-run
python -m qmdpa.adaptive_cli continuation.json --output fresh-continuation \
  --restart fresh-run/checkpoint-000000000020.json
```

The adapter installs only native `pair_style zero` and supplies the complete
potential through the existing `AdditiveExternalCoupling`. The native system
must contain exactly the declared full-region atoms and types, be nonperiodic,
and have no other native pair, bonded, or k-space energy. The existing runtime
adds NVE integration to all atoms; no water oxygen is fixed. It gathers atom
coordinates across MPI ranks, constructs providers on rank zero, and scatters
the total forces to all owned atoms. Collective failures clear the published
energy/forces and prevent accepting a new checkpoint.

One DPA provider is shared by the full baseline and both isolated corrections.
Each force call evaluates three DPA geometries and two ground-state QM
geometries; the full baseline is evaluated once. Both QM providers use the same
explicit settings. This first adapter requires PySCF on CPU, neutral singlets,
full integration-grid response, the original `.pt` model, and the strict raw
deterministic inference profile. GPU4PySCF, excited states, compressed inference,
periodicity, NPT and stress/virial are rejected or unsupported.

An explicit schema-1 input is:

```json
{
  "schema_version": 1,
  "hamiltonian": "finite-two-partition-mechanical",
  "data_file": "water-trimer.data",
  "model": "model.pt",
  "model_sha256": "<replace with the actual lowercase 64-character SHA256>",
  "type_symbols": ["O", "H"],
  "periodic": false,
  "full_region": {
    "atom_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "symbols": ["O", "H", "H", "O", "H", "H", "O", "H", "H"],
    "molecules": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  },
  "groups": [
    {"group_id": "core", "atom_ids": [1, 2, 3], "symbols": ["O", "H", "H"]},
    {"group_id": "a", "atom_ids": [4, 5, 6], "symbols": ["O", "H", "H"]},
    {"group_id": "b", "atom_ids": [7, 8, 9], "symbols": ["O", "H", "H"]}
  ],
  "partitions": [
    {
      "partition_id": "A",
      "qm_group_ids": ["core", "a"],
      "qm_charge": 0,
      "qm_multiplicity": 1,
      "method_identity": "PBE/def2-SVP ground-state control",
      "energy_reference": "same pinned QM and DPA reference",
      "electronic_state": "ground"
    },
    {
      "partition_id": "B",
      "qm_group_ids": ["core", "b"],
      "qm_charge": 0,
      "qm_multiplicity": 1,
      "method_identity": "PBE/def2-SVP ground-state control",
      "energy_reference": "same pinned QM and DPA reference",
      "electronic_state": "ground"
    }
  ],
  "switch": {
    "method": "quintic-distance-exchange",
    "center_atom_id": 1,
    "first_atom_id": 4,
    "second_atom_id": 7,
    "half_width_angstrom": 0.4
  },
  "full_charge_spin": [0, 1],
  "region_charge_spin": [0, 1],
  "provider": {"engine": "pyscf", "device": "cpu", "state": {"method": "ground"}},
  "qm": {
    "xc": "pbe",
    "basis": "def2-svp",
    "charge": 0,
    "multiplicity": 1,
    "grid_response": true,
    "threads": 1
  },
  "inference_profile": "torch-raw-deterministic-v1",
  "timestep_fs": 0.25,
  "steps": 20,
  "sample_interval": 1
}
```

The example is an engineering control, not a qualified water potential.
`temperature_K` and `velocity_seed` are optional and must appear together;
otherwise initial velocities come from the native data/restart.
Groups must exactly match `full_region.molecules`. Relative file paths resolve
against the input file. The loader rejects unknown fields, duplicate JSON
keys, nonfinite numbers, model checksum mismatch and unsupported states before
constructing scientific providers. A nonzero partition `energy_offset_ev`
requires `offset_source`; it changes both the potential and its identity.

The existing checkpoint records accepted coordinates, velocities, energy,
forces, the native binary checksum, and `physics_sha256`. The physics hash
binds source/dependency identity, model/data bytes, the full groups and
partitions, switching policy, offsets/references, QM settings/state and timestep.
Only file locations, requested step count, sampling interval and initial
velocity-generation metadata are excluded. Continuation checks the saved hash
and binary before re-evaluating the restart energy/forces. Changing a switch,
group, offset, method or source therefore requires a new run. Stateless weights
are re-evaluated from accepted coordinates; no hidden membership history is
stored or restored.

The driver currently publishes only the total evaluation to accepted frames.
The kernel's `evaluate_terms()` ledger remains available for independent
switching-force and partition diagnostics. The frozen `adaptive-native-v1`
revision passed the short native, MPI and restart gates described below. The bounded moving-water crossing described below has since completed.

## Qualification and later extensions

Analytic controls test all Cartesian derivatives, the center response, pure
endpoints and smooth boundaries, whole-group/state rejection, stable ordering,
translation/rotation, common and partition-specific offsets, immutable results,
provider rejection, one shared baseline call and reversible stateless scans.
These controls validate algebra. A frozen real small-cluster DPA/PySCF screen
also checks finite differences and reversible geometry scans. Its ledger and
switching-force controls passed, but reducing the displacement did not improve
the total DPA-containing force error monotonically. A broad model-noise allowance
therefore does not establish fine force convergence.

An independent audit of `adaptive-native-v1` passed native static assembly,
six serial/two-rank cases, 20 versus 10+10 steps at 0.25 fs, and an intentional
MPI SCF failure before any accepted frame or checkpoint. All 86 dynamic frames
and 80 checkpoint/binary pairs were verified. The five trajectory comparisons
passed; the largest force difference was 4.013e-11 eV/Angstrom. Maximum energy
excursion was 2.420e-4 eV, below the predeclared 1e-3 eV limit. Static parity
against independently assembled components was 7.276e-12 eV and
1.898e-11 eV/Angstrom. These results qualify that frozen 5 fs engineering case.

Two separate 50 fs moving-water runs at 0.25 and 0.125 fs completed the
frozen adaptive-crossing-v1 engineering checks. The terminal audit verified
602 frames and 600 checkpoint/binary pairs, actual coordinate crossing and
weight reversal, and the original energy screen. These are configuration-bound
engineering results; timestep convergence and physical full-QM agreement
remain unqualified.

The three-water example exchanges two neutral whole-water candidates while
retaining a central water. It does not reassign hydrogens. A dissociated or
shared proton cannot be promoted using these immutable molecular groups; a
future charge/electron mapping and complete smooth partition family must first
be defined and qualified. No claim about proton-transfer trajectories follows.

Reference-subtractive EE can later provide a correction with both QM and
environment response forces, using the same energy-weighting algebra. That
adapter and its reference-charge/double-counting qualification are not supplied
here. Excited-state extensions additionally require the same physical state to
be identified across partition orbital spaces; matching root numbers is not a
state map. Polarization, responsive charges and nonadiabatic dynamics require
further derivatives and state/history contracts.

## Complete candidate families

MOLECULAR_ADAPTIVE.md extends the ground mechanical family to a fixed core
plus exactly k candidates from an explicit complete molecular pool.
MOLECULAR_ADDITIVE.md supplies the separate additive EE family with immutable
atom-ID charges and complete environment forces. These reuse conservative
weighting; they do not change this historical two-partition Hamiltonian.
