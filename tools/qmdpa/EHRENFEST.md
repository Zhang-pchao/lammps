# Finite two-state Ehrenfest dynamics

Development 0.4.0.dev7 adds an explicit CPU, finite, fixed-QM two-state
mechanical Hamiltonian. It uses two equally averaged singlet SA-CASSCF(2,2)
states with real orbitals, a fixed all-electron basis and a prepared signed
wavefunction reference. This is a separate route from RHF/PBE0 TDA named-state
energies and forces. The latter do not supply these derivative couplings.

## Model and integrators

For each state, the complete potential is

    U_i = E_DPA(full) + E_i(QM) - E_DPA(QM).

The native pair is zero. The external callback supplies the complete expectation
potential and all forces; adding a native DPA potential again would double count.
With normalized complex amplitudes c and d01 = <0|grad 1>, the force is

    F = |c0|^2 F0 + |c1|^2 F1 - 2 Re(conj(c0)*c1) (U1-U0) d01.

Environment rows of d01 vanish for this model. Complete surface derivatives
and the coherent term have separate real-backend finite-difference tests.
The coherent-force witness projects displaced Hamiltonians using signed full
many-electron overlaps, including inactive core determinants and cross-AO
metrics. It does not differentiate energy at fixed adiabatic amplitudes.

LAMMPS velocity Verlet integrates nuclei. A unitary eigendecomposition of the
2 by 2 generator diag(U) - i*hbar*(v.d) propagates electrons, using averaged
endpoint values and nuclear secant velocities. Repeated callbacks at the same
step and coordinates reuse the same electronic history. No amplitude
renormalization or implicit root replacement repairs an invalid trajectory.

## Source and state identity

The provider requires the inspected PySCF 2.14.0 NAC response correction.
In nac/sacasscf.py, the bra CI response offset must be conditioned on the bra
index. The input source hash and corrected hash are recorded by the companion's
`environment/build_pyscf_nac_overlay.py`. The owning PySCF fork carries the
single-line correction and its reversed-state regression. A stock package with
the same version label is insufficient.

Independent signed-wavefunction derivatives establish that this pinned native
implementation's state=(0,1), use_etfs=False yields d01=<0|grad 1>, despite the
opposite tuple wording in its docstring. Both native orders are evaluated and
must be antisymmetric. No electron translation factors or force projections are
applied; nuclear momentum alone is not claimed to be conserved for a coherent
state under this convention.

Every geometry is mapped to the same prepared reference using signed overlaps.
Weak or ambiguous matches, a too-small gap, changed atom identities, failed SCF,
failed gradients or failed coupling response stop the run. Active-space changes,
conical intersections and unrestricted state tracking are outside this route.

## Prepare and configure

Prepare the reference through the lazy provider API in the same pinned runtime
that will execute the calculation:

```python
import json
from pathlib import Path
from qmdpa.core import Geometry
from qmdpa.casscf_pair import CASSCFPairSettings, build_casscf_pair_reference

qm = Geometry((1, 2), ("Li", "H"), [[0., 0., 0.], [1.5, 0., 0.]])
reference = build_casscf_pair_reference(qm, CASSCFPairSettings())
Path("pair.json").write_text(json.dumps(reference.record(), indent=2) + "\n")
```

The unified run JSON uses `hamiltonian: finite-two-state-ehrenfest-mechanical`
or the exact selector
`coupling: {surface: two-state, embedding: mechanical, me_model: ehrenfest}`.
Its required additions to ordinary native file/model controls are:

- `provider: {engine: pyscf, device: cpu, state: {method: sa-casscf22}}`;
- `periodic: false`, explicit `full_region` and `qm_region`, each with stable
  `atom_ids`, `symbols` and complete declared `molecules`;
- `pair_reference: {path: pair.json, sha256: <file SHA256>}`;
- `qm: {}` for default reference controls, or exactly matching explicit controls;
- `initial_amplitudes: [[real0, imag0], [real1, imag1]]` with unit norm;
- `full_charge_spin: [0,1]`, `region_charge_spin: [0,1]` and
  `inference_profile: torch-raw-deterministic-v1` for a compatible raw `.pt` model.

Native data provides positions and velocities unless both temperature and seed
are supplied. `timestep_fs`, steps and sample interval follow the shared CLI.
Use `python -m qmdpa.cli validate run.json`, then the ordinary `run` command.
Plain NVE is admitted; sampling, thermostat and practical execution extensions
are rejected for this coherent route.

## Restart and qualification

Every accepted frame includes complex amplitudes, both complete state energies
and forces, NAC, geometry, step and a checksum bound to the full physics identity.
A restart recomputes the provider at the accepted geometry, checks it against
stored history, then retains the verified endpoint to continue reproducibly.
Changed timestep, state preparation, model, source or amplitudes reject restart.
SCF, reference-match or ownership failure preserves earlier accepted files and
permanently invalidates that in-memory propagation instance.

The bounded real LiH plus H2 fixture checks two-state and coherent derivatives,
nonzero population change, energy/timestep refinement, serial/MPI equivalence,
cross-rank restart, duplicate callbacks and failure after acceptance. These are
software tests of a 0.2 fs fixture, not chemical accuracy, long-time stability,
decoherence, surface hopping, SOC, periodic NAD or adaptive-QM NAD qualification.
Selected-state adaptive partitioning is a separate Hamiltonian and milestone.
