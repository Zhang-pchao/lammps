# Ground-state CV sampling

This development interface is limited to finite fixed-region ground-state ME
with CPU PySCF. Numerical admission belongs to an identified runtime and test
record. Existing NVE and static-distance inputs retain their meanings.

Use the unified entry point, `qmdpa validate` or `qmdpa run`. An optional
`thermostat` selects the official LAMMPS Nose-Hoover chain integrator:

```json
"thermostat": {
  "form": "nose-hoover-chain",
  "temperature_K": 300.0,
  "damping_fs": 50.0,
  "chain_length": 3
}
```

The initial-velocity `temperature_K` remains a separate existing setting.
Omitting `thermostat` retains NVE. The target temperature is constant; ramps,
NPT, stochastic thermostats and other Hamiltonian/device combinations are not
part of this profile. Native restart retains the NHC state under the same fix
identity. Each frame records its thermostat energy and extended-energy ledger.
A working thermostat does not prove equilibration or a converged distribution.

An optional `sampling` object configures official PLUMED forces:

```json
"sampling": {
  "engine": "plumed",
  "kernel": {
    "path": "runtime/lib/libplumedKernel.so",
    "sha256": "REPLACE_WITH_THE_INSTALLED_KERNEL_SHA256"
  },
  "cv": {
    "form": "distance-difference",
    "atom_ids": [1, 2, 4]
  },
  "bias": {
    "form": "harmonic",
    "center_angstrom": 0.0,
    "k_ev_per_angstrom2": 0.2
  }
}
```

For `distance-difference`, IDs are donor, transferring H and acceptor, and the
CV is r(donor,H)-r(acceptor,H). For `distance`, provide two IDs and a positive
restraint center. Every CV atom must remain in the fixed QM region. Chemical
bond changes inside QM are allowed; this interface does not define a moving
reaction core or a covalent boundary.

The driver generates a restricted PLUMED input using Angstrom/eV/ps units,
DISTANCE with NOPBC, COMBINE for the signed difference, and a declared bias. PLUMED
alone evaluates the production bias energy and force. The ME correction stays
unchanged; the bias is added after its native external fix. The legacy `bias`
and new `sampling` objects are mutually exclusive.

Each run writes `sampling/plumed.dat`, `plumed.log` and `COLVAR` below its output
directory. Accepted frames record the CV, native bias energy and potential after
subtracting that bias. Parameters and kernel identity enter restart identity;
changed settings and inconsistent frame records fail closed. The first profile
verifies the actual Linux native-library mapping and requires paths without
whitespace or command characters. Native PLUMED also requires consecutive
global atom IDs; absent or invalid CV atoms cannot produce an accepted state.

For umbrella windows, use separate output directories and declare each restraint
center explicitly. Validate window overlap, correlation, equilibration and
statistical uncertainty separately before combining/reweighting samples.
COLVAR and unbiased energies alone are not a free-energy estimate.

## Umbrella windows

Start from a validated finite ground-ME configuration with the sampling and
thermostat objects above. Place window configuration files next to the template,
so relative model, data and kernel paths retain their original meaning. For
example, the following creates three separately identified inputs without
overwriting an existing file:

```python
import json
from copy import deepcopy
from pathlib import Path

template = Path("sampling-template.json")
base = json.loads(template.read_text())
for index, center in enumerate((-0.3, 0.0, 0.3)):
    config = deepcopy(base)
    config["sampling"]["bias"] = {
        "form": "harmonic",
        "center_angstrom": center,
        "k_ev_per_angstrom2": 0.2,
    }
    path = template.parent / f"umbrella-{index}.json"
    with path.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
```

Validate each input with `qmdpa validate umbrella-0.json`, then run, for example,
`qmdpa run umbrella-0.json --output runs/umbrella-0`. The centers and spring
constant are examples, not a converged sampling prescription. Select an
equilibrated production interval separately for each window, examine overlap
of its signed CV histogram, estimate correlation times, and retain independent
blocks. Supply the exact harmonic bias and temperature to an established WHAM
or MBAR analysis; report its uncertainty and overlap diagnostics. Do not fit a
barrier from three short software-test windows or treat biased times as rates.

## One well-tempered metadynamics route

Replace only the `sampling.bias` object:

```json
"bias": {
  "form": "well-tempered-metadynamics",
  "sigma_angstrom": 0.1,
  "height_ev": 0.001,
  "pace_steps": 100,
  "bias_factor": 5.0
}
```

The temperature comes from the required constant-temperature `thermostat`.
This is a single CV, fixed Gaussian width, single walker, direct-hill PLUMED
METAD route. It does not accept arbitrary PLUMED text, adaptive Gaussians,
grids, multiple walkers or changing temperatures. These values illustrate the
interface; application-specific hill width, height and pace need convergence.

Each accepted checkpoint includes an immutable `sampling-history-*.HILLS`
snapshot, its checksum, hill count and deposition schedule. Restart from the
checkpoint JSON in a fresh output directory; the driver copies only its
accepted snapshot. Keep that JSON, native restart binary and matching HILLS
snapshot together. The growing `sampling/HILLS` in a failed run is never a
restart authority. Modified, missing, duplicate or future hills are rejected.

The native fix must suppress history updates on repeated evaluation of the
same MD step. At checkpoint capture the driver recalculates the post-deposition
energy without another update. A fixed-width hill has zero derivative at its
deposition center; positions and velocities are unchanged by this refresh.
Both uninterrupted/segmented and restarted trajectories require verification
with the identified native build. Kernel availability alone is insufficient.

Frame records distinguish total potential, bias, unbiased potential and
cumulative `deposition_work_eV`. PLUMED WT HILLS heights describe the scaled
free-energy estimate; the deposition-work ledger converts them by
`(bias_factor - 1) / bias_factor`. It is an accounting quantity and is never
added again to the production potential or force. The NVT extended energy
includes the time-dependent bias, so it is not itself conserved. A work-corrected
diagnostic can subtract cumulative deposition work; it does not prove sampling
convergence.

Use official PLUMED `sum_hills` and documented time-dependent reweighting with
the same units and bias convention for later analysis. Convergence needs
sufficient exploration, independent evidence and uncertainty assessment.
This software contract does not qualify a free-energy surface or a physical
transition rate.
