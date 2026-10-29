# Provider combinations

The mechanical and additive dynamics entry points use the same lazy QM factory. Existing inputs
default to this explicit normalized request; callers may also declare it:

```json
"provider": {
  "engine": "pyscf",
  "device": "cpu",
  "state": {"method": "ground"}
}
```

The method/basis, electronic charge, multiplicity and convergence controls remain
under `qm`. The presence of the independently validated embedding charges
selects the fixed-charge provider. No device or excited-state request is ignored.
The factory rejects unsupported intersections before engine construction.

The resolved `provider` and `provider_contract` are retained in the manifest
and physics/restart identity. The contract declares total electronic+nuclear
energy, QM-only or QM+environment forces, the provider's declared QM boundary, no
provider history and unsupported stress. A periodic full mechanical environment
does not turn the isolated QM provider into a periodic DFT calculation.

This contract describes implementation, not numerical or scientific acceptance.
Qualification belongs to the exact source, native runtime, model, settings and
test records. New state/field/device combinations must pass their own derivative,
failure, native and restart gates. State and environment support cannot be
inferred by combining independent feature flags.

Direct Python API:

```python
from qmdpa.providers import make_qm_provider, provider_contract

backend = make_qm_provider({"xc": "pbe", "basis": "def2-svp"})
contract = provider_contract()
```

For embedding pass an explicit `FixedEmbeddingCharges` as `charges`.
The default factory delegates to the established backend without changing
SCF, gradient, force mapping or numerical settings. Both CLIs retain their
distinct Hamiltonians and correction/total-force delivery modes.

## Implemented fixed-region routes

This table lists implemented entry points. Numerical, native, scientific and
performance qualification still belongs to a matching configuration and runtime
record. The registry never substitutes a different engine or method.

| Engine / device | State method | Native modes | Method boundary |
| --- | --- | --- | --- |
| pyscf / cpu | ground | ME, additive EE | Declared molecular method; optional named density fitting |
| pyscf / cpu | rhf-tda | ME, additive EE, plain gap-anchor | Dense singlet fixed-reference TDA |
| pyscf / cpu | pbe0-tda | ME, additive EE, plain or explicitly corrected gap-anchor | Pinned PBE0 settings and complete grid response |
| pyscf / cpu | uks-hirshfeld-cdft | Experimental finite additive NVE only | Explicit charged doublet, PBE0/def2-svpd, frozen Hirshfeld pair; qualification pending |
| gpu4pyscf / cuda | ground | ME, additive EE | Conventional RHF or named fitted PBE/PBE0 |
| gpu4pyscf / cuda | rhf-tda | ME, additive EE, plain gap-anchor | Conventional singlet RHF; separate GPU references |
| abacus / cpu | ground | ME | Cold FileIO, centroid mapping, finite PW/PBE supercell |

Shared CPU RHF/PBE0 and GPU RHF state sets, including prepare-states, are
separate APIs; they do not propagate multiple electronic states in native dynamics. All EE rows above use
explicit fixed charges. Reference-subtractive remains a separate evaluator.
See [CONFIGURATION.md](CONFIGURATION.md) for Hamiltonian selectors and
[MULTISTATE.md](MULTISTATE.md) for shared quantum state evaluation.

## Experimental constrained pair

The separately admitted `uks-hirshfeld-cdft` pair uses the solver/derivative
module in the PySCF fork, actual source/library pins and complete state
records. It does not broaden neutral ground/TDA validation or enable electronic
sampling. See [CDFT.md](CDFT.md) for its fixed-fragment and fail-closed contract.

## Experimental native RHF/TDA route

An explicitly declared CPU RHF/TDA state can use either finite mechanical
embedding or finite additive fixed-charge embedding. The complete immutable
reference must be inline or resolved from an explicitly hash-checked file;
unchecked paths are rejected. See [STATE_INPUTS.md](STATE_INPUTS.md). Inline form:

    from dataclasses import asdict

    configuration["provider"] = {
        "engine": "pyscf",
        "device": "cpu",
        "state": {
            "method": "rhf-tda",
            "reference": reference.to_record(),
        },
    }
    configuration["qm"] = asdict(reference.settings)

Construct the reference with the API described in EXCITED.md. Its settings
must explicitly select tda_solver="dense"; the native route does not select
a solver or create a reference automatically. The dense occupied-virtual
dimension defaults to 256; an explicit reference may request a resource limit
up to 1024. A larger limit does not qualify its extra region/basis combinations
and does not change any residual, force or state threshold. The default iterative experimental API
remains separate from this registered native combination.

The reference fixes the QM atom IDs, elements and order, all RHF/TDA settings,
the seed-state transition density, and the inspected PySCF source identity.
For additive embedding it also fixes environment IDs, elements and order,
individual point charges and charge provenance. Native RHF/TDA requires
radii_angstrom=null; Gaussian references are rejected until a separate real
derivative gate qualifies them. The standalone experimental API is unchanged.
Schema-2 charge requests are resolved first; their complete resulting charge
identity must agree with the reference. Equal net charge is insufficient.
New RHF/TDA JSON inputs reject duplicate keys and nonfinite constants at every
level, including inline reference settings, before checking the reference hash.

All atom positions can evolve while membership and charge values remain fixed.
Reference coordinates define state-overlap calculations; they do not constrain
current MD coordinates. Both routes require a nonperiodic full system. A
different basis, convergence setting, source, device or embedding reference
is rejected rather than silently substituted. DFT grid, NLC and dipole controls
are not RHF/TDA settings and are rejected even when set to a ground-state default.

The factory returns RecordingTDAProvider around the corresponding TDA backend.
A successful evaluation supplies total E_RHF + omega, analytic forces and
electronic-state diagnostics from that same calculation. The wrapper does not
perform a second SCF to obtain diagnostics. The full reference remains in the
provider and restart identity; provider_contract contains only its SHA256
summary. Ground-state requests retain their existing settings and constructors.

Registration does not qualify long trajectories, larger QM regions, other
basis sets, charge models or devices. Source-matched derivative tests,
native force delivery and restart tests are separate evidence gates. The
mechanical route replaces the QM-region energy using the existing subtraction;
the additive route uses its existing explicit cross terms. Neither implies
adaptive atom exchange, mutual polarization, surface hopping or an excited DPA
model. Periodic, arbitrary DFT/TDA and adaptive TDA combinations are unsupported.
The bounded GPU RHF/TDA ME and additive EE routes are described below.

## Bounded PBE0/TDA method

Set `state.method` to `pbe0-tda` and use the complete `PBE0TDAReference`
from PBE0_TDA.md. Set `qm` to `asdict(reference.settings)`; solver, grid level,
pruning and grid response must be explicit. The registry preserves the original
RHF record schema and rejects method/reference/source swaps before construction.
PBE0 defaults to its pinned CPU neutral-singlet def2-SVP/grid4 setting.
An explicit STO-3G/grid1 development pair is available for fixed ME/additive,
matched gap-anchor and CPU fragment routes with unchanged complete-response
controls; see PBE0_TDA.md. Every primary and auxiliary reference must match.
Its dense cap defaults to 256; explicit cap1024 or cap2048 is additionally accepted
as an experimental resource configuration. Each cap is bound to its reference
and restart identity. Larger-region derivatives and native dynamics need their
own evidence; an offline full-system reference does not qualify this provider.

The same-evaluation state record identifies PBE0 and `E_PBE0+omega`, validates
its unrefined orbital response, and requires the complete analytic grid-response
diagnostic for every QM atom. A stale, incomplete or foreign record cannot be
accepted for restart. No additional SCF is performed for state recording.

Finite ME/additive and gap-anchor configurations accept this explicit method.
The gap-anchor route uses the matched PBE0 ground/selected factory and pair zero
with a complete total. API, complete-Hamiltonian, native and scientific evidence
remain separate qualifications tied to their source/runtime/model/reference.
An RHF reference or checkpoint cannot substitute for PBE0. Additive cross
interactions still require explicit parameters for the actual chemistry.

## Optional exact geometry reuse

For either experimental TDA provider, add
`"reuse_identical_geometry": true` beside `engine`, `device` and `state`.
The default is false. The option is retained in the resolved configuration,
provider contract and restart identity; a restart must keep the same option.
The direct TDA constructors accept the same keyword, defaulting to false.

Only the last successful input can be reused. QM/environment IDs, symbols,
coordinates, boundary and force request must match exactly, with the same
immutable settings, state reference, runtime and charges. There is no rounding,
SCF warm start, interpolation or previous-state selection. Changed provider
configuration requires a new provider. Every failure clears the saved result;
`clear_reuse()` also discards it explicitly. Reuse is process-local and starts
empty after a restart. The provider is intended for serial owner-rank calls,
not concurrent calls on the same object.

Returned forces and diagnostics are detached from the saved result. Every native
callback still renews and validates its electronic-state record, even on a hit.
LAMMPS setup, integration, MPI mapping, output and checkpoint operations are
unchanged. `reuse_statistics()` reports requests, hits, calculations and failures;
a hit count is not evidence of a complete-workload speedup. Qualification still
requires matched numerical, native/restart and wall-time measurements.

## Optional ground-state density fitting

An explicit ground-state QM request can include:

```json
"qm": {
  "xc": "pbe",
  "basis": "sto-3g",
  "density_fit": {"auxbasis": "def2-universal-jkfit"}
}
```

This is a small development example, not a recommended chemistry model. The
option uses official PySCF RKS/UKS density fitting for Coulomb and exchange.
The auxiliary basis must be named explicitly; no automatic auxiliary basis,
external basis file, J-only switch or disabled auxiliary-basis response is
accepted. Every fitted force includes the auxiliary-basis response as well as
the declared integration-grid response. Fixed-charge embedding retains the
complete QM and environment derivatives using the same fitted density.

Omitting the option (or null) preserves conventional integrals and the old
normalized settings. Enabling it or changing its auxiliary basis changes the
restart physics identity. Fitting changes the integral approximation: compare
with the same fitted official method for interface tests, and report differences
from conventional integrals separately. Small-system timings do not demonstrate
an end-to-end or large-system speedup.

This option applies to ordinary ground-state ME and additive EE. The existing
RHF/PBE0 selected-state and matched fragment/gap-anchor routes reject it; their
pinned response methods do not inherit ground-state fitting support. Unsupported
PySCF method/gradient combinations fail explicitly without an unfitted fallback.

## GPU ground mechanical provider

The isolated ground combination `gpu4pyscf/cuda/ground` supports fixed finite
mechanical embedding with explicit density fitting, singlet PBE/PBE0 and complete
grid response. See [GPU_BACKENDS.md](GPU_BACKENDS.md) for the request and runtime
scope. Finite additive EE additionally uses the complete GPU point-charge provider.
Selected RHF/TDA ME, additive EE and plain gap-anchor use the separate pinned
provider and matched ground terms described below.

## Selected GPU RHF/TDA

The optional engine gpu4pyscf, device cuda, method rhf-tda route supports
source-pinned singlet conventional TDA references with an explicit dense solver.
It accepts fixed finite ME and additive point-charge EE. State preparation may
declare engine and device at the top level; omission preserves pyscf/cpu.
Embedded preparation also requires environment_geometry and charges. Prepared
GPU files retain their separate reference/runtime identity.

ME uses full native DPA plus the local selected-total-minus-low correction.
Additive EE uses pair zero and returns the complete total, including environment
DPA and explicit cross terms. Accepted EE state records additionally require the
checked total relaxed density and electron-count diagnostics.
Corrected/fragment GPU gap-anchor, selected PBE0/DFT, density fitting, adaptive
partitions and periodic full systems remain rejected. Actual numerical/native qualification
requires a matching source and runtime record.

Conventional GPU RHF ground calculations use the existing gpu4pyscf/cuda/ground
provider with qm.xc="hf" and density_fit absent or null. This selects official
RHF, without DFT quadrature or fitted integrals, for isolated ME or finite
point-charge additive EE. Ground PBE/PBE0 still require named density fitting.
RHF and selected RHF/TDA must share basis/charge/field settings for matched
comparisons. See GPU_BACKENDS.md for derivative and execution boundaries.

The finite plain gap-anchor also accepts gpu4pyscf/cuda/rhf-tda with a pinned
embedded GPU reference. Its three matched quantum terms use the existing GPU
providers and complete derivatives. Corrected/fragment GPU routes remain
rejected. GPU runtime identity is included in native restart checks.


## ABACUS ground mechanical provider

The abacus/cpu/ground combination is admitted only for finite, nonperiodic
mechanical coupling through cold FileIO and arithmetic-centroid coordinates.
It uses an explicit pseudopotential PW/PBE supercell model, distinct from the
molecular PySCF model. A fresh runtime directory and checked shutdown are required.
See [ABACUS.md](ABACUS.md). ABACUS EE/selected/adaptive routes remain unsupported.
