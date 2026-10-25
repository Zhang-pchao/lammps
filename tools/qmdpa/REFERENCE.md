# Experimental reference-subtractive embedding

This module compares a declared fixed-charge reference subtraction with existing
mechanical and additive coupling. The finite CPU ground route also supports the
shared LAMMPS driver described below. Dynamic QM membership is not admitted.

```text
E = E_low(full) + E_high_emb(Q; qM) - E_low(Q) - V_ref(Q, M)
V_ref = k * sum(qQ[i] * qM[j] / distance(i, j))
```

`V_ref` is an explicitly chosen point-charge cross interaction. It is **not** an
extracted electrostatic component of a pretrained DPA potential. The low-level
potential can already include effective electrostatic and induction effects;
subtracting this reference need not remove them exactly. A successful force
finite-difference test establishes energy/force consistency, not freedom from
physical double counting or scientific accuracy. No additional cross LJ or
within-environment Coulomb term is inserted.

## Interface

```python
from qmdpa.embedding import FixedEmbeddingCharges
from qmdpa.reference import FixedChargeCrossCoulomb, ReferenceSubtractiveCalculator

reference = FixedChargeCrossCoulomb(
    FixedEmbeddingCharges(qm_ids, qm_charges_e, source=qm_charge_source),
    FixedEmbeddingCharges(environment_ids, environment_charges_e, source=env_charge_source),
)
calculator = ReferenceSubtractiveCalculator(
    full_low, region_low, embedded_qm, reference,
    qm_atom_ids=qm_ids,
    environment_atom_ids=environment_ids,
)
terms = calculator.evaluate_terms(full_geometry)
energy_ev = terms.total.energy
forces_ev_per_angstrom = terms.total.forces
```

The providers retain the existing `Geometry`, `Evaluation` and
`EmbeddedEvaluation` contracts. Optional scientific backends are not imported
by this module. The caller owns provider settings and lifetime, including the
charge and multiplicity of the full and isolated low-level evaluations.

- The two fixed nonempty regions must be disjoint and cover every input atom.
- The full low-level provider receives the configured QM IDs followed by the
  environment IDs. Its per-atom parameters must use that order. Other providers
  receive the configured region orders. Changing the full input order is safe;
  `terms.full_low` and `terms.total` are mapped back to that input order.
- `region_low`, `embedded` and `reference_cross` are unsigned constituents;
  the total subtracts the last reference and the isolated low-level term.
- Both environment and QM force responses are included. Charges are fixed,
  carry provenance, and are independent of geometry. Gaussian reference charges,
  periodicity, covalent cuts, adaptive membership and charge flux are unsupported.
- The caller must declare whether reference environment charges match those
  used for QM embedding. The generic embedded-provider interface does not expose
  an electronic Hamiltonian that this calculator could verify.
- Invalid output or SCF failure propagates; no old result or fallback is used.

`COULOMB_EV_ANGSTROM` and `COULOMB_SOURCE` record the conversion used for charges
in electron units and distances in Angstrom. Per-evaluation diagnostics retain
both charge sources, the conversion and the number of cross pairs.

The model label is `experimental-reference-subtractive`. Benchmark it first on
fixed finite systems against a declared full-QM reference, comparing energy
changes, all forces and electronic response. Parameters transferred from a water
force field do not by themselves qualify a DPA/QM potential.

## Finite CPU ground native configuration

Select `coupling = {"surface": "ground", "embedding": "electrostatic",
"ee_model": "reference-subtractive"}` or the explicit Hamiltonian
`finite-electrostatic-reference-subtractive`. Use the shared `qmdpa validate`
and `qmdpa run` commands.

Declare `qm_region` and `environment_region` as complete molecules with
disjoint IDs, matching symbol orders and explicit full/region charge-spin.
Supply environment `embedding = {"atom_ids": [...], "charges_e": [...],
"source": "..."}` and
`reference = {"form": "fixed-point-charge-cross-coulomb", "qm_charges":
{"atom_ids": [...], "charges_e": [...], "source": "..."}}`.
Reference QM charges must sum to the QM charge; QM plus environment charge
must agree with the full model condition.

This route requires `periodic: false`, CPU PySCF ground state, a raw `.pt`
model and `inference_profile: "torch-raw-deterministic-v1"`.
The callback returns the complete total on a zero native pair baseline.
Both QM and environment derivatives, reference charges, source identities and
physics settings participate in failure/restart validation. Changing reference
charges requires a new trajectory, not reuse of the old checkpoint.

The release screen covers full-coordinate finite differences, translation,
permutation, zero-field reduction, serial/MPI trajectories, cross-rank restart,
actual SCF failure and changed-reference rejection. Periodic, selected-state,
GPU QM and adaptive-reference extensions remain separate. Explicit reference
charges define an experimental Hamiltonian; they do not decompose DPA.
