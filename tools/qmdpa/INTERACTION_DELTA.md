# Experimental finite interaction replacement

`CrossInteractionDelta` evaluates a declared pair of disjoint, complete fragments
C and B using six explicit pair/fragment providers:

    Delta W = (E_H(C+B) - E_H(C) - E_H(B))
            - (U_L(C+B) - U_L(C) - U_L(B)).

`evaluate_terms` records each unsigned constituent, the two interaction energies,
and the correction. Forces follow the same expression and are scattered by atom
ID to the full input order; spectators receive zero. Finite boundaries only.
All providers must use the declared fragment charges, spins, basis convention and
model conditions. The caller owns this provenance; the algebra does not infer it.

This is a replacement of an explicitly calculable cross nonadditivity, without
assuming a unique electrostatic, exchange or dispersion decomposition of DPA.
Isolated fragment energies and the baseline's other/higher-order interactions
remain. It is neither mutual polarization nor a general correction of all
interactions involving B. Applying the same correction to ground and selected
surfaces leaves their excitation gap unchanged.

`CorrectedCalculator` combines an explicit full baseline with an explicit
full-order correction. Its return value is the complete total. It must not be
added to a native full-DPA baseline again. The generic composition API supplies no state history. Numerical, scientific,
native and cost qualification of each combined model remain separate requirements.
Existing configurations retain their original Hamiltonian.

## Explicit native candidate

The selected PBE0 route can be requested with the distinct Hamiltonian
`finite-electrostatic-gap-anchor-cross-delta`, or with
`coupling = {"surface": "selected", "embedding": "electrostatic",
"ee_model": "gap-anchor-cross-delta"}`. It uses the existing zero-pair complete
energy/force transport and state observer. Both the explicit model and its
`interaction_delta` block are mandatory:

```json
{
  "model": "finite-cross-interaction-delta-v1",
  "ground_method": "matched-pbe0",
  "core_region": {"atom_ids": [1, 2], "symbols": ["H", "H"], "molecules": [[1, 2]]},
  "fragment_region": {"atom_ids": [3, 4], "symbols": ["H", "H"], "molecules": [[3, 4]]},
  "charge_spin": {"pair": [0, 1], "core": [0, 1], "fragment": [0, 1]}
}
```

This block illustrates schema only, not an accuracy-qualified molecular model.
C must contain complete molecules from the existing QM region; the fragment must
contain complete molecules from the existing environment region. Symbols and
parent molecule memberships must agree. This native candidate supports neutral
singlets, the same fixed model, and PBE0 ground settings matched to the anchor's
basis/grid/convergence settings. It introduces no different electronic solver.
The entire block and distinct model ID enter the physics hash and restart checks.
The uncorrected configuration rejects this block rather than silently enabling it.
A ground-only corrected native route is not provided by this selected-state driver.

The matched CPU PBE0 ground terms also accept the explicit STO-3G/grid1
profile described in PBE0_TDA.md. The selected anchor and every correction term
must share that profile. No automatic downgrade, relaxed response tolerance or
conversion of an existing state reference occurs. Its accuracy is separate
from that of the default def2-SVP/grid4 model.
