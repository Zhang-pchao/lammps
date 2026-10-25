# Configurable offline region selection

This optional tool proposes QM atom IDs from saved finite coordinates. It supports
water-nearest geometry rules or explicit fixed groups for arbitrary chemistry.
It does not alter coordinates, velocities, forces, the Hamiltonian, or the
existing fixed-region LAMMPS driver. It does not supply a continuous proton
indicator or an energy-conserving adaptive partition algorithm.

## Water-nearest API

```python
from qmdpa.selection import WaterRegionSelector

selector = WaterRegionSelector(
    center_oxygen_ids=(1, 5),
    oxygen_count=12,
    max_oh_distance=1.35,
    shared_distance_gap=0.10,
)
proposal = selector.select(geometry)
```

`geometry` is the existing immutable `qmdpa.core.Geometry`, with Angstrom
coordinates, stable positive atom IDs, only O/H, and no periodic cell. Each call
is independent; the selector never retains a previous successful region.
Thresholds are explicit geometric choices, not universal chemical parameters.

Each H is assigned to its nearest O only if its O-H distance is positive and
at most `max_oh_distance`, and the second-nearest O is farther by more than
`shared_distance_gap`. Otherwise an issue is recorded and the entire proposal
has `status="unresolved"` with empty selected-ID tuples. An O with one, two,
or three assigned H atoms is labeled `hydroxide-like`, `water-like`, or
`hydronium-like`. These labels describe geometry, not electronic charge or spin.
An O with any other count also withholds the proposal. Ambiguous frames are
reported, not silently assigned to a molecule or replaced by the previous frame.

One or two explicit oxygen centers are required. For one center, other O atoms
are ranked by distance to it. For two centers, they are ranked by the sum of
the two distances, whose level sets are ellipsoids. Both centers are included
before filling the remaining oxygen budget. Selected groups contain every H
assigned to their O. Scores and reasons are returned for every O. Exact score
ties are ordered by stable atom ID; near-ties across the budget boundary are
reported using `score_tolerance`. Changing labels in a degenerate tie can change
which equally ranked group is proposed. This is an offline diagnostic policy,
not a smooth partition function.

The frozen result contains groups, nearest-neighbor assignments, selected IDs,
scores, boundary ties, and issues. Input ordering may change without changing
atom identity. No cell imaging or water-wire graph is inferred. The distance
ranking sorts all O atoms for each H, with O(N_H N_O log N_O) work. Large
droplet screening may later need a separately verified spatial neighbor search.

## Versioned settings

Use an explicit offline policy file for reusable configuration:

```sh
python -m qmdpa.selection_cli --input frames.json --output selection.json \
  --settings region-settings.json
```

A water-nearest policy is:

```json
{
  "schema_version": 1,
  "mode": "offline",
  "method": "water-nearest",
  "centers": [1, 5],
  "oxygen_count": 12,
  "max_oh_distance": 1.35,
  "shared_distance_gap": 0.10,
  "score_tolerance": 1e-10
}
```

`schema_version`, `mode`, `method`, `centers`, and `oxygen_count` are required.
The three thresholds may be omitted and then receive the displayed defaults.
All three are positive finite numbers; `shared_distance_gap` must be smaller
than `max_oh_distance`. Distances and score tolerances are in Angstrom.

| Setting | Default | Geometric assumption |
| --- | --- | --- |
| `max_oh_distance` | 1.35 | Largest accepted nearest O-H distance |
| `shared_distance_gap` | 0.10 | Required separation between the nearest two O-H distances |
| `score_tolerance` | 1e-10 | Distance-score gap reported as a boundary tie |

These are configurable geometry assumptions for finite-water diagnostics, not
universal chemical constants. Choose and record them before examining outcomes;
changing them does not change the water selector's algorithm or validate a new
chemical domain. Shared hydrogens still withhold the entire region proposal.

For arbitrary chemistry, declare complete fixed groups and their element
identities. No bond, molecule, charge, or electronic state is guessed:

```json
{
  "schema_version": 1,
  "mode": "offline",
  "method": "fixed-groups",
  "groups": [
    {"group_id": "oxide", "atom_ids": [1, 2], "symbols": ["Ti", "O"]},
    {"group_id": "environment", "atom_ids": [3], "symbols": ["Ar"]}
  ],
  "selected_group_ids": ["oxide"]
}
```

Every group needs a distinct, nonempty string `group_id`, nonempty distinct
positive integer `atom_ids`, and one explicit `symbols` entry per atom. Groups
must be disjoint and together cover the complete input atom-ID/element mapping,
including unselected groups. Missing, extra, or changed identities are errors.
`selected_group_ids` must be a nonempty distinct subset of declared group IDs.
Selection always includes every atom of a selected group. Atom IDs, group IDs,
and resolved settings are returned in stable order independent of input order.
The selected group membership remains fixed across frames; moving coordinates
never cause this policy to guess a new group.

Both policies require `mode: "offline"`. Unknown fields, unsupported versions
or methods, duplicate JSON fields, nonfinite values, and booleans in numeric
fields are rejected. Water-only fields are rejected in a fixed-groups policy.
An adaptive/dynamics setting is an error, not a request that is silently reduced
to offline analysis. Settings do not include charges, multiplicities, link
atoms, force weights, or MD update intervals.

The Python entry point `qmdpa.selection_config.selection_from_settings(document)`
returns `(selector, resolved_settings)`. Use `selector.select(geometry)` with the
existing `Geometry`. `FixedGroup` and `FixedGroupSelector` also support direct
Python construction with explicit immutable groups. Neither policy changes the
input or retains state from a previous frame.

## Input and report

The original water CLI remains supported:

```sh
python -m qmdpa.selection_cli --input frames.json --output selection.json \
  --centers 1 5 --oxygen-count 12
```

It accepts the optional `--max-oh-distance`, `--shared-distance-gap`, and
`--score-tolerance` flags with the same defaults and numerical behavior.
`--settings` cannot be combined with **any** selector flag, including a flag
whose value equals the default. This avoids hidden precedence between a file
and command-line options. Without `--settings`, both `--centers` and
`--oxygen-count` are required.

Input is either one geometry object or `{"frames": [geometry, ...]}`. Every
geometry supplies `units: "angstrom"`, `boundary: "nonperiodic"`, `atom_ids`,
`symbols`, and `positions`. All frames must retain the same atom-ID/element
mapping; atom order may change. Both policies use the existing `Geometry`
validation. A periodic flag or cell is rejected. The output path must not exist.
Invalid input produces no report; unresolved water assignments produce explicit
diagnostics in a successful report.

The report retains `source.sha256`, the selector configuration, frame counts,
and per-frame decisions. The new `settings` object includes `resolved`,
`origin`, `name`, `sha256`, and `hash_basis`. For a settings file, the hash covers
its exact bytes (`file-bytes`). For legacy flags, it covers the complete resolved
settings serialized with sorted keys and compact separators (`canonical-json`).
The water result fields remain unchanged. A fixed-groups result contains
`status`, `selected_atom_ids`, `selected_group_ids`, and the declared `groups`.
The report contains no live MD state, electronic-state assignment, or
authorization to change an active simulation's region.

## Method context and validation boundary

The idea of tracking ion-centered water regions and merging them geometrically
is discussed by Kwon et al., *J. Am. Chem. Soc.* **147**, 6583-6593 (2025),
[doi:10.1021/jacs.4c15103](https://doi.org/10.1021/jacs.4c15103).
This implementation was written independently; it does not copy their
CC BY-NC supplemental code or reproduce their complete ReQM workflow.

Tests cover whole-group selection, one/two centers, ionic structural labels,
shared-proton ambiguity, changed H ownership, stable large IDs, input ordering,
rigid motion, explicit ties, strict versioned settings, explicit nonwater groups,
complete identity coverage, report provenance, legacy CLI compatibility, mixed
option rejection, and no-overwrite behavior. Passing those tests establishes the selection contract only. Dynamic
partition weights, their gradients, electronic-state changes, QM/MM crossing
forces, periodic geometry, and physical proton-transfer accuracy remain separate
milestones.
