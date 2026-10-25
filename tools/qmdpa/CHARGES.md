# Explicit fixed-charge sources

The optional charge resolver prepares fixed external charges for the existing
finite, nonperiodic PySCF embedding provider. It does not infer molecules from
coordinates, assign electronic charge/spin, fit a charge model, or supply a
complete water force field. Lennard-Jones parameters remain separately explicit.

## API and source contract

```python
from qmdpa.charges import resolve_fixed_charges
from qmdpa.embedding import FixedEmbeddingCharges

request = {
    "kind": "water-preset",
    "units": "e",
    "source": "Explicit finite-water diagnostic choice",
    "expected_total_charge_e": 0,
    "preset": "TIP3P-original-1983",
    "groups": [[4, 5, 6]],
}
resolved = resolve_fixed_charges(
    request, [4, 5, 6], ["O", "H", "H"], base_directory="inputs"
)
charges = FixedEmbeddingCharges(**resolved["embedding"])
```

Every request requires `kind`, `units: "e"`, a nonempty `source`, and
`expected_total_charge_e`. Charges and expected totals must be finite real
numbers, not booleans. Atom IDs must be positive Python integers. The source
table must cover every declared atom exactly once; there is no default charge,
element fallback, or implicit all-zero result. Explicit zeros are permitted.

Exactly one of these source forms is selected:

| kind | Additional fields |
| --- | --- |
| `per-atom` | `charges: [{"atom_id": 4, "charge_e": -0.834}, ...]` |
| `atom-types` | `atom_types: [{"atom_id": 4, "type": "water-O"}, ...]` and `type_charges: [{"type": "water-O", "charge_e": -0.834}, ...]` |
| `water-preset` | `preset` and `groups: [[O_ID, H_ID, H_ID], ...]` |
| `file` | `path` and the complete `sha256` of the external JSON bytes |

Type names are caller-defined labels, separate from chemical elements and
LAMMPS numeric types. Every atom needs an explicit type assignment. Different
oxygen types may have different charges; a water O charge is never applied to
all oxygen atoms. Duplicate IDs, repeated type definitions, unknown fields,
missing assignments and unused type-charge entries are rejected.

## Water charge templates

| Exact preset name | O charge (e) | H charge (e) | Official parameter source |
| --- | --- | --- | --- |
| `TIP3P-original-1983` | -0.834 | 0.417 | [LAMMPS TIP3P](https://docs.lammps.org/Howto_tip3p.html) |
| `SPC` | -0.820 | 0.410 | [LAMMPS SPC](https://docs.lammps.org/Howto_spc.html) |
| `SPC/E` | -0.8476 | 0.4238 | [LAMMPS SPC](https://docs.lammps.org/Howto_spc.html) |

Groups explicitly identify distinct, present atoms in O,H,H order. Groups must
be disjoint and cover the complete target atom set. Element labels validate
these declarations; geometry is not read or used to infer a molecule. Ions,
hydroxide/hydronium, oxide oxygens and other species require an independently
declared per-atom or atom-type table. No charge/spin follows from a preset.

The presets provide charges only. They add no bonds, constraints, geometry,
LJ parameters, polarization correction, or guarantee of compatibility with a
DPA/QM Hamiltonian. In particular, the SPC/E charge template does not add the
SPC/E energy correction. TIP4P/OPC presets are rejected: the supported interface
does not implement their required model-specific charge-site/force mappings.

## Overrides and total charge

An optional list replaces charges only after the complete base table is resolved:

```json
"overrides": [
  {"atom_id": 4, "charge_e": -0.800, "source": "Explicit diagnostic adjustment"},
  {"atom_id": 5, "charge_e": 0.400, "source": "Compensating adjustment"},
  {"atom_id": 6, "charge_e": 0.400, "source": "Compensating adjustment"}
]
```

Overrides cannot introduce atoms or repeat an ID. Each needs its own source.
The resulting total is checked against the declared expected total with an
absolute tolerance of 1e-8 e and zero relative tolerance, after all overrides.
Nonzero totals are supported; neutrality is not assumed. Overrides mark the
result `customized: true`, including overrides that retain the same value.
A modified preset is explicitly a custom charge template.

## External JSON and provenance

Paths resolve relative to `base_directory` (the configuration directory in the
driver). Absolute paths are also accepted. File bytes are read once, checked
against the supplied SHA256, then parsed with duplicate-key and nonfinite-number
rejection. The external document is a flat per-atom table, with no nested files:

```json
{
  "schema_version": 1,
  "units": "e",
  "source": "Explicit imported fixed-charge table",
  "charges": [
    {"atom_id": 4, "charge_e": -0.834},
    {"atom_id": 5, "charge_e": 0.417},
    {"atom_id": 6, "charge_e": 0.417}
  ]
}
```

The returned `embedding` contains ordered IDs, explicit charges and a source
description. The JSON-safe `provenance` contains the resolver version, target
IDs/elements, base values, source details, overrides, declared/actual totals and
tolerance. A file source includes its exact SHA256 and internal source text.
No filename or absolute location enters this record: moving identical bytes
does not change the resolved identity. Output follows the supplied atom-ID order;
table, type-definition, water-group and override order otherwise do not matter.

## Additive configuration versions

Schema 1 retains its existing explicit `embedding` fields and checks. Schema 2
uses the same top-level configuration keys, with this embedding form:

```json
"embedding": {
  "charge_request": {
    "kind": "water-preset",
    "units": "e",
    "source": "Explicit diagnostic template",
    "expected_total_charge_e": 0,
    "preset": "TIP3P-original-1983",
    "groups": [[4, 5, 6]]
  }
}
```

An optional `radii_angstrom` array alongside `charge_request` retains the
existing fixed Gaussian-site behavior and follows environment atom order.
Omitting it retains point charges. The loader resolves once, replaces
`embedding` with the explicit provider fields, and records `charge_resolution`
in the physics/restart identity. External file contents and override provenance
therefore participate in restart checks. The driver still requires the resolved
total to match the explicitly declared DPA environment charge; it does not infer
that charge or either region's multiplicity.

Schema-2 configuration JSON rejects duplicate keys and nonfinite constants.
Mixed version-1/version-2 fields and extra fields are rejected. Resolved runtime
records are not configuration input files. Preserving schema-1 behavior does not
preserve an older source checksum: software changes still require a new qualified
restart identity. This module adds no adaptive charges or new electrostatics.
