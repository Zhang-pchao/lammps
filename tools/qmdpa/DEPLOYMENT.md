# Installing a reproducible software candidate

Build and install the wheel with an already selected compatible Python stack.
Keep the source checkout, package installation and calculation outputs separate.
For example, use explicit paths chosen for the target machine:

    "$QMDPA_PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$QMDPA_WHEEL_DIR" .
    "$QMDPA_PYTHON" -m pip install --no-deps --target "$QMDPA_PACKAGE_DIR" "$QMDPA_WHEEL"
    PYTHONPATH="$QMDPA_PACKAGE_DIR" "$QMDPA_PYTHON" -m qmdpa.cli --help
    PYTHONPATH="$QMDPA_PACKAGE_DIR" "$QMDPA_PYTHON" -m pytest tests

This does not install scientific backends or native libraries. NumPy, the
chosen PySCF/GPU4PySCF or ABACUS interface, DeePMD, Torch and the LAMMPS shared
library must already match the recorded runtime. MPI4Py and LAMMPS must use
compatible MPI libraries. Do not place a different source Python wrapper ahead
of the installed LAMMPS library. On clusters, imports and tests belong inside
the scheduler allocation.

Use qmdpa validate before qmdpa run. Supply original model/data files and their
expected hashes, explicit provider/method, fixed charge/cross provenance when
needed, and a new output directory. The raw-model routes require the strict
inference environment declared by qmdpa.inference; do not silently choose a
compiled model or another device when it fails.

Optional CPU RHF/TDA requires the source-pinned PySCF response-hook variant in
EXCITED.md; it is not provided by this wheel. Rebuild its source overlay with the companion repository's
environment/build_pyscf_response_overlay.py and pinned response recipe.
Verify the generated manifest with environment/verify_stack.py before the
tiny real-provider checks. Keep the resulting source and native-library
identities in the runtime manifest. Generated links are local to that runtime;
copy the portable recipe and regenerate them on another machine. PBE0 and GPU selected profiles
use their own source contracts. A portable source package is not proof that
an arbitrary dependency stack is compatible.

Before promoting a runtime, record the wheel SHA256, all installed module
hashes, native build identities, dependency/overlay hashes, model revision,
commands, settings and scheduler resources. Check an installed-package tiny
real E/F case and the relevant native mapping, MPI, restart and failure paths.
Keep logs and results outside the source package. Review the backend limits in
CAPABILITIES.md. Default unit-test skips for unavailable real backends are not
numerical qualification.

Changing package/source/dependency identity starts a new run. Do not edit an
old checkpoint's physics hash to make it load. A deployment does not replace
the separate scientific, long-dynamics or complete-cost comparisons needed for
an application. Official upstream submission requires its own human review.

## SA-CASSCF NAC overlay

The two-state Ehrenfest provider requires the source-pinned PySCF 2.14.0 bra
response-offset correction. Build an isolated overlay with the companion's
`environment/build_pyscf_nac_overlay.py --base <installed-pyscf> --source
<reviewed-pyscf-source>/nac/sacasscf.py --rhf-source
<reviewed-pyscf-source>/grad/tdrhf.py --output <fresh-output>`. It validates
both the NAC correction and retained RHF response-hook hashes, records every installed input and
leaves the base untouched. Put `<fresh-output>/python` on PYTHONPATH and set
PYSCF_EXT_PATH to the parent of the installed PySCF package. The prepared pair
reference additionally binds all provider source modules. A changed source
requires fresh reference and runtime qualification; the version label alone
never identifies this dependency.
