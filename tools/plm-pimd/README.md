# Regional mixPIMD installation

This entry installs one locked combination of the LAMMPS, DeePMD-kit and
PLUMED forks. Optional installation groups provide molsimflow PIMD analysis
and the i-PI reference implementation. The component commits are recorded in
components.json. The selected entry commit supplies the LAMMPS revision.

## Build environment

Load a C++ compiler, CUDA toolkit, CUDA-compatible MPI and CMake before
starting the installation. Use a Python environment with PyTorch 2.11.0.
The installer creates a project virtual environment that shares its base
Python dependencies, then installs the locked project packages there.

Qualified GPU profiles are CUDA architecture 70 with the VOLTA70 build and
architecture 86 with the AMPERE86 build. The qualified Torch wheel uses CUDA
12.6. Source builds were exercised with CUDA toolkits 12.6 and 12.9.
Python 3.12 was used for qualification.

Run the build inside the resources allocated by your scheduler. Compilation
uses the requested job count. Native MPI verification launches eight ranks.
Keep the compiler, MPI and CUDA modules consistent when activating the result.

## Select the integration branch

Set an external build workspace and a fresh versioned installation prefix.

~~~bash
PLM_WORKSPACE="$PWD/plm-pimd-work"
PLM_PREFIX="$PWD/mixpimd-consolidation-v1"
mkdir -p "$PLM_WORKSPACE/sources"
git clone --depth 1 --branch mixpimd/main \
  https://github.com/Zhang-pchao/lammps.git \
  "$PLM_WORKSPACE/sources/lammps"
PLM_ENTRY="$PLM_WORKSPACE/sources/lammps"
~~~

Choose the GPU architecture and the Python containing the selected Torch
build. The following command also installs analysis and reference packages.

~~~bash
python "$PLM_ENTRY/tools/plm-pimd/install.py" \
  --prefix "$PLM_PREFIX" \
  --workspace "$PLM_WORKSPACE" \
  --python "$(command -v python)" \
  --cuda-root "$CUDA_HOME" \
  --cuda-arch 86 \
  --jobs 8 \
  --with-analysis \
  --with-reference
~~~

Use architecture 70 for a V100 target. Optional dependency archives can be
supplied with --dependency-archives DIRECTORY. The expected filenames are
googletest-1.18.0.tar.gz and yaml-0.2.5.tar.gz. LAMMPS verifies their configured
checksums.

The installer fetches full commit IDs into the workspace, builds outside the
source checkouts, and writes installation.json with the resolved five-component
identity. Completed stages are recorded after successful execution. Repeating
the same command resumes that installation. A different component combination
uses a new versioned prefix.

## Activate and export a model

Load the recorded toolchain, then activate the installed environment.

~~~bash
source "$PLM_PREFIX/env.sh"
"$PYTHON" -m deepmd --version
"$LAMMPS_EXE" -h
~~~

Export the checkpoint on the target GPU through the experimental PyTorch
backend. Supply your own trained checkpoint.

~~~bash
"$PYTHON" -m deepmd --pt-expt freeze \
  -c "$MODEL_CHECKPOINT" -o "$MODEL_ARCHIVE" --compress
~~~

GPU archives contain architecture-specific compiled code. Retain the original
checkpoint and its checksum for target export and numerical comparisons.

## Verify the model, PIMD and bias interface

Prepare an atomic LAMMPS data file and a compatible PLUMED input. The latter
defines the desired CV and its shared ENSEMBLE bias argument for cv_mean.
Element labels follow the data type ordering. H and O below are an example.

The launcher must provide GPU access for the bead partitions. For a shared
single GPU, use the site's established CUDA MPS setup. The helper accepts a
site launcher with --mpi-command.

~~~bash
"$PYTHON" "$PLM_ENTRY/tools/plm-pimd/verify.py" \
  --model "$MODEL_ARCHIVE" --data "$ATOMIC_DATA" --type-map H O \
  --plumed-input "$PLUMED_INPUT" --output "$FUNCTIONAL_OUTPUT" \
  --beads 4 --contraction 0.5 --steps 20
~~~

The default quantum region covers every atom. A spherical regional check uses
--quantum-region sphere together with --center X Y Z, --core R and --outer R.
Match the PLUMED region and the PIMD region to the intended sampling input.

The helper saves the command, model/input hashes and finite-state verification
in verification.json. Native installation tests exercise adaptive mapping,
eight-rank MPI, bias coupling and restart. Model/checkpoint numerical agreement
and physical sampling convergence require their own qualification inputs.

## Interface and model scope

The paired DeePMD provider and LAMMPS plugin use C API31 and the project
selected-energy backend factory. Build them from the locked component set.

Complete-PES evaluation is the default model path. Explicit selected-energy
evaluation admits eligible one-hop short-range exports, including audited
compressed variants with folded neutral-singlet conditioning. The model
capability checks and complete neighborhood derivatives apply to both
selected and complement evaluations.

The adaptive integrator provides opt-in force, velocity and bias-moment
reductions on qualified one-MPI-rank-per-bead layouts. Export
LAMMPS_PIMD_FORCE_SUMMARY_EXPERIMENT=1,
LAMMPS_PIMD_VELOCITY_SUMMARY_EXPERIMENT=1 and
LAMMPS_PIMD_BIAS_MOMENT_EXPERIMENT=1 to request these paths. Leave them unset
to use the established gather path. Set the same options for every bead.

Bias path contraction and spatial quantum amplitudes have separate meanings.
Keep cv_mean, coordinate_mean and bias_mean definitions explicit in sampling
inputs. A cv_mean calculation uses one synchronized bias history. The regional
ensemble and its normalization are described in doc/src/fix_pimd_adaptive.rst.

The milestone qualification covers short functional execution and numerical
interface checks on the two GPU profiles, including MPI and checkpoint/state
restoration. Long-time stability, bead/region convergence, reactive free
energies and throughput improvements have separate acceptance criteria.

## Traceability and rollback

Keep installation.json, the source component commits, the model/checkpoint
checksums and the activation script with each deployment. A previous versioned
prefix, including the plm-pimd-api31-v1 installation, can be selected from a fresh shell through its own env.sh. The source
checkouts and build workspace are distinct from the installed prefix.
