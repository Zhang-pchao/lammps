# Fixed-region QM/DPA water regression

This example exercises the real LAMMPS DeePMD pair style plus the molecular QM
correction in `tools/qmdpa`. It is a formal small-system integration regression,
not evidence that a pretrained model describes equilibrium liquid water.

The two asymmetric waters are project-generated coordinates. Their atom IDs,
elements and positions match `qm-dpa/benchmarks/water/dimer.json` in the companion
integration repository; `fixture-provenance.json` records that input checksum.
The first complete water (IDs 1-3) is the fixed QM region. All atoms move.
`water-dimer.data` uses atomic style with explicit masses and zero image flags.
Small deterministic velocities are paired to give zero total linear momentum;
they are not a thermal sample. The box is nonperiodic and all molecules begin
intact. No classical bond or angle terms are added.

## Prerequisites and model preparation

Use the `lammps-qmdpa` package from `tools/qmdpa`, an MPI-compatible shared LAMMPS
build with EXTRA-FIX and the DeePMD pair style, the matching DeePMD installation,
and PySCF. Run actual evaluations inside a compute allocation.

Supply a real `.pt2` exported and validated on the target GPU/runtime. A raw
`.pt` checkpoint is suitable for separate backend validation, not this native
pair-style input. For DPA4C, use the compressed archive after separately
validating the frozen and compressed outputs. Keep inference TF32 and AMP off
for this numerical regression and record their export-time values.

`run.json` is deliberately invalid until both model placeholders are replaced.
`prepare_config.py` checks the supplied file's extension and SHA256 and writes a
new configuration with absolute file references. Preparation does not load the
archive or establish its model/runtime compatibility.

## Fresh run

From this example directory, with the environment already loaded:

```bash
MODEL=/absolute/path/to/validated-model.pt2
MODEL_SHA256=REPLACE_WITH_64_HEX_SHA256_OF_PT2
WORK_DIR=/absolute/path/to/a/new/regression-directory

python prepare_config.py --model "$MODEL" --sha256 "$MODEL_SHA256" \
  --threads 4 --output "$WORK_DIR/run.json"
qmdpa run "$WORK_DIR/run.json" --output "$WORK_DIR/first-10"
```

The template runs 10 additional NVE steps at 0.25 fs, saving each accepted step.
PBE/def2-SVP, SCF tolerance `1e-10`, and grid level 4 are engineering settings;
they do not reproduce OMol25's training reference method. The explicit initial
velocities are used because the configuration does not request random velocity
initialization. For an unconditioned model, pass `--unconditioned` to preparation;
for OMol models retain the explicit neutral-singlet conditions `[0, 1]`.

## Restart and continuity check

Continue from the accepted step-10 checkpoint into a new output directory:

```bash
qmdpa run "$WORK_DIR/run.json" --output "$WORK_DIR/continued-10" \
  --restart "$WORK_DIR/first-10/checkpoint-000000000010.json"
```

`steps` always means additional steps, so this continuation ends at step 20.
Use the companion JSON checkpoint, not the binary file by itself. The driver
checks model/input identity and recomputed state before advancing. It does not
reinitialize velocities on restart. A failed segment must be discarded; only
an accepted checkpoint is a recovery source.

For an uninterrupted comparison from exactly the same initial data:

```bash
python prepare_config.py --model "$MODEL" --sha256 "$MODEL_SHA256" \
  --threads 4 --steps 20 --output "$WORK_DIR/continuous.json"
qmdpa run "$WORK_DIR/continuous.json" --output "$WORK_DIR/continuous-20"
```

Compare step-20 accepted frames from `continued-10/frames.jsonl` and
`continuous-20/frames.jsonl`, retaining differences in coordinates, velocities,
forces and total energy. For an unconditioned model use `--unconditioned` in both
preparation commands. MPI rank comparisons require the same MPI implementation
for LAMMPS and mpi4py and a model export supporting that rank configuration.

Completion of these commands establishes neither timestep convergence nor
water accuracy. Review energy drift, finite-difference diagnostics and restart
agreement separately. Correction virial and hybrid pressure are not qualified;
this fixed-cell example must not be repurposed as an NPT validation.
