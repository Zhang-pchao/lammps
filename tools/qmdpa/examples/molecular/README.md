# Tiny molecular exchange input templates

These generated eight-hydrogen inputs exercise an explicit core plus one of
three H2 candidates. Velocities deliberately move candidates across the outer
weight radius within 0.4 fs. They are software examples, not chemical trajectories.
EE charges and LJ parameters are artificial interface controls.

Copy the files into a new calculation directory. Supply an appropriate original
raw DeePMD model as model.pt and replace the all-zero model_sha256 in the chosen
JSON with that file's actual SHA256. Confirm its element and neutral-singlet
conditioning contract. The example does not distribute weights or select a
pretrained model automatically.

With the declared runtime and strict raw inference environment active:

    qmdpa validate me.json
    qmdpa run me.json --output me-run
    qmdpa validate ee.json
    qmdpa run ee.json --output ee-run

Use a new output directory for every call. The JSON also accepts the equivalent
explicit coupling selectors described in CONFIGURATION.md. MPI and checkpoint
continuation use the shared command documented there. The examples use CPU
HF/STO-3G only; they do not qualify a larger system, another method or long-time
dynamics.
