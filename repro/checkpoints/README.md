# Model files (not committed)

This directory is intentionally empty in the public repository. KAMP data and
trained weights are not redistributed here.

Place the following files here when they are supplied through an authorized
team channel:

- `maluq_seed20260930.pth`: frozen D-FINE-S + MAL + UQ representative checkpoint.
- `dfine_s_coco_init.pth`: single-class D-FINE-S initialization used before MAL training.

`manifests/model.json` records the expected SHA256 of the frozen checkpoint.
`run.sh predict` verifies that hash before inference. Do not replace a file at
this path without updating the experiment manifest and documenting the change.
