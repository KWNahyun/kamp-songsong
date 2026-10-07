# Model files

`maluq_seed20260930.pth` is the frozen D-FINE-S + MAL + UQ representative
checkpoint used by `run.sh predict` and `run.sh evaluate`. Its SHA-256 is
recorded in `../manifests/model.json` and is verified before inference.

`dfine_s_coco_init.pth` is required only to reproduce MAL training from the
initial COCO-pretrained D-FINE-S model. Obtain it through the upstream D-FINE
release channel and place it in this directory before running `train-mal`.

Do not replace the representative checkpoint without updating the manifest and
documenting the change.
