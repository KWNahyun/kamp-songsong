# Model files

`maluq_seed20260930.pth` is the frozen D-FINE-S + MAL + UQ representative
checkpoint used by `run.sh predict` and `run.sh evaluate`. Its SHA-256 is
recorded in `../manifests/model.json` and is verified before inference.

`dfine_s_coco_init.pth` is the project-specific single-class initialization
used to reproduce MAL training. Its SHA-256 is
`0ac6124d45341889b9999b2294540cc6105add3679fe6182ea692c9e6917f531`.
It was derived from the D-FINE COCO checkpoint as described in
`../THIRD_PARTY_NOTICES.txt`.

Do not replace the representative checkpoint without updating the manifest and
documenting the change.
