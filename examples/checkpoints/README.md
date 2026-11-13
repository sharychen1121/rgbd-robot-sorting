# Bundled pilot model

`yolo26-pilot.pt` is the actual project-trained YOLO26n-seg checkpoint used by
the learned end-to-end verification. It predicts `cube`, `cylinder`, and `box`.
It is provided so a fresh checkout can run the learned demonstration without
first training a model. The file is included in ordinary Git, not an external
download or a missing placeholder.

- Bytes: 6,566,173
- SHA-256: `dd22994db1c3bc4cac73b816087bb0d11ec5d930653853cd1018a173b0fd04ac`
- Original artifact: `artifacts/yolo26-smoke-30/best.pt`
- Training: 90 independent synthetic scenes, randomized colors/lighting,
  320 x 240 RGB, seed 241114, CPU, nano instance segmentation.
- Lineage: 5 epochs, then a new optimizer with a maximum of 25 further epochs;
  early stopping ended stage 2 at epoch 20 and chose its epoch-12 checkpoint.
- Retained 15-image partition retest: box mAP50 .6912, mask mAP50 .5079.
  That partition was inspected during the pilot and is not a pristine final test.

The small-data model still misses objects and returns duplicate candidates.
It is a functional demonstration artifact, not the final research model.
One fresh random-color scene completed all 3 placements using its predictions;
that result does not establish general reliability or mask superiority.

From the repository root, after installing `.[learning]`:

```powershell
python -m robosort demo --catalogue examples/verification/learned-sorting/scenes.json --method yolo_mask --weights examples/checkpoints/yolo26-pilot.pt --output results/bundled-mask-demo --gui --plots
```

Remove `--gui` for headless execution. To replay the box/mask/Hybrid smoke
comparison, use the same catalogue with `evaluate --methods yolo_box yolo_mask hybrid`.
Full training recipe and original model hashes are in `../verification/learning-extension.json`
and the corresponding Python replay scripts. Generated datasets and training
logs are rebuilt locally; they are not embedded in this file.

This checkpoint is fine-tuned using Ultralytics' AGPL-licensed implementation
and official pretrained YOLO26 weights. It is distributed as part of this
AGPL-3.0-or-later research project; upstream model/dependency notices apply.
