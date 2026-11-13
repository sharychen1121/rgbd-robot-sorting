# Learned end-to-end smoke verification

This bundle contains one independently generated, frozen random-color scene
(seed 241118, 320 x 240 RGB-D, standard position/spacing envelope). All three
methods used the same locally trained YOLO26 segmentation checkpoint and the
same guarded execution backend. Confidence 0.25 and Hybrid high threshold 0.8
were retained; no scenes or failures were removed.

This particular scene was chosen for the functional demonstration after an
initial prediction preview of three new seeds (241116–241118); seed 241118
had predictions for all three classes. It is not a pre-registered random
performance sample. The other previews, including misses and duplicates,
remain in `../learning-extension.json`. All methods then replayed this same
frozen scene and every resulting success/failure is retained.

| Method | Correctly placed | Terminal result |
|---|---:|---|
| `yolo_box` | 3 / 3 | completed: all_objects_placed |
| `yolo_mask` | 3 / 3 | completed: all_objects_placed |
| `hybrid` | 0 / 3 | stopped: no_valid_targets |

This is a functional smoke test with a smoke-trained checkpoint, not a formal
comparison or a general success-rate claim. It confirms that learned prediction
can drive a complete sorting task in this scene. Both box and mask succeeded;
this scene does not demonstrate that masks improve task success over boxes.

The original command was:

```powershell
python -m robosort evaluate --seed 241118 --count 1 --regime color --width 320 --height 240 --methods yolo_box yolo_mask hybrid --weights artifacts/yolo26-smoke-30/best.pt --output examples/verification/learned-sorting --plots
```

To replay the exact scene, restore the same checkpoint (SHA-256 in metadata)
and choose a new output directory:

```powershell
python -m robosort evaluate --catalogue examples/verification/learned-sorting/scenes.json --methods yolo_box yolo_mask hybrid --weights artifacts/yolo26-smoke-30/best.pt --output results/learned-sorting-replay --plots
```

`results.jsonl` contains the scored action records and is fingerprinted by
`metadata.json`; `scenes.json` is independently fingerprinted. CSV, summary,
progress journal and three outcome/timing plots are included.

The initial-frame diagnostic was recorded in a separate, action-free replay
after the original evaluation. `initial_predictions.png` shows the model's
predicted masks and labels, `initial_rgb.png` is the RGB input, and
`initial_hybrid_output.png` shows Hybrid's empty output.
`initial_prediction_diagnostic.json` records three raw predictions with
confidences approximately 0.417 (box), 0.348 (cylinder) and 0.295 (cube). Every
predicted class had zero matching fixed-class color contour area within its
box, and none reached Hybrid's 0.8 mask-retention threshold. Hybrid therefore
removed all three candidates under its fixed-color assumption. No ground-truth
segmentation was supplied to inference and no thresholds were changed.

Hybrid's internal filters act before `run_sorting` receives predictions. The
action result records zero returned predictions and `no_valid_targets`; the
empty execution-stage rejection list must not be read as evidence that no
perception filtering occurred. The additional initial-frame diagnostic makes
this distinction explicit.

Elapsed task times are shared-CPU wall-clock measurements. They exclude scene
loading and model load/warm-up, and do not support efficiency conclusions from
this single scene. The run uses a simulated proximity constraint rather than
physical fingers; success requires measured lifting and verified destination
contact/containment.
