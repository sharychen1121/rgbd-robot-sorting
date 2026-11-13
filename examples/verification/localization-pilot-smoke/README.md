# Static localization smoke check

Three new standard scenes, seeds 571014–571016, 320×240, confidence .25.
The bundled YOLO26 pilot checkpoint is identified by SHA256 in localization.json.
Predictor and localizer receive no simulator segmentation. True segmentation
and target poses are read by the scorer only, after capture.

Reproduce with a fresh directory:

```powershell
python -m robosort localization-evaluate --count 3 --seed 571014 --width 320 --height 240 --weights examples/checkpoints/yolo26-pilot.pt --output results/localization-replay
```

There are 9 true targets: 4 class-agnostically matched predictions, 5 misses,
0 unmatched false-positive predictions, and 3 correctly classified matches.
Box and mask both have 4/9 valid localization coverage. Conditional mean XY
errors are 6.92/7.13 mm respectively. This tiny sample shows no mask advantage;
the low coverage prevents interpreting millimetre errors as complete success.

Matching globally maximizes cardinality and then bbox IoU at threshold .25,
without true-class filtering. Errors condition on matched, valid predictions,
including wrong classes. Coverage uses all targets. The CSV retains matched,
missed and false-positive rows. This is a static perception diagnostic, not a
sorting-task success evaluation or a formal generalization benchmark.
