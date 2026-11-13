# Three-scene backend smoke verification

This is a small functional smoke test on three frozen standard scenes (seeds
43, 44 and 45), with `oracle` and `hsv` replaying the same scenes independently.
It is not a formal method comparison, a claim of general success rate, or a
deep-learning result. No scenes were filtered according to execution success.

Reproduce from the repository root using a new, empty output directory:

```powershell
python -m robosort evaluate --catalogue examples/verification/random-reference/scenes.json --methods oracle hsv --output results/random-reference-replay --plots
```

The original generation and execution command was:

```powershell
python -m robosort evaluate --seed 43 --count 3 --regime standard --methods oracle hsv --output examples/verification/random-reference --plots
```

`results.jsonl` holds complete scored outcomes and candidate rejection events;
`trial_results.csv` and `summary.json` are derived from those records.
`metadata.json` fingerprints the source records and records environment,
controller budgets, the scene catalogue hash and smoke-test scope. `scenes.json`
is the frozen, fingerprinted scene catalogue. `progress.jsonl` is the original
flushed execution journal before final result bundling.

Elapsed task time is wall-clock time on a shared CPU. It includes controller
setup, sensor captures, planning and motion, but excludes scene loading and any
model loading/warm-up. Predictor and localizer time fields record their own
cumulative calls. Timing from this small run must not be interpreted as a
method efficiency comparison.

Physical execution uses a fingerless KUKA with a checked proximity constraint,
joint limits, collision checks, motor-driven trajectories, measured lifting,
and destination containment/contact verification. Oracle reads true target
poses as a backend diagnostic. HSV inference receives RGB-D without privileged
segmentation, and its predicted class is not corrected using the true class.
