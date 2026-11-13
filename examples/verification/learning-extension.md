# Two-stage learning verification

YOLO26n-seg first ran for 5 epochs. Its best checkpoint initialized a separate run with a fresh optimizer and a maximum of 25 epochs; patience 8 stopped that stage at 20 epochs and selected its epoch-12 checkpoint. This is not an uninterrupted 30-epoch experiment. Both stages used the same 90-scene pilot dataset.

Retained 15-image test partition retest: box mAP50 0.6912, mask mAP50 0.5079, mask mAP50-95 0.1806. The test partition was already inspected after stage 1, so this is a pilot retest rather than a pristine final benchmark. A formal comparison still needs a new frozen test set.

Fresh seeds 241116, 241117, and 241118 produced 2, 4, and 3 candidates at confidence 0.25; all have predicted masks and valid RGB-D points, but misses and duplicates remain. Seed 241118 predicted all three classes. Empty seed 241119 produced 0 candidates. Camera frames had no simulator segmentation input.

Checkpoint size, SHA256, full metrics, lineage, and individual candidates are in learning-extension.json. Image detections and 3D points do not imply verified completed sorting.
