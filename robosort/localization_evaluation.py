"""Static, class-agnostic detection matching and conditional RGB-D errors.

Simulator labels are used by this evaluator only. The predictor/localizer get
the same truth-free RGB-D frame and predictions for both region ablations.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import io
import math
from pathlib import Path

import numpy as np

from .evaluation import _atomic_text, _dumps, catalogue_fingerprint
from .perception import localize
from .sim import SimScene


def _box_iou(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.
    intersection = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0., a[2:]-a[:2]).prod()+np.maximum(0., b[2:]-b[:2]).prod()-intersection
    return float(intersection/union) if union > 0 else 0.


def _global_matches(ious, threshold):
    """Exact one-to-one DP: maximize count, then total IoU, without class labels.

    Rows are true visible instances and columns are predictions. This is small
    because this simulator has at most three targets. Duplicate detections can
    never match the same target and remain false positives.
    """
    ious = np.asarray(ious, float)
    if ious.ndim != 2 or ious.shape[0] > 15:
        raise ValueError("Matching requires a matrix and at most 15 targets")
    states = {0: (0., ())}
    for prediction in range(ious.shape[1]):
        updated = dict(states)
        for mask, (score, pairs) in states.items():
            for target in range(ious.shape[0]):
                value = ious[target, prediction]
                if mask & (1 << target) or not np.isfinite(value) or value < threshold:
                    continue
                next_mask = mask | (1 << target)
                if next_mask not in updated or score+value > updated[next_mask][0]:
                    updated[next_mask] = (score+float(value), pairs+((target, prediction, float(value)),))
        states = updated
    return list(max(states.items(), key=lambda item: (item[0].bit_count(), item[1][0]))[1][1])


def _statistics(values):
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None}


def evaluate_localization(scenes, predictor, representations=("box", "mask"), iou_threshold=.25):
    """Evaluate frozen scenes with one inference per scene; no arm movement.

    All true targets remain in coverage denominators, including invisible and
    missed ones. Errors condition on class-agnostically matched AND valid
    localizations, even if the predicted class is wrong. Read coverage beside
    error: an accurate subset with many misses is not a complete solution.
    """
    scenes, representations = list(scenes), tuple(representations)
    if not scenes or len({s.scene_id for s in scenes}) != len(scenes):
        raise ValueError("Provide nonempty scenes with unique IDs")
    if not representations or len(set(representations)) != len(representations) or any(r not in {"box", "mask"} for r in representations):
        raise ValueError("Representations must be unique box/mask entries")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in (0, 1]")
    details = []
    for spec in scenes:
        with SimScene(spec, gui=False) as scene:
            labelled = scene.capture(include_truth=True)
            if labelled.segmentation is None:
                raise ValueError("Scoring requires an explicit simulator label capture")
            truth = scene.ground_truth_poses()
            ids = labelled.segmentation.astype(np.int64) & 0xFFFFFF
            targets = []
            for item in truth:
                rows, cols = np.nonzero(ids == item["body_id"])
                targets.append(dict(item, visible=bool(len(rows)),
                    bbox=[int(cols.min()), int(rows.min()), int(cols.max()+1), int(rows.max()+1)] if len(rows) else None))
            # Never expose the privileged segmentation to inference/localization.
            frame = replace(labelled, segmentation=None)
            predictions = list(predictor.predict(frame))
            visible_indices = [i for i, target in enumerate(targets) if target["visible"]]
            matrix = np.array([[_box_iou(targets[i]["bbox"], p.bbox) for p in predictions] for i in visible_indices], float).reshape(len(visible_indices), len(predictions))
            pairs = [(visible_indices[i], j, iou) for i, j, iou in _global_matches(matrix, iou_threshold)]
            matched_targets = {i for i, _, _ in pairs}; matched_predictions = {j for _, j, _ in pairs}
            matches = []
            for i, j, iou in pairs:
                target, prediction = targets[i], predictions[j]
                locations = {}
                for representation in representations:
                    location = localize(prediction, frame, representation=representation)
                    valid = location.valid and np.isfinite(location.point_world).all()
                    error = np.asarray(location.point_world)-np.asarray(target["top_position"]) if valid else None
                    locations[representation] = {"valid": bool(valid), "reason": location.reason,
                        "point_world": location.point_world, "depth_pixels": location.depth_pixels,
                        "xy_error_m": float(np.linalg.norm(error[:2])) if valid else None,
                        "z_error_m": float(abs(error[2])) if valid else None,
                        "error_3d_m": float(np.linalg.norm(error)) if valid else None}
                matches.append({"body_id": target["body_id"], "true_class": target["class_name"],
                    "true_top_position": target["top_position"], "true_bbox": target["bbox"],
                    "prediction_index": j, "predicted_class": prediction.class_name,
                    "predicted_bbox": prediction.bbox, "confidence": prediction.confidence,
                    "iou": iou, "classification_correct": prediction.class_name == target["class_name"],
                    "localizations": locations})
            details.append({"scene_id": spec.scene_id, "total_targets": len(targets),
                "visible_targets": len(visible_indices), "prediction_count": len(predictions), "matches": matches,
                "missed": [dict(targets[i], reason="unmatched" if targets[i]["visible"] else "not_visible") for i in range(len(targets)) if i not in matched_targets],
                "false_positives": [{"prediction_index": j, "predicted_class": p.class_name, "bbox": p.bbox,
                    "confidence": p.confidence} for j, p in enumerate(predictions) if j not in matched_predictions]})
    total = sum(s["total_targets"] for s in details)
    matched = sum(len(s["matches"]) for s in details)
    correct = sum(m["classification_correct"] for s in details for m in s["matches"])
    summary = {"total_targets": total, "visible_targets": sum(s["visible_targets"] for s in details),
        "prediction_count": sum(s["prediction_count"] for s in details), "matched_targets": matched,
        "missed_targets": total-matched, "false_positive_predictions": sum(len(s["false_positives"]) for s in details),
        "classification_correct_matches": correct, "classification_accuracy_matched": correct/matched if matched else None,
        "correct_class_coverage_all_targets": correct/total if total else None, "representations": {}}
    for representation in representations:
        locations = [m["localizations"][representation] for s in details for m in s["matches"]]
        valid = [location for location in locations if location["valid"]]
        summary["representations"][representation] = {"valid_matched_localizations": len(valid),
            "valid_localization_coverage_all_targets": len(valid)/total if total else None,
            "valid_fraction_of_matched_targets": len(valid)/matched if matched else None,
            "invalid_matched_localizations": matched-len(valid),
            **{field: _statistics([location[field] for location in valid]) for field in ("xy_error_m", "z_error_m", "error_3d_m")}}
    return {"schema_version": 1, "evaluation": "static_localization", "iou_threshold": iou_threshold,
        "matching": "exact class-agnostic global one-to-one; maximum cardinality then maximum summed bbox IoU",
        "error_denominator": "matched AND valid predictions; independent of classification correctness",
        "coverage_denominator": "all true scene targets, including invisible and missed targets",
        "scene_catalogue_sha256": catalogue_fingerprint(scenes), "scene_specs": [s.to_dict() for s in scenes],
        "summary": summary, "scenes": details}


def save_localization_results(report, output_dir, metadata=None, overwrite=False):
    """Save source JSON and per-target/prediction CSV; refuse implicit overwrite."""
    output = Path(output_dir)
    targets = (output/"localization.json", output/"localization.csv")
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Localization results already exist; choose a fresh output directory")
    document = dict(report, metadata=dict(metadata or {}))
    rows = []
    for scene in report["scenes"]:
        for representation in report["summary"]["representations"]:
            for match in scene["matches"]:
                location = match["localizations"][representation]
                rows.append(dict(scene_id=scene["scene_id"], representation=representation, status="matched",
                    body_id=match["body_id"], true_class=match["true_class"], prediction_index=match["prediction_index"],
                    predicted_class=match["predicted_class"], classification_correct=match["classification_correct"],
                    iou=match["iou"], **location))
            for missed in scene["missed"]:
                rows.append(dict(scene_id=scene["scene_id"], representation=representation, status="missed",
                    body_id=missed["body_id"], true_class=missed["class_name"], reason=missed["reason"]))
            for false_positive in scene["false_positives"]:
                rows.append(dict(scene_id=scene["scene_id"], representation=representation, status="false_positive", **false_positive))
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["scene_id", "representation", "status"]
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    _atomic_text(targets[0], _dumps(document, pretty=True)+"\n")
    _atomic_text(targets[1], "\ufeff"+stream.getvalue())
    return targets[0]
