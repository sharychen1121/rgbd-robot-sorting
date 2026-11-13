"""Auditable experiment storage and metrics for the sorting task.

Every stored row belongs to one named run and one (method, scene) pair. A run
bundle is immutable unless overwrite is explicitly requested. Counts are
derived from scored per-object outcomes, never from rejection messages.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable, Mapping
import uuid

from robosort.types import CLASS_IDS, DESTINATIONS, RunResult, SceneSpec

SCHEMA_VERSION = 1
TIME_FIELDS = ("initialization_s", "perception_s", "localization_s", "motion_s", "total_s")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, default=_json_default,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":"))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def new_run_id(prefix: str = "experiment") -> str:
    """Create a unique ID; save the returned ID in the experiment metadata."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def catalogue_fingerprint(scenes: Iterable[SceneSpec]) -> str:
    """Fingerprint ordered scene content, including physical and camera settings."""
    documents = [scene.to_dict() for scene in scenes]
    return sha256(_dumps(documents).encode("utf-8")).hexdigest()


def save_catalogue(scenes: Iterable[SceneSpec], path: str | Path,
                   metadata: Mapping[str, Any] | None = None, *, overwrite: bool = False) -> Path:
    scenes = list(scenes)
    ids = [scene.scene_id for scene in scenes]
    if not scenes or len(ids) != len(set(ids)):
        raise ValueError("A scene catalogue must be nonempty with unique scene IDs")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Scene catalogue exists; choose a new path: {path}")
    document = {"schema_version": SCHEMA_VERSION,
                "scene_catalogue_sha256": catalogue_fingerprint(scenes),
                "metadata": dict(metadata or {}), "scenes": [s.to_dict() for s in scenes]}
    _atomic_text(path, _dumps(document, pretty=True) + "\n")
    return path


def load_catalogue(path: str | Path) -> list[SceneSpec]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported scene catalogue schema")
    scenes = [SceneSpec.from_dict(s) for s in document["scenes"]]
    ids = [s.scene_id for s in scenes]
    if not scenes or len(set(ids)) != len(ids):
        raise ValueError("Empty catalogue or duplicate scene IDs")
    if catalogue_fingerprint(scenes) != document.get("scene_catalogue_sha256"):
        raise ValueError("Scene catalogue fingerprint mismatch; content was changed")
    return scenes


def _nonnegative_time(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a measured duration, not a boolean")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a nonnegative duration or null")
    return number


def _normalize(result: RunResult | Mapping[str, Any], run_id: str | None = None) -> dict[str, Any]:
    raw = result.to_dict() if isinstance(result, RunResult) else dict(result)
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in ("scene_id", "method")):
        raise ValueError("Every result needs scene_id and method")
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Unsupported result schema")
    if raw.get("status") not in {"completed", "stopped", "failed", "skipped"}:
        raise ValueError("Unsupported run status")
    count = raw.get("total_objects")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("total_objects must be a nonnegative integer")
    record_id = run_id or raw.get("run_id")
    if record_id is not None:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if raw.get("run_id") not in (None, record_id):
            raise ValueError("Result belongs to a different run_id")
        raw["run_id"] = record_id
    raw["schema_version"] = SCHEMA_VERSION
    for field in TIME_FIELDS:
        raw[field] = _nonnegative_time(raw.get(field), field)
    raw.setdefault("outcomes", [])
    raw.setdefault("rejections", [])
    raw.setdefault("reason", "unspecified")
    raw.setdefault("observations", 0)
    raw.setdefault("prediction_count", 0)
    if any(isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 0 for field in ("observations", "prediction_count")):
        raise ValueError("Observation and prediction counts must be nonnegative integers")
    if not isinstance(raw["outcomes"], list) or not isinstance(raw["rejections"], list):
        raise ValueError("Outcomes and rejections must be lists")
    if raw["status"] == "skipped" and raw["outcomes"]:
        raise ValueError("A skipped trial cannot contain executed object outcomes")
    bodies = set()
    true_classes = {}
    for outcome in raw["outcomes"]:
        if not isinstance(outcome, dict):
            raise ValueError("Object outcomes must be structured records")
        body_id = outcome.get("body_id")
        if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
            raise ValueError("Object body IDs must be nonnegative integers")
        if outcome.get("status") not in {"placed", "mis_sorted", "failed"}:
            raise ValueError("Unsupported object outcome status")
        if outcome.get("true_class") not in CLASS_IDS or outcome.get("predicted_class") not in CLASS_IDS:
            raise ValueError("Object outcomes require supported true and predicted classes")
        if outcome.get("status") in {"placed", "mis_sorted"} and outcome.get("destination") not in set(DESTINATIONS.values()):
            raise ValueError("A physically placed object needs a destination")
        if body_id in true_classes and true_classes[body_id] != outcome["true_class"]:
            raise ValueError("One body's true class cannot change across attempts")
        true_classes[body_id] = outcome["true_class"]
        bodies.add(body_id)
    if len(bodies) > count:
        raise ValueError("More distinct attempted objects than scene targets")
    for rejection in raw["rejections"]:
        if not isinstance(rejection, dict):
            raise ValueError("Rejections must be structured stage/reason records")
    if raw["status"] == "completed" and count > 0:
        if _counts(raw)["correct_placements"] != count:
            raise ValueError("A completed result must have every object correctly placed")
    return raw


def _counts(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Use each body's latest scored attempt; repeated attempts do not add objects."""
    final = {}
    lifted = set()
    for outcome in raw["outcomes"]:
        body = outcome["body_id"]
        final[body] = outcome
        if outcome.get("lifted"):
            lifted.add(body)
    correct, wrong = set(), set()
    for body, outcome in final.items():
        physically_placed = outcome["status"] in {"placed", "mis_sorted"}
        matches = outcome["predicted_class"] == outcome["true_class"] and outcome.get("destination") == DESTINATIONS[outcome["true_class"]]
        if physically_placed:
            (correct if outcome["status"] == "placed" and matches else wrong).add(body)
    total = raw["total_objects"]
    return {"correct_placements": len(correct), "mis_sorted_objects": len(wrong),
            "lifted_objects": len(lifted), "lift_only_objects": len(lifted - correct - wrong),
            "attempted_objects": len(final), "grasp_attempts": len(raw["outcomes"]),
            "any_correct_placement": bool(correct),
            "full_success": total > 0 and len(correct) == total and not wrong and raw["status"] == "completed"}


def _records(results: Iterable[RunResult | Mapping[str, Any]], run_id: str | None = None) -> list[dict[str, Any]]:
    records = [_normalize(r, run_id) for r in results]
    if not records:
        raise ValueError("No experiment results")
    keys = [(r["method"], r["scene_id"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (method, scene_id); store repeats as distinct scene IDs or run IDs")
    ids = {r.get("run_id") for r in records}
    if len(ids) != 1:
        raise ValueError("Cannot mix different or missing run IDs")
    target_counts = {}
    for row in records:
        previous = target_counts.setdefault(row["scene_id"], row["total_objects"])
        if previous != row["total_objects"]:
            raise ValueError("Methods disagree on the target count for the same scene ID")
    return records


def summarize(results: Iterable[RunResult | Mapping[str, Any]], *, require_paired: bool = True) -> dict[str, Any]:
    records = _records(results)
    methods = sorted({r["method"] for r in records})
    scene_sets = {m: {r["scene_id"] for r in records if r["method"] == m} for m in methods}
    paired = all(s == scene_sets[methods[0]] for s in scene_sets.values())
    if require_paired and not paired:
        raise ValueError("Methods were evaluated on different scene sets; a paired comparison is required")
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
        "run_id": records[0].get("run_id"), "paired_scene_sets": paired,
        "scene_ids_by_method": {m: sorted(ids) for m, ids in scene_sets.items()}, "methods": {}}
    for method in methods:
        subset = [r for r in records if r["method"] == method]
        counts = [_counts(r) for r in subset]
        nonempty = [i for i, r in enumerate(subset) if r["total_objects"] > 0]
        evaluated = [i for i in nonempty if subset[i]["status"] != "skipped"]
        completed = sum(counts[i]["full_success"] for i in evaluated)
        object_count = sum(subset[i]["total_objects"] for i in nonempty)
        placed_count = sum(counts[i]["correct_placements"] for i in evaluated)
        mis_sorted = sum(c["mis_sorted_objects"] for c in counts)
        stages: Counter[str] = Counter()
        rejection_reasons: Counter[str] = Counter()
        terminal: Counter[str] = Counter()
        for row, c in zip(subset, counts):
            if row["total_objects"] > 0 and not c["full_success"] and row["status"] != "skipped":
                terminal[str(row.get("reason") or "unspecified")] += 1
            for rejection in row["rejections"]:
                stages[str(rejection.get("stage") or "unspecified")] += 1
                rejection_reasons[str(rejection.get("reason") or "unspecified")] += 1
        means, medians, measured = {}, {}, {}
        for field in TIME_FIELDS:
            values = [r[field] for r in subset if r[field] is not None and r["status"] != "skipped"]
            means[field] = statistics.mean(values) if values else None
            medians[field] = statistics.median(values) if values else None
            measured[field] = len(values)
        summary["methods"][method] = {
            "trials": len(subset), "planned_nonempty_trials": len(nonempty),
            "evaluated_trials": len(evaluated),
            "evaluation_coverage": len(evaluated) / len(nonempty) if nonempty else None,
            "skipped_trials": sum(r["status"] == "skipped" for r in subset),
            "empty_scene_trials": sum(r["total_objects"] == 0 for r in subset),
            "completed_trials": completed,
            "full_success_rate": completed / len(nonempty) if nonempty else None,
            "at_least_one_correct_placement_trials": sum(c["any_correct_placement"] for c in counts),
            "grasp_attempted_trials": sum(c["grasp_attempts"] > 0 for c in counts),
            "grasp_attempts": sum(c["grasp_attempts"] for c in counts),
            "correct_placements": placed_count, "total_objects": object_count,
            "object_correct_placement_rate": placed_count / object_count if object_count else None,
            "mis_sorted_objects": mis_sorted,
            "lifted_objects": sum(c["lifted_objects"] for c in counts),
            "lift_only_objects": sum(c["lift_only_objects"] for c in counts),
            "terminal_failures": dict(sorted(terminal.items())),
            "rejection_events_by_stage": dict(sorted(stages.items())),
            "rejection_events_by_reason": dict(sorted(rejection_reasons.items())),
            "mean_time_s": means, "median_time_s": medians, "timing_measured_trials": measured,
        }
    return summary


def flatten_result(record: Mapping[str, Any]) -> dict[str, Any]:
    counts = _counts(record)
    row = {key: record.get(key) for key in ("run_id", "scene_id", "method", "status", "reason", "total_objects", "observations", "prediction_count", *TIME_FIELDS)}
    row.update(counts)
    row["rejection_events"] = len(record["rejections"])
    row["outcomes_json"] = _dumps(record["outcomes"])
    row["rejections_json"] = _dumps(record["rejections"])
    return row


def save_results(results: Iterable[RunResult | Mapping[str, Any]], output_dir: str | Path,
                 metadata: Mapping[str, Any], *, overwrite: bool = False,
                 require_paired: bool = True) -> Path:
    """Save JSONL truth, readable CSV, summary and metadata without silent appends.

    Supply a run_id in metadata to replay an explicitly named experiment. If
    omitted, one unique ID is generated for this bundle and written everywhere.
    """
    metadata = dict(metadata)
    run_id = metadata.get("run_id") or new_run_id()
    records = _records(results, run_id)
    summary = summarize(records, require_paired=require_paired)
    output_dir = Path(output_dir)
    targets = [output_dir / name for name in ("results.jsonl", "trial_results.csv", "summary.json", "metadata.json")]
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Result bundle already exists; use a new directory or explicit overwrite: {existing[0]}")
    rows = [flatten_result(r) for r in records]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    metadata.update({"schema_version": SCHEMA_VERSION, "run_id": run_id,
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "record_count": len(records), "methods": sorted(summary["methods"]),
                     "scene_ids_by_method": summary["scene_ids_by_method"],
                     "results_sha256": sha256(("\n".join(_dumps(r) for r in records) + "\n").encode("utf-8")).hexdigest()})
    _atomic_text(targets[0], "\n".join(_dumps(r) for r in records) + "\n")
    _atomic_text(targets[1], "\ufeff" + stream.getvalue())
    _atomic_text(targets[2], _dumps(summary, pretty=True) + "\n")
    # Metadata is written last and fingerprints the complete source JSONL.
    _atomic_text(targets[3], _dumps(metadata, pretty=True) + "\n")
    return targets[0]


def load_results(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    text = path.read_text(encoding="utf-8-sig")
    records = _records(json.loads(line) for line in text.splitlines() if line.strip())
    if any(not r.get("run_id") for r in records):
        raise ValueError("Stored experiment records must have a run_id")
    metadata_path = path.with_name("metadata.json")
    if not metadata_path.exists():
        raise ValueError("Incomplete result bundle: metadata.json is missing")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported result metadata schema")
        if metadata.get("run_id") != records[0]["run_id"]:
            raise ValueError("Metadata run_id does not match results")
        expected = metadata.get("results_sha256")
        if not expected or sha256(text.encode("utf-8")).hexdigest() != expected:
            raise ValueError("Results fingerprint mismatch; files changed or a write was incomplete")
        scene_sets = {method: sorted(r["scene_id"] for r in records if r["method"] == method) for method in sorted({r["method"] for r in records})}
        if metadata.get("record_count") != len(records) or metadata.get("methods") != sorted(scene_sets) or metadata.get("scene_ids_by_method") != scene_sets:
            raise ValueError("Result metadata does not describe the stored method/scene records")
    return records


def plot_results(results: str | Path | Iterable[RunResult | Mapping[str, Any]], output_dir: str | Path) -> list[Path]:
    """Generate honest task-rate, rejection-event and measured-time plots."""
    records = load_results(results) if isinstance(results, (str, Path)) else list(results)
    summary = summarize(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
    methods = list(summary["methods"])
    stats = [summary["methods"][m] for m in methods]
    x = np.arange(len(methods))
    paths = []
    fig, ax = plt.subplots(figsize=(max(7, len(methods)*2), 4.8))
    for offset, key, label, color in [(-.18, "full_success_rate", "Complete sorting", "#376B9D"), (.18, "object_correct_placement_rate", "Correct object placement", "#64A98A")]:
        values = [100*s[key] if s[key] is not None else np.nan for s in stats]
        bars = ax.bar(x+offset, values, .34, label=label, color=color)
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(bar.get_x()+bar.get_width()/2, value+1.5, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, methods); ax.set_ylim(0, 112); ax.set_ylabel("Success (%)")
    ax.set_title("Sorting outcomes — paired scenes")
    ax.legend(); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); path=folder/"success_rates.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    stages = sorted({stage for s in stats for stage in s["rejection_events_by_stage"]})
    fig, ax = plt.subplots(figsize=(max(7, len(methods)*2), 4.8))
    bottom=np.zeros(len(methods))
    if stages:
        for stage in stages:
            values=np.array([s["rejection_events_by_stage"].get(stage,0) for s in stats])
            ax.bar(x,values,bottom=bottom,label=stage); bottom+=values
        ax.legend(title="Stage", bbox_to_anchor=(1.02,1),loc="upper left")
    else:
        ax.text(.5,.5,"No candidate rejection events recorded",transform=ax.transAxes,ha="center")
    ax.set_xticks(x,methods); ax.set_ylabel("Rejection events (not unique objects)")
    ax.set_title("Candidate rejection records"); fig.tight_layout()
    path=folder/"candidate_failures.png"; fig.savefig(path,dpi=160); plt.close(fig); paths.append(path)
    fig,ax=plt.subplots(figsize=(max(8,len(methods)*2.3),5.2))
    width=.15
    for index,field in enumerate(TIME_FIELDS):
        values=[s["mean_time_s"][field] if s["mean_time_s"][field] is not None else np.nan for s in stats]
        ax.bar(x+(index-2)*width,values,width,label=field.removesuffix("_s").replace("_"," "))
    if all(all(s["mean_time_s"][field] is None for field in TIME_FIELDS) for s in stats):
        ax.text(.5,.5,"Timing was not measured",transform=ax.transAxes,ha="center")
    ax.set_xticks(x,methods); ax.set_ylabel("Measured mean elapsed time (s)")
    ax.set_title("Elapsed time — early failures can shorten task duration")
    ax.legend(ncol=3,loc="upper left"); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); path=folder/"timing.png"; fig.savefig(path,dpi=160); plt.close(fig); paths.append(path)
    return paths
