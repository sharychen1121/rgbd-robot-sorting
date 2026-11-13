"""Versioned scene-level synthetic detection/segmentation datasets.

Only this annotation path consumes privileged simulator segmentation. Both
tasks share the exact RGB images and scene-level split. Empty scenes are kept
as explicit negative examples instead of silently discarded.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import random
import shutil
import tempfile

import cv2
import numpy as np
import yaml

from .types import CLASS_IDS, CameraConfig, SceneSpec

DATASET_SCHEMA_VERSION = 2
SPLITS = ("train", "val", "test")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_rgb(path: Path, rgb: np.ndarray):
    # imwrite's native Windows path handling varies between OpenCV versions.
    # Encoding bytes and using Python's filesystem layer preserves Unicode.
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Could not encode image {path}")
    path.write_bytes(encoded.tobytes())


def labels_from_segmentation(segmentation: np.ndarray, truth: dict[int, str], *, min_area: float = 8.):
    """Build visible-instance polygon and box labels from explicit target IDs.

    A single polygon is exported per object. Standard non-overlapping scenes
    avoid disconnected visibility; any discarded fragments are counted in the
    annotation metadata, so that label limitations are inspectable.
    """
    if segmentation.ndim != 2:
        raise ValueError("Segmentation must be a 2-D body-ID buffer")
    height, width = segmentation.shape
    body_ids = segmentation.astype(np.int64) & 0xFFFFFF
    segments, boxes, audit = [], [], []
    for body_id, name in sorted(truth.items()):
        if name not in CLASS_IDS:
            raise ValueError(f"Unsupported label class {name}")
        mask = (body_ids == int(body_id)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            audit.append({"body_id": int(body_id), "class_name": name, "visible": False})
            continue
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(.35, .002*cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        normalized = polygon.astype(float) / np.array([width, height])
        normalized = np.clip(normalized, 0, 1)
        x,y,w,h = cv2.boundingRect(contour)
        class_id = CLASS_IDS[name]
        segments.append(str(class_id) + " " + " ".join(f"{v:.7f}" for v in normalized.ravel()))
        boxes.append(f"{class_id} {(x+w/2)/width:.7f} {(y+h/2)/height:.7f} {w/width:.7f} {h/height:.7f}")
        audit.append({"body_id": int(body_id), "class_name": name, "visible": True,
                      "polygon_points": len(polygon), "visible_pixels": int(mask.sum()),
                      "discarded_fragments": max(0, len(contours)-1)})
    return segments, boxes, audit


def _split_indices(count: int, seed: int):
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    train = max(1, int(math.floor(.70*count)))
    val = max(1, int(math.floor(.15*count)))
    train = min(train, count-2)
    return {i: ("train" if rank < train else "val" if rank < train+val else "test")
            for rank,i in enumerate(indices)}


def _validate_output(output: Path, overwrite: bool):
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Dataset directory is not empty: {output}. Use a new version directory or explicit overwrite.")
        manifest = output/"manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("owner") != "robosort":
            raise ValueError("Refusing to replace an unrecognized directory; choose a fresh output path")
    if output == Path(output.anchor) or output == Path.home():
        raise ValueError("A filesystem root or home directory cannot be a dataset output")


def generate_dataset(output: str | Path, count: int = 300, seed: int = 42, width: int = 640,
                     height: int = 480, color_randomization: bool = False,
                     negative_fraction: float = .15, overwrite: bool = False,
                     lighting_randomization: bool = False) -> Path:
    """Generate a new dataset atomically; explicit overwrite preserves a backup.

    Returns the manifest path. ``segment.yaml`` and ``detect.yaml`` live beside
    it; the manifest includes all sampled scenes and exact label hashes.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 5:
        raise ValueError("Use at least 5 scenes so train/val/test can all be present")
    if not 0 <= negative_fraction < .8:
        raise ValueError("Negative fraction must lie in [0, .8)")
    output = Path(output).expanduser().resolve()
    _validate_output(output, overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = CameraConfig(width=width, height=height)
    splits = _split_indices(count, seed)
    order = list(range(count)); random.Random(seed+173).shuffle(order)
    negatives = set(order[:round(count*negative_fraction)])
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        from .sim import SimScene, sample_scene
        records = []
        for task in ("segment", "detect"):
            for split in ("train", "val", "test"):
                (work/task/"images"/split).mkdir(parents=True)
                (work/task/"labels"/split).mkdir(parents=True)
        for index in range(count):
            scene_seed = seed*100_003 + index*97 + 11
            scene_id = f"scene_{index:06d}"
            lighting = random.Random(scene_seed+13).uniform(.55, 1.25) if lighting_randomization else 1.
            spec = sample_scene(scene_seed, scene_id=scene_id, split=splits[index],
                                random_colors=color_randomization, lighting=lighting,
                                camera=camera, min_objects=3, max_objects=3)
            if index in negatives:
                spec = replace(spec, objects=())
            with SimScene(spec, gui=False) as simulation:
                frame = simulation.capture(include_truth=True)
                if frame.segmentation is None:
                    raise RuntimeError("Annotation capture did not supply simulator segmentation")
                segments, boxes, annotation_audit = labels_from_segmentation(frame.segmentation, simulation.truth)
                rgb = frame.rgb
            split = splits[index]
            image_rel = Path("images")/split/f"{scene_id}.png"
            segment_image = work/"segment"/image_rel
            _write_rgb(segment_image, rgb)
            detect_image = work/"detect"/image_rel
            try:
                os.link(segment_image, detect_image)
            except OSError:
                shutil.copy2(segment_image, detect_image)
            segment_label = work/"segment"/"labels"/split/f"{scene_id}.txt"
            detect_label = work/"detect"/"labels"/split/f"{scene_id}.txt"
            segment_label.write_text("\n".join(segments)+("\n" if segments else ""), encoding="utf-8")
            detect_label.write_text("\n".join(boxes)+("\n" if boxes else ""), encoding="utf-8")
            records.append({"scene": spec.to_dict(), "negative": index in negatives,
                            "image": str(Path("segment")/image_rel).replace("\\", "/"),
                            "instances": len(segments), "audit": annotation_audit,
                            "image_sha256": sha256_file(segment_image),
                            "label_sha256": sha256_file(segment_label),
                            "segment_label_sha256": sha256_file(segment_label),
                            "detect_label_sha256": sha256_file(detect_label)})
            if (index+1) % 10 == 0 or index+1 == count:
                print(f"Generated {index+1}/{count} scenes", flush=True)
        for task in ("segment", "detect"):
            # Ultralytics resolves an omitted ``path`` against yaml_file.parent.
            # A relative ``path`` may instead use its global datasets_dir, and
            # an absolute one breaks as soon as this dataset is moved.
            config = {split: f"{task}/images/{split}" for split in SPLITS}
            config.update({
                      "names": {value: key for key,value in CLASS_IDS.items()}}
            )
            (work/f"{task}.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest = {"owner": "robosort", "schema_version": DATASET_SCHEMA_VERSION,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                    "scene_count": count, "negative_count": len(negatives),
                    "split_counts": {s:sum(v==s for v in splits.values()) for s in ("train","val","test")},
                    "classes": CLASS_IDS, "color_randomization": color_randomization,
                    "lighting_randomization": lighting_randomization,
                    "segment_yaml": "segment.yaml", "detect_yaml": "detect.yaml",
                    "annotation": "largest visible target contour from simulator body IDs; GT used only for labels",
                    "records": records}
        (work/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if output.exists():
            if any(output.iterdir()):
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                backup = output.with_name(output.name+f".backup-{stamp}")
                output.rename(backup)
                print(f"Previous dataset preserved at {backup}", flush=True)
            else:
                output.rmdir()
        work.rename(output)
        return output/"manifest.json"
    except Exception:
        # Only the private, freshly-created staging directory is removed.
        if work.exists():
            shutil.rmtree(work)
        raise


def validate_dataset(dataset: str | Path) -> dict:
    """Audit bytes, labels and scene-level split membership without mutations.

    Schema 1 manifests remain readable, but did not record detection-label
    hashes. A warning makes that weaker provenance explicit. Schema 2 records
    both label hashes, and all generated configurations are portable.
    """
    root = Path(dataset).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    errors, warnings, summary, seen = [], [], {}, set()
    schema = manifest.get("schema_version")
    if manifest.get("owner") != "robosort" or schema not in (1, DATASET_SCHEMA_VERSION):
        errors.append("Unsupported or unrecognized dataset manifest")
    if schema == 1:
        warnings.append("Schema 1 did not record detection-label hashes; regenerate for full byte-level provenance")
    records = manifest.get("records", [])
    if not isinstance(records, list) or not records:
        errors.append("Manifest must contain scene records")
        records = []
    expected = {split: {} for split in SPLITS}
    ids = set()
    negatives = 0
    for record in records:
        try:
            spec = SceneSpec.from_dict(record["scene"])
            if spec.scene_id in ids or spec.split not in SPLITS:
                raise ValueError("duplicate scene ID or unsupported dataset split")
            ids.add(spec.scene_id)
            if not isinstance(record.get("negative"), bool):
                raise ValueError("negative flag must be boolean")
            if record["negative"] != (len(spec.objects) == 0):
                raise ValueError("negative flag disagrees with scene objects")
            if isinstance(record.get("instances"), bool) or not isinstance(record.get("instances"), int) or not 0 <= record["instances"] <= len(spec.objects):
                raise ValueError("instance count disagrees with scene objects")
            image = f"segment/images/{spec.split}/{spec.scene_id}.png"
            if record.get("image") != image:
                raise ValueError("image path disagrees with scene ID/split")
            expected[spec.split][spec.scene_id] = (record, spec)
            negatives += record["negative"]
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"Invalid manifest scene record: {error}")
    actual = {task: {split: {} for split in SPLITS} for task in ("segment", "detect")}
    label_classes = {}
    for task in ("segment", "detect"):
        summary[task] = {}
        config_path = root/f"{task}.yaml"
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config.get("names") != {value: key for key, value in CLASS_IDS.items()}:
                errors.append(f"Class mapping differs in {config_path.name}")
            base = Path(config.get("path") or root)
            if not base.is_absolute():
                errors.append(f"Ambiguous relative dataset path in {config_path.name}; omit path to use YAML directory")
            for split in SPLITS:
                if (base/str(config.get(split, ""))).resolve() != root/task/"images"/split:
                    errors.append(f"YAML split path differs from dataset: {task}/{split}")
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as error:
            errors.append(f"Unreadable dataset YAML {config_path.name}: {error}")
        for split in SPLITS:
            image_directory = root/task/"images"/split
            label_directory = root/task/"labels"/split
            if not image_directory.is_dir() or not label_directory.is_dir():
                errors.append(f"Missing image/label directory: {task}/{split}")
            images = sorted(image_directory.glob("*.png"))
            count = 0
            for image in images:
                key = (task,image.stem)
                if key in seen:
                    errors.append(f"Repeated scene ID across splits: {key}")
                seen.add(key)
                actual[task][split][image.stem] = image
                decoded = cv2.imdecode(np.frombuffer(image.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
                if decoded is None:
                    errors.append(f"Unreadable image: {image}")
                record_spec = expected[split].get(image.stem)
                if decoded is not None and record_spec is not None:
                    camera = record_spec[1].camera
                    if decoded.shape != (camera.height, camera.width, 3):
                        errors.append(f"Image resolution disagrees with recorded camera: {image}")
                label = label_directory/(image.stem+".txt")
                if not label.exists():
                    errors.append(f"Missing label: {label}"); continue
                classes = []
                for line_no,line in enumerate(label.read_text(encoding="utf-8").splitlines(),1):
                    try:
                        values = [float(v) for v in line.split()]
                        class_id = values[0]
                        coords = values[1:]
                        if class_id not in CLASS_IDS.values() or class_id != int(class_id):
                            raise ValueError("class ID")
                        if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
                            raise ValueError("coordinate bounds")
                        if task == "segment" and (len(coords) < 6 or len(coords)%2):
                            raise ValueError("polygon requires at least 3 points")
                        if task == "detect" and (len(coords) != 4 or min(coords[2:]) <= 0):
                            raise ValueError("positive-width bounding box")
                        if task == "detect" and (coords[0]-coords[2]/2 < -1e-6 or coords[0]+coords[2]/2 > 1+1e-6 or coords[1]-coords[3]/2 < -1e-6 or coords[1]+coords[3]/2 > 1+1e-6):
                            raise ValueError("bounding box extends beyond image")
                        if task == "segment":
                            polygon = np.asarray(coords, np.float32).reshape(-1, 2)
                            if len(np.unique(polygon, axis=0)) < 3 or cv2.contourArea(polygon) <= 0:
                                raise ValueError("degenerate polygon")
                        classes.append(int(class_id))
                        count += 1
                    except (ValueError,IndexError,OverflowError):
                        errors.append(f"Malformed label {label}:{line_no}")
                label_classes[(task, split, image.stem)] = sorted(classes)
                if record_spec is not None:
                    record, spec = record_spec
                    if len(classes) != record["instances"]:
                        errors.append(f"Label instance count differs from manifest: {label}")
                    if record["negative"] and classes:
                        errors.append(f"Negative scene contains object labels: {label}")
                    available = [CLASS_IDS[obj.class_name] for obj in spec.objects]
                    if any(classes.count(c) > available.count(c) for c in set(classes)):
                        errors.append(f"Label classes disagree with recorded scene: {label}")
                    expected_image_hash = record.get("image_sha256")
                    expected_label_hash = record.get(f"{task}_label_sha256")
                    if task == "segment" and expected_label_hash is None:
                        expected_label_hash = record.get("label_sha256")
                    for file, expected_hash, required in ((image, expected_image_hash, True), (label, expected_label_hash, task == "segment" or schema == DATASET_SCHEMA_VERSION)):
                        if expected_hash is None and not required:
                            continue
                        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash) or sha256_file(file) != expected_hash:
                            errors.append(f"Missing or mismatched SHA256: {file}")
            for orphan in label_directory.glob("*.txt"):
                if not (image_directory/(orphan.stem+".png")).is_file():
                    errors.append(f"Orphan label without image: {orphan}")
            if set(actual[task][split]) != set(expected[split]):
                errors.append(f"Actual image scene IDs differ from manifest: {task}/{split}")
            summary[task][split] = {"images":len(images),"instances":count}
    for split in SPLITS:
        for scene_id in expected[split]:
            if label_classes.get(("segment", split, scene_id)) != label_classes.get(("detect", split, scene_id)):
                errors.append(f"Detection and segmentation labels disagree: {split}/{scene_id}")
    if len(records) != manifest.get("scene_count") or sum(summary["segment"][s]["images"] for s in SPLITS) != manifest.get("scene_count"):
        errors.append("Manifest scene count differs from image count")
    if manifest.get("split_counts") != {s: len(expected[s]) for s in SPLITS}:
        errors.append("Manifest split counts disagree with recorded scene splits")
    if manifest.get("negative_count") != negatives:
        errors.append("Manifest negative count disagrees with scene records")
    return {"valid":not errors,"errors":errors,"warnings":warnings,"splits":summary,
            "negative_count":manifest.get("negative_count"),"schema_version":schema}
