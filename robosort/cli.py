"""Reproducible command line entry points; no implicit inference downloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

METHODS = ("oracle", "hsv", "yolo_box", "yolo_mask", "hybrid", "yolov8_mask")


def _json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _environment():
    result = {"python": platform.python_version(), "platform": platform.system(), "packages": {}}
    for name in ("numpy", "opencv-python", "pybullet", "PyYAML", "matplotlib", "ultralytics", "torch", "torchvision"):
        try:
            result["packages"][name] = version(name)
        except PackageNotFoundError:
            result["packages"][name] = None
    try:
        code_dir = Path(__file__).resolve().parent.parent
        if not (code_dir/".git").exists():
            raise FileNotFoundError("Installed source has no Git metadata")
        result["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir, stderr=subprocess.DEVNULL, text=True).strip()
        result["code_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=code_dir, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result["code_commit"] = None
        result["code_dirty"] = None
    return result


def _fresh_output(path):
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output must be a directory, but a file already exists: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output is not empty: {path}. Choose a new version/run directory.")
    return path


def _catalogue(args):
    from .sim import demo_scene, sample_scene
    from .types import CameraConfig
    if getattr(args, "count", 1) <= 0:
        raise ValueError("Scene count must be positive")
    camera = CameraConfig(width=args.width, height=args.height)
    if getattr(args, "catalogue", None):
        from .evaluation import load_catalogue
        return load_catalogue(args.catalogue)
    if getattr(args, "fixed", False):
        return [demo_scene(args.seed, camera)]
    return [sample_scene(args.seed + i, scene_id=f"{args.regime}-{args.seed+i}",
                         regime=args.regime, camera=camera) for i in range(getattr(args, "count", 1))]


def _build_predictors(methods, args, first_scene):
    from .dataset import sha256_file
    from .perception import HSVPredictor, HybridPredictor, UltralyticsPredictor
    from .sim import SimScene
    predictors, metadata, loaded = {}, {}, {}
    for method in methods:
        if method == "oracle":
            predictors[method] = None
            metadata[method] = {"diagnostic_truth_input": True, "learned": False}
            continue
        if method == "hsv":
            predictors[method] = HSVPredictor()
            metadata[method] = {"fixed_class_colors": True, "learned": False}
            continue
        weights = args.yolov8_weights if method == "yolov8_mask" else args.weights
        if not weights or not Path(weights).is_file():
            raise FileNotFoundError(f"{method} requires a local trained segmentation checkpoint; supply --weights or --yolov8-weights")
        key = str(Path(weights).resolve())
        if key not in loaded:
            predictor = UltralyticsPredictor(weights, conf=args.conf, device=args.device)
            start = time.perf_counter()
            predictor.load()
            model_load_s = time.perf_counter()-start
            if predictor.model.task != "segment":
                raise ValueError("The mask/box ablation requires a segmentation checkpoint trained on cube/cylinder/box")
            names = set(predictor.model.names.values())
            if names != {"cube", "cylinder", "box"}:
                raise ValueError(f"Checkpoint has incompatible classes: {sorted(names)}")
            with SimScene(first_scene) as sim:
                frame = sim.capture(include_truth=False)
                start = time.perf_counter()
                predictor.predict(frame)
                warmup_s = time.perf_counter()-start
            loaded[key] = (predictor, {"checkpoint_name": Path(weights).name,
                "checkpoint_sha256": sha256_file(weights), "model_load_s": model_load_s,
                "warmup_s": warmup_s, "device": predictor.resolved_device or args.device,
                "device_requested": args.device, "confidence": args.conf, "learned": True,
                "task": str(predictor.model.task)})
        predictor, info = loaded[key]
        predictors[method] = HybridPredictor(predictor, low=args.conf, high=args.hybrid_high) if method == "hybrid" else predictor
        metadata[method] = dict(info, shared_predictor=True, representation="box" if method == "yolo_box" else "mask")
    return predictors, metadata


def _experiment(args, demo=False):
    from .evaluation import save_catalogue, catalogue_fingerprint, save_results, plot_results, new_run_id, flatten_result
    from .execution import run_sorting
    from .sim import SimScene
    methods = [args.method] if demo else args.methods
    if len(set(methods)) != len(methods):
        raise ValueError("Each method must occur once")
    if args.max_observations <= 0 or args.max_attempts <= 0:
        raise ValueError("Observation and attempt budgets must be positive")
    if not math.isfinite(args.conf) or not 0 <= args.conf <= 1:
        raise ValueError("Confidence threshold must lie in [0, 1]")
    if "hybrid" in methods and (not math.isfinite(args.hybrid_high) or not args.conf < args.hybrid_high <= 1):
        raise ValueError("Hybrid high confidence must be greater than --conf and at most 1")
    output = _fresh_output(args.output)
    scenes = _catalogue(args)
    predictors, model_info = _build_predictors(methods, args, scenes[0])
    output.mkdir(parents=True, exist_ok=True)
    save_catalogue(scenes, output/"scenes.json")
    metadata = {"run_id": new_run_id("demo" if demo else "paired"), "environment": _environment(),
        "models": model_info, "scene_catalogue_sha256": catalogue_fingerprint(scenes),
        "budgets": {"max_observations": args.max_observations, "max_attempts": args.max_attempts},
        "timing_scope": "Task excludes model load/warmup and scene load. initialization_s inside run_sorting measures controller setup, not model initialization. RGB-D capture contributes only to total_s. Perception/localization are cumulative measured observations; model cold load and warmup are in metadata.models.",
        "oracle_note": "Oracle reads true poses for diagnosis only; learned/hsv inputs have no simulator segmentation.",
        "created_utc": datetime.now(timezone.utc).isoformat()}
    results = []
    # The journal survives interruption; only a completed metadata-fingerprinted
    # bundle is accepted by summarize. Do not silently resume an interrupted run.
    with (output/"progress.jsonl").open("w", encoding="utf-8") as journal:
        for spec in scenes:
            for method in methods:
                print(f"Running {spec.scene_id}: {method}", flush=True)
                with SimScene(spec, gui=args.gui) as sim:
                    result = run_sorting(sim, predictors[method], method=method,
                        representation="box" if method == "yolo_box" else "mask",
                        max_observations=args.max_observations, max_attempts=args.max_attempts)
                    if demo:
                        from .dataset import _write_rgb
                        _write_rgb(output/"final_scene.png", sim.capture().rgb)
                results.append(result)
                journal.write(json.dumps(result.to_dict(), allow_nan=False)+"\n"); journal.flush()
                counts = flatten_result(result.to_dict())
                print(f"  {result.status}: {result.reason}; correctly_placed={counts['correct_placements']}/{result.total_objects}; mis_sorted={counts['mis_sorted_objects']}", flush=True)
    path = save_results(results, output, metadata)
    if args.plots:
        plot_results(path, output/"plots")
    print(f"Saved complete result bundle: {path}")


def parser():
    ap = argparse.ArgumentParser(description="Predicted instance masks + RGB-D localization + guarded PyBullet sorting")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only dependency and asset checks")
    def scene_args(p):
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
        p.add_argument("--regime", choices=("standard", "stress", "color", "lighting", "appearance"), default="standard")
        p.add_argument("--fixed", action="store_true", help="Use the fixed diagnostic demonstration")
    p = sub.add_parser("scenes", help="Freeze a replayable scene catalogue")
    scene_args(p); p.add_argument("--count", type=int, default=30); p.add_argument("--output", required=True)
    p = sub.add_parser("dataset", help="Generate synchronized scene-level segmentation/detection labels")
    p.add_argument("--output", required=True); p.add_argument("--count", type=int, default=600)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=480)
    p.add_argument("--random-colors", action="store_true"); p.add_argument("--random-lighting", action="store_true")
    p.add_argument("--negative-fraction", type=float, default=.15)
    p = sub.add_parser("check-dataset"); p.add_argument("path")
    p = sub.add_parser("train", help="Train a segmentation model; pretrained downloads require an explicit flag")
    p.add_argument("--data", required=True); p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--output", required=True); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-download", action="store_true"); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("validate"); p.add_argument("--weights", required=True); p.add_argument("--data", required=True)
    p.add_argument("--output", required=True); p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--device", default="cpu"); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--batch", type=int, default=2)
    for name in ("demo", "evaluate"):
        p = sub.add_parser(name); scene_args(p)
        p.add_argument("--catalogue"); p.add_argument("--output", required=True)
        if name == "demo":
            p.add_argument("--method", choices=METHODS, default="hsv"); p.set_defaults(fixed=True)
        else:
            p.add_argument("--methods", choices=METHODS, nargs="+", default=["oracle", "hsv"])
            p.add_argument("--count", type=int, default=30)
        p.add_argument("--weights"); p.add_argument("--yolov8-weights"); p.add_argument("--device", default="cpu")
        p.add_argument("--conf", type=float, default=.25); p.add_argument("--hybrid-high", type=float, default=.8)
        p.add_argument("--max-observations", type=int, default=12); p.add_argument("--max-attempts", type=int, default=6)
        p.add_argument("--gui", action="store_true"); p.add_argument("--plots", action="store_true")
    p = sub.add_parser("predict", help="Save a model overlay and measured RGB-D coordinates without moving the arm")
    scene_args(p); p.add_argument("--catalogue"); p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--output", required=True); p.set_defaults(hybrid_high=.8, yolov8_weights=None)
    p = sub.add_parser("summarize"); p.add_argument("path"); p.add_argument("--plots")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            import pybullet_data
            report = _environment()
            report["plane_asset_exists"] = (Path(pybullet_data.getDataPath())/"plane.urdf").is_file()
            report["asset_path_ascii"] = str(pybullet_data.getDataPath()).isascii()
            try:
                import torch
                report["cuda_available"] = torch.cuda.is_available()
            except ImportError:
                report["cuda_available"] = None
            _json(report)
        elif args.command == "scenes":
            from .evaluation import save_catalogue
            print(save_catalogue(_catalogue(args), args.output))
        elif args.command == "dataset":
            from .dataset import generate_dataset
            print(generate_dataset(args.output, args.count, args.seed, args.width, args.height,
                args.random_colors, args.negative_fraction, lighting_randomization=args.random_lighting))
        elif args.command == "check-dataset":
            from .dataset import validate_dataset
            report = validate_dataset(args.path); _json(report)
            if not report["valid"]: return 1
        elif args.command == "train":
            from .training import train_model
            print(train_model(args.data, args.model, args.output, args.epochs, args.imgsz, args.batch,
                              args.device, args.seed, args.resume, args.allow_download))
        elif args.command == "validate":
            from .training import validate_model
            _json(validate_model(args.weights, args.data, args.output, args.split, args.imgsz, args.batch, args.device))
        elif args.command in {"demo", "evaluate"}:
            _experiment(args, demo=args.command == "demo")
        elif args.command == "predict":
            from .sim import SimScene
            from .dataset import _write_rgb, sha256_file
            from .evaluation import save_catalogue, catalogue_fingerprint, new_run_id
            from .perception import localize, draw_predictions
            output = _fresh_output(args.output); spec = _catalogue(args)[0]
            predictors, model_info = _build_predictors(["yolo_mask"], args, spec)
            predictor = predictors["yolo_mask"]
            output.mkdir(parents=True, exist_ok=True)
            save_catalogue([spec], output/"scenes.json")
            started = time.perf_counter()
            with SimScene(spec) as scene:
                frame = scene.capture()
                stamp = time.perf_counter(); predictions = predictor.predict(frame)
                perception_s = time.perf_counter()-stamp
            _write_rgb(output/"predictions.png", draw_predictions(frame, predictions))
            stamp = time.perf_counter()
            coordinates = [{"class":p.class_name, "confidence":p.confidence, "bbox":p.bbox,
                "point_world":(loc:=localize(p,frame)).point_world, "reason":loc.reason} for p in predictions]
            localization_s = time.perf_counter()-stamp
            (output/"coordinates.json").write_text(json.dumps(coordinates,indent=2),encoding="utf-8")
            metadata = {"schema_version": 1, "run_id": new_run_id("prediction"), "command": "predict",
                "created_utc": datetime.now(timezone.utc).isoformat(), "environment": _environment(),
                "models": model_info, "scene_catalogue_sha256": catalogue_fingerprint([spec]),
                "scene_selection": "first scene of supplied catalogue, or generated scene",
                "timing_s": {"perception_s": perception_s, "localization_s": localization_s,
                             "prediction_pipeline_total_s": time.perf_counter()-started},
                "timing_scope": "Predict total includes scene load, RGB-D capture, inference, localization and output preparation; excludes model cold load/warmup recorded in models. It is not sorting-task total_s.",
                "files_sha256": {name: sha256_file(output/name) for name in ("predictions.png", "coordinates.json", "scenes.json")}}
            (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
            _json(coordinates)
        elif args.command == "summarize":
            from .evaluation import load_results, summarize, plot_results
            _json(summarize(load_results(args.path)))
            if args.plots: plot_results(args.path, args.plots)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
