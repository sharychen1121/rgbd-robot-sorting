"""Explicit training/validation entry points and portable checkpoint metadata."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import shutil
import sys

from .dataset import sha256_file
from .perception import resolve_device


def package_versions():
    result = {"python":sys.version.split()[0]}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pybullet"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _load_model(model: str | Path, allow_download: bool):
    path = Path(model)
    if not path.is_file() and not allow_download:
        raise FileNotFoundError(f"Model {model} is not local. Supply a checkpoint or explicitly enable pretrained downloads.")
    from ultralytics import YOLO
    try:
        return YOLO(str(model))
    except Exception as error:
        raise RuntimeError(f"Could not load/download model {model}: {error}") from error


def train_model(data: str | Path, model: str | Path = "yolo26n-seg.pt", run_dir: str | Path = "runs/yolo26-seg",
                epochs: int = 30, imgsz: int = 640, batch: int = 2, device: str = "cpu", seed: int = 42,
                resume: bool = False, allow_download: bool = False, workers: int = 0) -> Path:
    """Train explicitly, copy best weights to ``run_dir/best.pt`` and write metadata.

    A nonempty run directory is rejected unless resume is requested. The
    resulting checkpoint is suitable for inference without downloading assets.
    """
    data, run_dir = Path(data).resolve(), Path(run_dir).resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    if epochs < 1 or imgsz < 32 or batch < 1:
        raise ValueError("Positive epochs/batch and image size >=32 are required")
    resolved_device = resolve_device(device)
    if resume:
        checkpoint = run_dir/"fit"/"weights"/"last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        network = _load_model(checkpoint, False)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Training run already exists: {run_dir}; choose a new run name or resume")
        network = _load_model(model, allow_download)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Absolute project prevents Ultralytics' global runs_dir from nesting paths.
    args = dict(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=resolved_device,
                seed=seed, workers=workers, project=str(run_dir), name="fit", exist_ok=resume,
                deterministic=True, patience=max(5,epochs//3), plots=True, save=True)
    metadata = {"schema_version":1,"started_utc":datetime.now(timezone.utc).isoformat(),
                "model_requested":str(model),"task":str(network.task),"arguments":args,
                "data_sha256":sha256_file(data),"versions":package_versions(),
                "resume":resume,"status":"running"}
    manifest=data.parent/'manifest.json'
    if manifest.is_file():
        metadata['dataset_manifest_sha256']=sha256_file(manifest)
    if Path(model).is_file():
        metadata['initial_checkpoint_sha256']=sha256_file(model)
    (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    try:
        network.train(**args, resume=resume)
        fit_dir = Path(network.trainer.save_dir)
        best = fit_dir/"weights"/"best.pt"
        if not best.is_file():
            raise RuntimeError(f"Training returned without a best checkpoint: {best}")
        destination = run_dir/"best.pt"
        shutil.copy2(best, destination)
        metadata.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat(),
                        checkpoint=str(destination),checkpoint_sha256=sha256_file(destination),
                        fit_directory=str(fit_dir),completed_epochs=int(network.trainer.epoch)+1,
                        checkpoint_selection="Ultralytics best validation fitness; held-out test is not used for selection")
        (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        return destination
    except Exception as error:
        metadata.update(status="failed",error=f"{type(error).__name__}: {error}")
        (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        raise


def validate_model(weights: str | Path, data: str | Path, output: str | Path, split: str = "test",
                   imgsz: int = 640, batch: int = 2, device: str = "cpu") -> dict:
    """Evaluate a named split and save real box/mask metrics with provenance."""
    if split not in {"train","val","test"}:
        raise ValueError("Split must be train, val or test")
    weights,data,output = Path(weights).resolve(),Path(data).resolve(),Path(output).resolve()
    if not weights.is_file() or not data.is_file():
        raise FileNotFoundError("Both local checkpoint and dataset YAML must exist")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Validation output already exists: {output}")
    output.mkdir(parents=True,exist_ok=True)
    network = _load_model(weights, False)
    metrics = network.val(data=str(data),split=split,imgsz=imgsz,batch=batch,
                          device=resolve_device(device),workers=0,
                          project=str(output),name="evaluation",plots=True,exist_ok=False)
    values = {key:float(value) for key,value in metrics.results_dict.items()}
    result = {"split":split,"checkpoint":str(weights),"checkpoint_sha256":sha256_file(weights),
              "dataset_yaml":str(data),"data_sha256":sha256_file(data),
              "versions":package_versions(),"metrics":values,
              "completed_utc":datetime.now(timezone.utc).isoformat()}
    (output/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
