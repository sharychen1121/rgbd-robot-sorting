"""Shared, explicit-unit contracts for learning, geometry and simulation.

RGB images are uint8 RGB, depth is optical-axis depth in metres, world positions
are metres, image boxes are (left, top, right, bottom) pixels, angles radians.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

CLASS_NAMES = ("cube", "cylinder", "box")
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
DESTINATIONS = {"cube": "A", "cylinder": "B", "box": "C"}
COLORS = {"cube": (0.90, 0.15, 0.15, 1.0), "cylinder": (0.15, 0.80, 0.20, 1.0), "box": (0.15, 0.35, 0.90, 1.0)}
DIMENSIONS = {"cube": (0.05, 0.05, 0.05), "cylinder": (0.05, 0.05, 0.08), "box": (0.08, 0.06, 0.05)}
TABLE_Z = 0.625


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    eye: tuple[float, float, float] = (0.5, 0.0, 2.2)
    target: tuple[float, float, float] = (0.5, 0.0, TABLE_Z)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 55.0
    near: float = 0.01
    far: float = 5.0

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 16 for v in (self.width, self.height)) or not np.isfinite((self.near, self.far, self.fov)).all() or not 0 < self.near < self.far or not 1 < self.fov < 179:
            raise ValueError("Invalid camera dimensions or clipping planes")
        for name in ("eye", "target", "up"):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("Camera eye, target and up must be finite 3-D vectors")
            object.__setattr__(self, name, tuple(float(v) for v in vector))
        forward = np.asarray(self.target) - np.asarray(self.eye)
        if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(np.cross(forward, self.up)) < 1e-9:
            raise ValueError("Camera viewing direction and up vector must define a nondegenerate frame")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))


@dataclass(frozen=True)
class ObjectSpec:
    class_name: str
    xy: tuple[float, float]
    yaw: float = 0.0
    color: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.class_name not in CLASS_IDS:
            raise ValueError(f"Unsupported class: {self.class_name}")
        if len(self.xy) != 2 or not np.isfinite((*self.xy, self.yaw)).all():
            raise ValueError("Object coordinates must be finite metres/radians")
        if self.color is not None and (len(self.color) != 4 or not all(0 <= v <= 1 for v in self.color)):
            raise ValueError("RGBA channels must lie in [0, 1]")
        object.__setattr__(self, "xy", tuple(float(v) for v in self.xy))
        if self.color is not None:
            object.__setattr__(self, "color", tuple(float(v) for v in self.color))


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    seed: int
    objects: tuple[ObjectSpec, ...]
    camera: CameraConfig = field(default_factory=CameraConfig)
    lighting: float = 1.0
    split: str = "test"
    regime: str = "standard"

    def __post_init__(self):
        if not isinstance(self.scene_id, str) or not self.scene_id.strip() or self.split not in {"train", "val", "test", "diagnostic"}:
            raise ValueError("Scene ID and a supported split are required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("Scene seed must be an integer; fractional seeds cannot be reproduced")
        if not isinstance(self.camera, CameraConfig) or any(not isinstance(obj, ObjectSpec) for obj in self.objects):
            raise ValueError("Scene camera and objects must use the shared typed contracts")
        if not 0.1 <= self.lighting <= 2:
            raise ValueError("Lighting multiplier must lie in [0.1, 2]")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "objects", tuple(self.objects))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneSpec":
        camera = dict(data.get("camera", {}))
        for key in ("eye", "target", "up"):
            if key in camera:
                camera[key] = tuple(camera[key])
        objects = tuple(ObjectSpec(class_name=o["class_name"], xy=tuple(o["xy"]), yaw=o.get("yaw", 0.0), color=tuple(o["color"]) if o.get("color") is not None else None) for o in data["objects"])
        return cls(scene_id=data["scene_id"], seed=data["seed"], objects=objects,
                   camera=CameraConfig(**camera), lighting=float(data.get("lighting", 1.0)),
                   split=data.get("split", "test"), regime=data.get("regime", "standard"))


@dataclass
class Frame:
    rgb: np.ndarray
    depth_m: np.ndarray
    view: np.ndarray
    projection: np.ndarray
    camera: CameraConfig
    segmentation: np.ndarray | None = None  # annotation/diagnostic only; never used by learned predictors


@dataclass
class Prediction:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    source: str = "model"


@dataclass
class Localization:
    prediction: Prediction
    point_world: tuple[float, float, float] | None
    yaw: float = 0.0
    depth_pixels: int = 0
    reason: str = "ok"

    @property
    def valid(self) -> bool:
        return self.point_world is not None and self.reason == "ok"


@dataclass
class ObjectOutcome:
    body_id: int
    predicted_class: str
    true_class: str
    status: str  # placed / mis_sorted / failed
    reason: str
    destination: str | None = None
    lifted: bool = False


@dataclass
class RunResult:
    scene_id: str
    method: str
    status: str
    reason: str
    total_objects: int
    outcomes: list[ObjectOutcome] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    observations: int = 0
    prediction_count: int = 0
    perception_s: float | None = None
    localization_s: float | None = None
    motion_s: float | None = None
    total_s: float | None = None
    initialization_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
