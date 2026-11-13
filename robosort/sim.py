"""Deterministic PyBullet scenes and synchronized RGB-D capture.

The segmentation buffer and ``truth`` metadata are privileged annotation /
evaluation information. Learned predictors receive captures without that buffer.
The robot has no physical fingers: execution uses a guarded fixed constraint.
"""
from __future__ import annotations

import colorsys
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pybullet as p
import pybullet_data

from robosort.types import CameraConfig, CLASS_NAMES, COLORS, DIMENSIONS, Frame, ObjectSpec, SceneSpec, TABLE_Z


def _resources() -> Path:
    source = Path(pybullet_data.getDataPath())
    override = os.environ.get("ROBOSORT_ASSET_DIR")
    if override:
        target = Path(override).expanduser().resolve()
    elif str(source).isascii():
        return source
    else:
        # Bullet's native Windows file I/O may reject Unicode asset paths. Copy
        # only the required assets into an ASCII cache; never move/delete data.
        target = Path(tempfile.gettempdir()) / "robosort-assets" / hashlib.sha256(str(source).encode()).hexdigest()[:12]
    if not str(target).isascii():
        raise RuntimeError("Set ROBOSORT_ASSET_DIR to an ASCII-only directory for Bullet resource files")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("table", "kuka_iiwa"):
        shutil.copytree(source / name, target / name, dirs_exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and (item.name.startswith("plane") or item.suffix in {".png", ".jpg", ".mtl"}):
            shutil.copy2(item, target / item.name)
    return target


def demo_scene(seed: int = 42, camera: CameraConfig | None = None) -> SceneSpec:
    return SceneSpec("demo", seed, (
        ObjectSpec("cube", (0.50, 0.15), 0.0),
        ObjectSpec("cylinder", (0.58, -0.15), 0.0),
        ObjectSpec("box", (0.68, 0.05), 0.35),
    ), camera or CameraConfig(), split="diagnostic", regime="fixed")


default_scene = demo_scene


def sample_scene(seed: int, scene_id: str | None = None, regime: str = "standard", split: str = "test",
                 random_colors: bool = False, camera: CameraConfig | None = None,
                 min_objects: int = 3, max_objects: int = 3, lighting: float = 1.0,
                 color_randomization: bool | None = None) -> SceneSpec:
    """Sample a frozen scene, not a scene selected according to method success.

    Standard scenes maintain 13 cm centre spacing. Stress scenes use wider
    workspace/closer spacing and are reported separately. They are not promised
    robot-reachable; oracle diagnostics establish backend limitations.
    """
    if not 0 <= min_objects <= max_objects <= len(CLASS_NAMES):
        raise ValueError("Object count must satisfy 0 <= min <= max <= 3")
    if regime not in {"standard", "stress", "appearance", "color", "lighting"}:
        raise ValueError("Unsupported scene regime")
    rng = random.Random(seed)
    if color_randomization is not None:
        random_colors = color_randomization
    if regime in {"appearance", "color"}:
        random_colors = True
    if regime in {"appearance", "lighting"}:
        lighting = rng.uniform(0.55, 1.25)
    n = rng.randint(min_objects, max_objects)
    names = rng.sample(list(CLASS_NAMES), n)
    points: list[tuple[float, float]] = []
    clearance = 0.13 if regime != "stress" else 0.065
    bounds = ((0.38, 0.70), (-0.18, 0.17)) if regime != "stress" else ((0.32, 0.75), (-0.25, 0.24))
    for _ in names:
        for attempt in range(1000):
            xy = (rng.uniform(*bounds[0]), rng.uniform(*bounds[1]))
            if all(math.dist(xy, other) >= clearance for other in points):
                points.append(xy)
                break
        else:
            raise RuntimeError("Cannot sample a non-overlapping scene within bounded attempts")
    objects = []
    for name, xy in zip(names, points):
        color = None
        if random_colors:
            color = (*colorsys.hsv_to_rgb(rng.random(), rng.uniform(.55, .95), rng.uniform(.55, .95)), 1.0)
        objects.append(ObjectSpec(name, xy, rng.uniform(-math.pi, math.pi), color))
    return SceneSpec(scene_id or f"{regime}-{seed}", seed, tuple(objects), camera or CameraConfig(), lighting, split, regime)


class SimScene:
    """One physics connection with known collision geometry and sensor capture."""

    def __init__(self, spec: SceneSpec, gui: bool = False):
        self.spec, self.gui = spec, gui
        self.client_id: int | None = None
        self.robot_id: int | None = None
        self.environment: dict[str, int] = {}
        self.destinations: dict[str, int] = {}
        self.truth: dict[int, str] = {}
        self._spec_by_body: dict[int, ObjectSpec] = {}

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(self.truth)

    def __enter__(self) -> "SimScene":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self) -> "SimScene":
        if self.client_id is not None:
            raise RuntimeError("Scene is already open")
        # Reopening the same Scene wrapper must not retain body handles from a
        # disconnected (or partially constructed) physics connection.
        self.environment.clear()
        self.destinations.clear()
        self.truth.clear()
        self._spec_by_body.clear()
        self.robot_id = None
        client = p.connect(p.GUI if self.gui else p.DIRECT)
        if client < 0:
            raise RuntimeError("Unable to connect to PyBullet")
        self.client_id = client
        try:
            assets = _resources()
            p.setAdditionalSearchPath(str(assets), physicsClientId=client)
            p.setGravity(0, 0, -9.81, physicsClientId=client)
            p.setTimeStep(1 / 240, physicsClientId=client)
            p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, numSolverIterations=150, physicsClientId=client)
            self.environment["plane"] = p.loadURDF(str(assets / "plane.urdf"), physicsClientId=client)
            self.environment["table"] = p.loadURDF(str(assets / "table/table.urdf"), basePosition=(.5, 0, 0), physicsClientId=client)
            self.robot_id = p.loadURDF(str(assets / "kuka_iiwa/model.urdf"), basePosition=(0, 0, TABLE_Z),
                                       useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                                       physicsClientId=client)
            p.setJointMotorControlArray(self.robot_id, list(range(7)), p.POSITION_CONTROL,
                                         targetPositions=[0.] * 7, forces=[300.] * 7, physicsClientId=client)
            for i, (key, xy, color) in enumerate(zip(("destination_1", "destination_2", "destination_3"),
                                                   ((.35, .30), (.50, .30), (.65, .30)),
                                                   ((.90, .70, .10, 1), (.70, .20, .80, 1), (.10, .75, .75, 1)))):
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(.055, .045, .003), rgbaColor=color, physicsClientId=client)
                collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.055, .045, .003), physicsClientId=client)
                self.destinations[key] = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual, basePosition=(*xy, TABLE_Z + .003), physicsClientId=client)
            for obj in self.spec.objects:
                dims = DIMENSIONS[obj.class_name]
                color = obj.color or COLORS[obj.class_name]
                if obj.class_name == "cylinder":
                    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=dims[0] / 2, length=dims[2], rgbaColor=color, physicsClientId=client)
                    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=dims[0] / 2, height=dims[2], physicsClientId=client)
                else:
                    half = [v / 2 for v in dims]
                    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=client)
                    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
                body = p.createMultiBody(baseMass=.1, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                    basePosition=(*obj.xy, TABLE_Z + dims[2] / 2 + .001),
                    baseOrientation=p.getQuaternionFromEuler((0, 0, obj.yaw)), physicsClientId=client)
                p.changeDynamics(body, -1, lateralFriction=.8, spinningFriction=.01, rollingFriction=.001,
                                 restitution=0, physicsClientId=client)
                self.truth[body] = obj.class_name
                self._spec_by_body[body] = obj
            self.settle()
            if self.gui:
                p.resetDebugVisualizerCamera(1.5, 45, -45, (.5, 0, TABLE_Z), physicsClientId=client)
            return self
        except Exception:
            self.close()
            raise

    def settle(self, steps: int = 480) -> bool:
        client = self._client()
        stable = 0
        for i in range(steps):
            p.stepSimulation(physicsClientId=client)
            if i % 12:
                continue
            velocities = [p.getBaseVelocity(body, physicsClientId=client) for body in self.object_ids]
            if all(np.linalg.norm(v) < .005 and np.linalg.norm(w) < .05 for v, w in velocities):
                stable += 1
                if stable >= 10:
                    return True
            else:
                stable = 0
        return False

    def _client(self) -> int:
        if self.client_id is None or not p.isConnected(self.client_id):
            raise RuntimeError("Scene must be open")
        return self.client_id

    def capture(self, include_truth: bool = False) -> Frame:
        client = self._client()
        cam = self.spec.camera
        view_flat = p.computeViewMatrix(cam.eye, cam.target, cam.up)
        projection_flat = p.computeProjectionMatrixFOV(cam.fov, cam.width / cam.height, cam.near, cam.far)
        _, _, rgba, depth_buffer, segmentation = p.getCameraImage(cam.width, cam.height,
            viewMatrix=view_flat, projectionMatrix=projection_flat, renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            lightAmbientCoeff=min(.9, .6 * self.spec.lighting),
            lightDiffuseCoeff=min(.9, .4 * self.spec.lighting), lightSpecularCoeff=.05,
            physicsClientId=client)
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(cam.height, cam.width, 4)[..., :3].copy()
        buffer = np.asarray(depth_buffer, dtype=np.float64).reshape(cam.height, cam.width)
        depth = (cam.far * cam.near / (cam.far - (cam.far - cam.near) * buffer)).astype(np.float32)
        seg = np.asarray(segmentation, dtype=np.int32).reshape(cam.height, cam.width).copy() if include_truth else None
        return Frame(rgb, depth, np.asarray(view_flat).reshape(4, 4, order="F"),
                     np.asarray(projection_flat).reshape(4, 4, order="F"), cam, seg)

    def ground_truth_poses(self) -> list[dict]:
        """Privileged oracle/evaluation API; never consumed by learned inference."""
        result = []
        for body, name in self.truth.items():
            xyz, quat = p.getBasePositionAndOrientation(body, physicsClientId=self._client())
            result.append({"body_id": body, "class_name": name, "position": tuple(xyz),
                           "top_position": (xyz[0], xyz[1], xyz[2] + DIMENSIONS[name][2] / 2),
                           "yaw": p.getEulerFromQuaternion(quat)[2]})
        return result

    def close(self):
        if self.client_id is not None and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
        self.client_id = None

