"""Learned/HSV observations and camera-calibrated RGB-D localization.

The learned paths never read ``Frame.segmentation``. That optional simulator
buffer is reserved for dataset annotation and oracle diagnostics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import cv2
import numpy as np

from .types import CLASS_NAMES, Frame, Localization, Prediction


def _check_frame(frame: Frame) -> None:
    if frame.rgb.dtype != np.uint8 or frame.rgb.shape != (frame.camera.height, frame.camera.width, 3):
        raise ValueError("Frame.rgb must be H x W x 3 uint8 RGB at the configured resolution")
    if frame.depth_m.shape != frame.rgb.shape[:2]:
        raise ValueError("Depth and RGB must be spatially aligned")
    if frame.depth_m.dtype.kind not in "fiu":
        raise ValueError("Depth must contain real numeric optical-axis distances in metres")
    if np.asarray(frame.view).shape != (4, 4) or np.asarray(frame.projection).shape != (4, 4):
        raise ValueError("View and projection must be actual 4 x 4 matrices, not flattened column arrays")
    for name, matrix in (("view", frame.view), ("projection", frame.projection)):
        matrix = np.asarray(matrix, dtype=float)
        if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 4:
            raise ValueError(f"The {name} matrix must be finite and invertible")
    # Optical-axis depth below assumes the OpenGL perspective convention used
    # by PyBullet. Reject an orthographic/general matrix instead of producing
    # an apparently valid but incorrect world point.
    projection = np.asarray(frame.projection, dtype=float)
    if not np.allclose(projection[3], (0., 0., -1., 0.), atol=1e-8) or not np.allclose(projection[2, :2], 0., atol=1e-8):
        raise ValueError("Optical-axis unprojection requires an OpenGL perspective matrix")


def resolve_device(device: str = "cpu") -> str:
    """Validate the requested device; an unavailable GPU never silently falls back."""
    import torch
    device = str(device).lower()
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return device
    index = device.removeprefix("cuda:")
    if not index.isdigit():
        raise ValueError("Use cpu, auto, 0, or cuda:0 for device")
    if not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device!r} is unavailable; select --device cpu or install a compatible CUDA PyTorch build")
    return index


class UltralyticsPredictor:
    """Load one local checkpoint lazily and reuse it for every frame/trial.

    ``weights`` must exist: evaluation never downloads an unexpected model.
    Explicit pretrained downloads are supported only by the training helper.
    """
    def __init__(self, weights: str | Path, conf: float = 0.25, device: str = "cpu", *, model=None):
        if not 0 <= conf <= 1 or not math.isfinite(conf):
            raise ValueError("Confidence threshold must lie in [0, 1]")
        self.weights = Path(weights)
        self.conf = float(conf)
        self.device = device
        self.model = model
        self.resolved_device: str | None = None

    def load(self):
        if self.model is not None:
            return self.model
        if not self.weights.is_file():
            raise FileNotFoundError(f"Trained checkpoint not found: {self.weights}. Generate a dataset and train a segment model first.")
        self.resolved_device = resolve_device(self.device)
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights))
        return self.model

    def predict(self, frame: Frame) -> list[Prediction]:
        _check_frame(frame)
        model = self.load()
        # NumPy images in Ultralytics use BGR. Keep the public Frame contract RGB.
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        results = model.predict(source=image, conf=self.conf,
                                device=self.resolved_device or self.device,
                                retina_masks=True, verbose=False, save=False)
        predictions: list[Prediction] = []
        height, width = frame.rgb.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else None
            for i, (bbox, score, class_id) in enumerate(zip(boxes, scores, classes)):
                if not math.isfinite(float(score)) or not 0 <= float(score) <= 1 or not math.isfinite(float(class_id)) or int(class_id) != class_id:
                    continue
                name = str(result.names[int(class_id)])
                if name not in CLASS_NAMES or not np.isfinite(bbox).all():
                    continue
                x1, y1, x2, y2 = map(float, bbox)
                box = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = mask > 0.5
                predictions.append(Prediction(name, float(score), box, mask, "yolo"))
        return predictions


def _class_color_mask(rgb: np.ndarray, class_name: str) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if class_name == "cube":
        return cv2.inRange(hsv, (0, 80, 55), (12, 255, 255)) | cv2.inRange(hsv, (168, 80, 55), (179, 255, 255))
    if class_name == "cylinder":
        return cv2.inRange(hsv, (35, 70, 45), (88, 255, 255))
    if class_name == "box":
        return cv2.inRange(hsv, (92, 70, 45), (137, 255, 255))
    raise ValueError(f"Unsupported class {class_name}")


class HSVPredictor:
    """Fixed class-color reference. Scores are heuristic, not learned probabilities."""
    def __init__(self, min_area: float = 35.):
        self.min_area = float(min_area)

    def load(self):
        return self

    def predict(self, frame: Frame) -> list[Prediction]:
        _check_frame(frame)
        out: list[Prediction] = []
        min_area = max(3., self.min_area*frame.rgb.shape[0]*frame.rgb.shape[1]/(640*480))
        for name in CLASS_NAMES:
            mask = _class_color_mask(frame.rgb, name)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                isolated = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(isolated, [contour], -1, 1, -1)
                out.append(Prediction(name, min(1., area / max(w*h, 1)),
                                      (float(x), float(y), float(x+w), float(y+h)), isolated.astype(bool), "hsv"))
        return sorted(out, key=lambda p: -p.confidence)


class HybridPredictor:
    """A reproducible color/shape-check reference around learned candidates.

    This intentionally retains the fixed-color assumption. Unlike the old
    experimental 0.03 gate, thresholds are explicit and validation-tunable.
    """
    def __init__(self, predictor: UltralyticsPredictor, low: float = .25, high: float = .8, min_area: float = 35.):
        if not 0 <= low < high <= 1:
            raise ValueError("Hybrid thresholds must satisfy 0 <= low < high <= 1")
        self.predictor, self.low, self.high, self.min_area = predictor, low, high, min_area

    def load(self):
        return self.predictor.load()

    def predict(self, frame: Frame) -> list[Prediction]:
        out = []
        for pred in self.predictor.predict(frame):
            if pred.confidence < self.low:
                continue
            h, w = frame.rgb.shape[:2]
            min_area = max(3., self.min_area*h*w/(640*480))
            x1, y1, x2, y2 = _integer_box(pred.bbox, w, h)
            color = _class_color_mask(frame.rgb, pred.class_name)
            roi = np.zeros_like(color)
            roi[y1:y2, x1:x2] = color[y1:y2, x1:x2]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else None
            if contour is None or cv2.contourArea(contour) < min_area:
                # High-confidence learned masks can be retained, but the
                # medium band requires independent evidence from the pixels.
                if pred.confidence >= self.high and pred.mask is not None:
                    out.append(Prediction(pred.class_name, pred.confidence, pred.bbox, pred.mask, "hybrid_high"))
                continue
            area = cv2.contourArea(contour)
            hull = cv2.contourArea(cv2.convexHull(contour))
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            if hull <= 0 or area/hull < .75 or rw*rh <= 0 or area/(rw*rh) < .5:
                continue
            refined = np.zeros_like(roi)
            cv2.drawContours(refined, [contour], -1, 1, -1)
            out.append(Prediction(pred.class_name, pred.confidence, pred.bbox, refined.astype(bool), "hybrid_color"))
        return out


def _integer_box(box, width, height):
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError("BBox must contain four finite pixel coordinates")
    # Both ends need both bounds: a negative right edge would otherwise make
    # NumPy's [0:-1] select nearly the entire image for an off-screen box.
    x1, y1 = np.clip(int(math.floor(box[0])), 0, width), np.clip(int(math.floor(box[1])), 0, height)
    x2, y2 = np.clip(int(math.ceil(box[2])), 0, width), np.clip(int(math.ceil(box[3])), 0, height)
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    return x1, y1, x2, y2


def unproject_pixels(frame: Frame, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Inverse OpenGL projection using optical-axis depth, returning world XYZ."""
    _check_frame(frame)
    projection = np.asarray(frame.projection, float)
    u, v, depth = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(depth, float))
    flat_depth = depth.ravel()
    if not np.isfinite(flat_depth).all() or np.any(flat_depth <= 0) or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Unprojection requires finite pixel coordinates and positive depth")
    ndc_x = 2*(u.ravel()+.5)/frame.camera.width-1
    ndc_y = 1-2*(v.ravel()+.5)/frame.camera.height
    # Perspective OpenGL has clip_w = -z_camera = optical depth.
    ndc_z = -projection[2, 2] + projection[2, 3] / flat_depth
    clip = np.column_stack((ndc_x, ndc_y, ndc_z, np.ones_like(flat_depth)))
    camera_points = (np.linalg.inv(projection) @ clip.T).T
    camera_points /= camera_points[:, 3, None]
    world = (np.linalg.inv(np.asarray(frame.view, float)) @ camera_points.T).T
    return world[:, :3] / world[:, 3, None]


def localize(prediction: Prediction, frame: Frame, representation: str = "mask", *,
             erode_pixels: int = 1, min_pixels: int = 6, surface_tolerance_m: float = .008) -> Localization:
    """Estimate a visible top-surface point from predicted pixels and measured depth.

    ``box`` and ``mask`` use exactly the same depth estimator. A same-checkpoint
    comparison therefore isolates the chosen image region. No true object pose,
    simulator segmentation, or fixed class-color mask enters this estimator.
    """
    _check_frame(frame)
    if prediction.class_name not in CLASS_NAMES:
        return Localization(prediction, None, reason="unsupported_class")
    if representation not in {"mask", "box"}:
        raise ValueError("Representation must be mask or box")
    if isinstance(erode_pixels, bool) or not isinstance(erode_pixels, int) or erode_pixels < 0:
        raise ValueError("erode_pixels must be a nonnegative integer")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int) or min_pixels < 1:
        raise ValueError("min_pixels must be a positive integer")
    if not math.isfinite(surface_tolerance_m) or surface_tolerance_m <= 0:
        raise ValueError("surface_tolerance_m must be a finite positive distance")
    region = np.zeros(frame.rgb.shape[:2], np.uint8)
    if representation == "mask":
        if prediction.mask is None or prediction.mask.shape != region.shape:
            return Localization(prediction, None, reason="missing_or_misaligned_mask")
        mask = np.asarray(prediction.mask)
        if mask.dtype.kind not in "bifu" or not np.isfinite(mask).all() or np.any(mask < 0):
            return Localization(prediction, None, reason="invalid_mask")
        # Preserve probabilities and bool/0-255 masks without truncating .9 to
        # zero or allowing negative values to wrap into foreground pixels.
        region = (mask > .5).astype(np.uint8)
    else:
        x1, y1, x2, y2 = _integer_box(prediction.bbox, region.shape[1], region.shape[0])
        region[y1:y2, x1:x2] = 1
    if erode_pixels > 0:
        kernel = np.ones((2*erode_pixels+1, 2*erode_pixels+1), np.uint8)
        region = cv2.erode(region, kernel)
    valid = region.astype(bool) & np.isfinite(frame.depth_m) & (frame.depth_m > frame.camera.near) & (frame.depth_m < frame.camera.far-.001)
    rows, cols = np.nonzero(valid)
    if len(rows) < min_pixels:
        return Localization(prediction, None, depth_pixels=len(rows), reason="insufficient_depth")
    depths = frame.depth_m[rows, cols].astype(float)
    # Use the nearer surface band, suppressing farther background. This is
    # shared by the ablation's box and mask versions and is not a GT filter.
    anchor = float(np.quantile(depths, .10))
    select = (depths >= max(frame.camera.near, anchor-surface_tolerance_m)) & (depths <= anchor+surface_tolerance_m)
    rows, cols, depths = rows[select], cols[select], depths[select]
    if len(rows) < min_pixels:
        return Localization(prediction, None, depth_pixels=len(rows), reason="insufficient_surface_depth")
    points = unproject_pixels(frame, cols, rows, depths)
    centre = np.median(points, axis=0)
    yaw = 0.
    if prediction.class_name != "cylinder" and len(points) >= 3:
        covariance = np.cov(points[:, :2], rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, np.argmax(values)]
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        period = math.pi/2 if prediction.class_name == "cube" else math.pi
        yaw %= period
    return Localization(prediction, tuple(float(x) for x in centre), float(yaw), len(rows), "ok")


def draw_predictions(frame: Frame, predictions: Iterable[Prediction]) -> np.ndarray:
    """Return an RGB overlay suitable for reports without changing the input."""
    out = frame.rgb.copy()
    for pred in predictions:
        if pred.mask is not None:
            selected = pred.mask.astype(bool)
            out[selected] = (out[selected]*.6 + np.array([255, 180, 25])*.4).astype(np.uint8)
        x1,y1,x2,y2 = _integer_box(pred.bbox, out.shape[1], out.shape[0])
        cv2.rectangle(out, (x1,y1), (x2,y2), (20,220,100), 1)
        cv2.putText(out, f"{pred.class_name} {pred.confidence:.2f}", (x1,max(12,y1-3)), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return out
