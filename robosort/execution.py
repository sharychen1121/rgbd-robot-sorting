"""Shared guarded backend for every perception method.

Only oracle diagnostics obtain target positions/classes from simulator truth.
Normal runs associate a nearby physical body by geometry to realize a Bullet
constraint; true classes are read only AFTER execution to score mis-sorting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import pybullet as p

from core.contracts import CartesianPose
from robot.controller import MotionConfig, MotionError, RobotController
from robot.destinations import DestinationPlanner
from robot.gripper import ConstraintGripper, GripperConfig
from robot.kinematics import Kinematics, downward_pose
from robosort.sim import SimScene
from robosort.types import CLASS_IDS, DESTINATIONS, DIMENSIONS, Localization, ObjectOutcome, Prediction, RunResult, TABLE_Z


@dataclass
class _Candidate:
    body: int
    localization: Localization
    grasp: CartesianPose
    pre_grasp: CartesianPose
    placement: object
    route: tuple[CartesianPose, ...]


def _nearby_body(scene: SimScene, point, remaining: set[int]) -> int:
    """Map visual contact coordinates to a simulator body WITHOUT class lookup."""
    matches = []
    for body in remaining:
        lo, hi = p.getAABB(body, physicsClientId=scene.client_id)
        center = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
        distance = math.dist(point[:2], center)
        if distance <= .035 and abs(point[2] - hi[2]) <= .035:
            matches.append((distance, body))
    matches.sort()
    if not matches:
        raise ValueError("no_body_near_visual_grasp")
    if len(matches) > 1 and matches[1][0] - matches[0][0] < .008:
        raise ValueError("ambiguous_geometric_contact")
    return matches[0][1]


class _Executor:
    def __init__(self, scene: SimScene, realtime: bool = False):
        self.scene = scene
        self.kin = Kinematics(scene.robot_id, scene.client_id)
        self.obstacles = tuple(scene.environment.values()) + tuple(scene.destinations.values()) + scene.object_ids
        self.controller = RobotController(self.kin, MotionConfig(realtime=realtime), self.obstacles)
        self.gripper = ConstraintGripper(scene.robot_id, self.kin.end_effector_link, scene.client_id,
                                         GripperConfig(standoff=.05))
        self.destinations = DestinationPlanner.from_scene(scene.destinations, scene.client_id)

    def observation_pose(self):
        if self.gripper.object_id is not None:
            raise MotionError("cannot_reobserve_with_payload")
        self.controller.move_to_joint_positions((0.,) * 7)
        if not self.scene.settle():
            raise MotionError("scene_not_settled")

    def plan(self, body: int, loc: Localization, yaw: float) -> _Candidate:
        """Validate a whole arm route in a temporary state before moving motors.

        Carried-object collisions are additionally rechecked using the measured
        attachment transform after pickup; preplanning is not a grasp guarantee.
        """
        name = loc.prediction.class_name
        if name not in CLASS_IDS:
            raise ValueError("unsupported_predicted_class")
        x, y, z = loc.point_world
        if not (.25 <= x <= .78 and -.32 <= y <= .32 and TABLE_Z + .012 <= z <= TABLE_Z + .15):
            raise ValueError("visual_target_outside_task_envelope")
        lo, hi = p.getAABB(body, physicsClientId=self.scene.client_id)
        # Ground-truth position is NOT substituted for the visual point.
        if math.dist((x, y), ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)) > self.gripper.config.xy_tolerance:
            raise ValueError("visual_grasp_outside_attachment_tolerance")
        placement = self.destinations.placement_for(name, DIMENSIONS[name])
        grasp = downward_pose(x, y, z + self.gripper.config.standoff, yaw)
        pre = downward_pose(x, y, grasp.position[2] + .14, yaw)
        place = downward_pose(*placement.object_position[:2],
                              placement.object_position[2] + DIMENSIONS[name][2] / 2 + self.gripper.config.standoff,
                              0.0)
        flight = max(pre.position[2], place.position[2] + .14)
        route = (pre, grasp, pre, CartesianPose((x, y, flight), place.orientation),
                 CartesianPose((*place.position[:2], flight), place.orientation), place)
        # Temporary FK inspection is rolled back, including live object states.
        with self.kin.inspection_configuration(self.kin.current_joints()):
            for i, pose in enumerate(route):
                targets, _ = self.controller.plan_end_effector(pose, allowed_bodies=(body,), linear=i != 0)
                for joint, value in zip(self.kin.joints, targets[-1]):
                    p.resetJointState(self.kin.robot_id, joint, value, physicsClientId=self.scene.client_id)
                p.performCollisionDetection(physicsClientId=self.scene.client_id)
        return _Candidate(body, loc, grasp, pre, placement, route)

    def execute(self, candidate: _Candidate) -> ObjectOutcome:
        body = candidate.body
        predicted_class = candidate.localization.prediction.class_name
        placed = candidate.placement
        lifted = False
        try:
            self.gripper.open_gripper()
            self.controller.move_end_effector(candidate.pre_grasp, allowed_bodies=(body,), linear=False)
            self.controller.move_end_effector(candidate.grasp, allowed_bodies=(body,))
            self.gripper.close_gripper(body)
            self.controller.payload_id = body
            self.controller.move_end_effector(candidate.pre_grasp, allowed_bodies=(body,))
            bottom = p.getAABB(body, physicsClientId=self.scene.client_id)[0][2]
            lifted = bottom >= TABLE_Z + .03
            if not lifted:
                raise MotionError("payload_not_lifted")
            # Use the constraint's measured relative transform for a correct
            # release, not simulator target coordinates for visual pickup.
            release = self.gripper.flange_pose_for_object(placed.object_position, placed.object_orientation)
            flight = max(candidate.pre_grasp.position[2], release.position[2] + .14)
            over_source = CartesianPose((*candidate.pre_grasp.position[:2], flight), release.orientation)
            over_destination = CartesianPose((*release.position[:2], flight), release.orientation)
            self.controller.execute_waypoints((over_source, over_destination, release), allowed_bodies=(body,))
            self.gripper.release_object()
            self.controller.payload_id = None
            self.controller.move_end_effector(over_destination, allowed_bodies=(body,))
            if not self.scene.settle():
                raise MotionError("placement_not_settled")
            lower, upper = p.getAABB(body, physicsClientId=self.scene.client_id)
            surface = self.destinations.surfaces[placed.bin_name]
            contacts = p.getContactPoints(bodyA=body, bodyB=placed.body_id, physicsClientId=self.scene.client_id)
            within = all(lower[d] >= surface.lower[d] - .002 and upper[d] <= surface.upper[d] + .002 for d in (0, 1))
            if not contacts or not within or abs(lower[2] - surface.upper[2]) > .008:
                raise MotionError("placement_not_supported_in_destination")
            self.destinations.mark_occupied(placed.bin_name)
            true_class = self.scene.truth[body]  # evaluator only, after action
            correct = DESTINATIONS[true_class] == placed.bin_name
            return ObjectOutcome(body, predicted_class, true_class, "placed" if correct else "mis_sorted",
                                 "destination_contact_verified" if correct else "wrong_class_destination", placed.bin_name, lifted)
        except Exception as exc:
            self.controller.hold()
            return ObjectOutcome(body, predicted_class, self.scene.truth[body], "failed",
                                 f"{type(exc).__name__}:{exc}", placed.bin_name, lifted)


def run_sorting(scene: SimScene, predictor=None, method: str = "oracle", representation: str = "mask",
                max_observations: int = 12, max_attempts: int = 6, realtime: bool | None = None) -> RunResult:
    """Execute one scene, with explicit stage failures and a bounded action budget.

    ``oracle`` is a separately labelled diagnostic and must not be reported as a
    learned result. All other modes call predictor.predict on a truth-free frame.
    Perception model construction/warm-up belongs outside this timed function.
    """
    if max_observations <= 0 or max_attempts <= 0:
        raise ValueError("Observation and attempt budgets must be positive")
    if method != "oracle" and predictor is None:
        raise ValueError("A predictor is required for non-oracle runs")
    if representation not in {"mask", "box"}:
        raise ValueError("Representation must be mask or box")
    if scene.client_id is None:
        raise RuntimeError("Open the SimScene before execution")
    result = RunResult(scene.spec.scene_id, method, "stopped", "not_started", len(scene.object_ids),
                       perception_s=None, localization_s=None, motion_s=0., total_s=0.)
    started = time.perf_counter()
    setup = time.perf_counter()
    executor = _Executor(scene, scene.gui if realtime is None else realtime)
    result.initialization_s = time.perf_counter() - setup
    remaining = set(scene.object_ids)
    try:
        for observation in range(max_observations):
            if not remaining:
                result.status = "completed" if all(o.status == "placed" for o in result.outcomes) else "failed"
                result.reason = "all_objects_placed" if result.status == "completed" else "incorrect_sorting"
                break
            motion_start = time.perf_counter()
            executor.observation_pose()
            result.motion_s += time.perf_counter() - motion_start
            result.observations += 1
            if method == "oracle":
                locs = [Localization(Prediction(item["class_name"], 1., (0, 0, 0, 0), source="oracle"),
                                     item["top_position"], item["yaw"])
                        for item in scene.ground_truth_poses() if item["body_id"] in remaining]
                result.prediction_count += len(locs)
            else:
                frame = scene.capture(include_truth=False)
                stamp = time.perf_counter()
                predictions = predictor.predict(frame)
                result.perception_s = (result.perception_s or 0.) + time.perf_counter() - stamp
                result.prediction_count += len(predictions)
                from robosort.perception import localize
                stamp = time.perf_counter()
                locs = [localize(prediction, frame, representation=representation) for prediction in predictions]
                result.localization_s = (result.localization_s or 0.) + time.perf_counter() - stamp
            candidates = []
            successful_bodies = set()
            for loc in sorted(locs, key=lambda x: -x.prediction.confidence):
                if not loc.valid:
                    result.rejections.append({"stage": "localization", "reason": loc.reason, "class_name": loc.prediction.class_name})
                    continue
                try:
                    body = _nearby_body(scene, loc.point_world, remaining)
                except ValueError as exc:
                    result.rejections.append({"stage": "association", "reason": str(exc), "class_name": loc.prediction.class_name})
                    continue
                if body in successful_bodies:
                    continue
                yaw_values = [loc.yaw, loc.yaw + math.pi, loc.yaw + math.pi / 2, loc.yaw - math.pi / 2]
                last_error = "no_kinematic_candidate"
                stamp = time.perf_counter()
                for yaw in yaw_values:
                    try:
                        candidate = executor.plan(body, loc, yaw)
                        candidates.append(candidate)
                        successful_bodies.add(body)  # only AFTER a feasible candidate
                        break
                    except (ValueError, MotionError) as exc:
                        last_error = str(exc)
                result.motion_s += time.perf_counter() - stamp
                if body not in successful_bodies:
                    result.rejections.append({"stage": "planning", "reason": last_error, "class_name": loc.prediction.class_name})
            if not candidates:
                result.reason = "no_valid_targets"
                break
            if len(result.outcomes) >= max_attempts:
                result.reason = "pick_attempt_limit"
                break
            stamp = time.perf_counter()
            outcome = executor.execute(candidates[0])
            result.motion_s += time.perf_counter() - stamp
            result.outcomes.append(outcome)
            if outcome.status in {"placed", "mis_sorted"}:
                remaining.remove(outcome.body_id)
            else:
                # A physical failure may leave a payload attached. Hold safely;
                # do not blindly continue with a changed, unobserved state.
                result.status, result.reason = "failed", outcome.reason
                break
        else:
            result.reason = "observation_limit"
        if not remaining:
            result.status = "completed" if all(o.status == "placed" for o in result.outcomes) else "failed"
            result.reason = "all_objects_placed" if result.status == "completed" else "incorrect_sorting"
    except Exception as exc:
        result.status, result.reason = "failed", f"{type(exc).__name__}:{exc}"
        result.rejections.append({"stage": "runtime", "reason": result.reason})
    finally:
        executor.controller.hold()
        result.total_s = time.perf_counter() - started
    return result

