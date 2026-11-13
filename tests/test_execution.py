"""Truth isolation, safety guards and physical placement regressions."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from robosort.execution import _Executor, _nearby_body, run_sorting
from robosort.perception import HSVPredictor
from robosort.sim import SimScene, demo_scene
from robosort.types import Localization, ObjectSpec, Prediction, SceneSpec
from robot.controller import MotionError


class ExecutionTests(unittest.TestCase):
    def test_empty_oracle_and_missing_predictor(self):
        with SimScene(SceneSpec('empty', 1, ())) as scene:
            result = run_sorting(scene, method='oracle')
            self.assertEqual((result.status, result.reason), ('completed', 'all_objects_placed'))
            self.assertEqual(result.total_objects, 0)
            self.assertEqual(result.outcomes, [])
            self.assertEqual(result.observations, 0)
            self.assertIsNone(result.perception_s)
            self.assertIsNone(result.localization_s)
            with self.assertRaises(ValueError):
                run_sorting(scene, method='yolo_mask')

    def test_geometric_association_and_failed_planning_do_not_change_state(self):
        with SimScene(demo_scene()) as scene:
            executor = _Executor(scene)
            target = scene.ground_truth_poses()[0]
            body = _nearby_body(scene, target['top_position'], set(scene.object_ids))
            self.assertEqual(body, target['body_id'])
            joints = executor.kin.current_joints()
            position = p.getBasePositionAndOrientation(body, physicsClientId=scene.client_id)
            loc = Localization(Prediction('box', 1., (0, 0, 0, 0)), (.50, .15, .20))
            with self.assertRaisesRegex(ValueError, 'outside_task_envelope'):
                executor.plan(body, loc, 0.)
            np.testing.assert_array_equal(joints, executor.kin.current_joints())
            self.assertEqual(position, p.getBasePositionAndOrientation(body, physicsClientId=scene.client_id))
            invalid = tuple(hi + 1 for hi in executor.kin.upper)
            with self.assertRaises(MotionError):
                executor.controller.move_to_joint_positions(invalid)
            np.testing.assert_array_equal(joints, executor.kin.current_joints())

    def test_contact_permission_does_not_disable_collision_checks_for_arm_links(self):
        with SimScene(SceneSpec('empty', 1, ())) as scene:
            kin = _Executor(scene).kin
            obstacle = 99
            def points(a, b, *args, **kwargs):
                if b != obstacle:
                    return []
                contact = [0] * 10
                contact[3] = 2
                contact[8] = -.01
                return [contact]
            with patch('robot.kinematics.p.getClosestPoints', side_effect=points):
                reason = kin.current_collision_reason((obstacle,), allowed_bodies=(obstacle,))
            self.assertEqual(reason, 'arm_collision:body=99:link=2')

    def test_wrong_visual_class_is_physically_mis_sorted_without_truth_correction(self):
        # The single red cube is intentionally predicted as box. The backend
        # must execute the predicted C destination and score it as mis-sorted.
        spec = SceneSpec('wrong-class', 9, (ObjectSpec('cube', (.50, .15)),))
        class WrongClassPredictor:
            def __init__(self):
                self.reference = HSVPredictor()
            def predict(self, image):
                if image.segmentation is not None:
                    raise AssertionError('Privileged segmentation leaked to inference')
                return [replace(pred, class_name='box') for pred in self.reference.predict(image)
                        if pred.class_name == 'cube']
        with SimScene(spec) as scene:
            with patch.object(scene, 'ground_truth_poses', side_effect=AssertionError('Oracle API used')):
                result = run_sorting(scene, WrongClassPredictor(), method='diagnostic_wrong_classifier')
            self.assertEqual(len(result.outcomes), 1, result.to_dict())
            outcome = result.outcomes[0]
            self.assertEqual(outcome.status, 'mis_sorted', result.to_dict())
            self.assertEqual((outcome.true_class, outcome.predicted_class, outcome.destination), ('cube', 'box', 'C'))
            self.assertTrue(outcome.lifted)
            self.assertEqual(result.reason, 'incorrect_sorting')
            self.assertEqual(result.status, 'failed')


if __name__ == '__main__':
    unittest.main()
