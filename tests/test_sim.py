"""Sensor and scene contracts, using the actual TinyRenderer physics backend."""
import itertools
import math
import unittest

import numpy as np
import pybullet as p

from robosort.sim import SimScene, demo_scene, sample_scene
from robosort.types import CameraConfig, CLASS_NAMES, SceneSpec


class SceneTests(unittest.TestCase):
    def test_sampler_is_reproducible_separated_and_serializable(self):
        a = sample_scene(478, random_colors=True)
        self.assertEqual(a, sample_scene(478, random_colors=True))
        self.assertEqual(a, SceneSpec.from_dict(a.to_dict()))
        self.assertEqual({o.class_name for o in a.objects}, set(CLASS_NAMES))
        for left, right in itertools.combinations(a.objects, 2):
            self.assertGreaterEqual(math.dist(left.xy, right.xy), .13)
        self.assertTrue(all(o.color is not None for o in a.objects))
        self.assertNotEqual(a.objects, sample_scene(479, random_colors=True).objects)

    def test_empty_scenes_supported_and_invalid_counts_rejected(self):
        self.assertEqual(sample_scene(1, min_objects=0, max_objects=0).objects, ())
        for low, high in ((-1, 3), (3, 2), (0, 4)):
            with self.assertRaises(ValueError):
                sample_scene(1, min_objects=low, max_objects=high)

    def test_capture_truth_is_opt_in_and_rgbd_is_synchronized(self):
        camera = CameraConfig(width=160, height=120)
        with SimScene(demo_scene(camera=camera)) as sim:
            normal = sim.capture()
            annotated = sim.capture(include_truth=True)
            self.assertIsNone(normal.segmentation)
            self.assertEqual(annotated.segmentation.shape, (120, 160))
            self.assertEqual(normal.rgb.shape, (120, 160, 3))
            self.assertEqual(normal.rgb.dtype, np.uint8)
            self.assertEqual(normal.depth_m.dtype, np.float32)
            self.assertEqual(normal.view.shape, (4, 4))
            self.assertEqual(normal.projection.shape, (4, 4))
            self.assertTrue(np.isfinite(normal.depth_m).all())
            self.assertGreaterEqual(float(normal.depth_m.min()), camera.near)
            self.assertLessEqual(float(normal.depth_m.max()), camera.far + .001)
            np.testing.assert_array_equal(normal.rgb, annotated.rgb)
            np.testing.assert_array_equal(normal.depth_m, annotated.depth_m)
            ids = annotated.segmentation.astype(np.int64) & 0xFFFFFF
            self.assertTrue(set(sim.object_ids).issubset(set(np.unique(ids))))
            self.assertEqual(len(sim.ground_truth_poses()), 3)
        self.assertIsNone(sim.client_id)
        with self.assertRaises(RuntimeError):
            sim.capture()

    def test_explicit_client_ids_keep_parallel_scenes_isolated(self):
        empty = sample_scene(1, min_objects=0, max_objects=0)
        with SimScene(demo_scene()) as populated, SimScene(empty) as blank:
            self.assertNotEqual(populated.client_id, blank.client_id)
            self.assertEqual(len(populated.object_ids), 3)
            self.assertEqual(len(blank.object_ids), 0)
            self.assertEqual(p.getNumBodies(physicsClientId=populated.client_id), 9)
            self.assertEqual(p.getNumBodies(physicsClientId=blank.client_id), 6)
            self.assertTrue(populated.settle())


if __name__ == '__main__':
    unittest.main()
