"""Scene files must preserve valid, immutable sensor and sampling contracts."""
from dataclasses import replace
import unittest

import numpy as np

from robosort.types import CameraConfig, ObjectSpec, SceneSpec


class ContractTests(unittest.TestCase):
    def test_camera_rejects_degenerate_or_nonfinite_calibration(self):
        for options in ({'width': 100.5}, {'far': np.inf}, {'eye': (0., 0., 0.), 'target': (0., 0., 0.)},
                        {'up': (0., 0., -1.)}, {'target': (0., np.nan, 0.)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CameraConfig(**options)

    def test_fractional_seed_cannot_silently_change_a_saved_scene(self):
        scene = SceneSpec('one', 42, (ObjectSpec('cube', (.5, .1)),))
        data = scene.to_dict(); data['seed'] = 42.9
        with self.assertRaisesRegex(ValueError, 'integer'):
            SceneSpec.from_dict(data)

    def test_frozen_scene_copies_mutable_input_sequences(self):
        xy = [.5, .1]; objects = [ObjectSpec('cube', xy)]
        up = [0., 1., 0.]
        scene = SceneSpec('one', 42, objects, CameraConfig(up=up))
        xy[0] = 99.; objects.clear(); up[1] = 0.
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.objects[0].xy, (.5, .1))
        self.assertEqual(scene.camera.up, (0., 1., 0.))


if __name__ == '__main__':
    unittest.main()
