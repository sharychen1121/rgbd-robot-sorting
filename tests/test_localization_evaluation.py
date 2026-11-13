import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robosort.localization_evaluation import _global_matches, evaluate_localization, save_localization_results
from robosort.types import CameraConfig, Frame, Localization, Prediction, SceneSpec


class LocalizationEvaluationTests(unittest.TestCase):
    def test_global_matching_avoids_greedy_duplicate_assignment(self):
        pairs = _global_matches(np.array([[.9, .8], [.85, 0.]]), .25)
        self.assertEqual({(i, j) for i, j, _ in pairs}, {(0, 1), (1, 0)})

    def test_truth_free_same_predictions_and_full_denominators(self):
        segmentation = np.full((32, 32), -1, np.int32)
        segmentation[2:8, 2:8] = 1; segmentation[12:18, 12:18] = 2
        image = Frame(np.zeros((32, 32, 3), np.uint8), np.ones((32, 32)), np.eye(4), np.eye(4), CameraConfig(width=32, height=32), segmentation)
        predictions = [Prediction('box', .9, (2, 2, 8, 8), np.ones((32, 32), bool)),
                       Prediction('cube', .8, (2, 2, 8, 8)),
                       Prediction('cylinder', .9, (12, 12, 18, 18))]
        calls = []
        class Predictor:
            def predict(self, frame):
                self.asserted_no_truth = frame.segmentation is None
                self.calls = getattr(self, 'calls', 0)+1
                return predictions
        class Scene:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def capture(self, include_truth=False):
                assert include_truth
                return image
            def ground_truth_poses(self):
                return [{'body_id': i+1, 'class_name': name, 'top_position': (float(i), 0., 1.)}
                        for i, name in enumerate(('cube', 'cylinder', 'box'))]
        def locate(prediction, frame, representation):
            self.assertIsNone(frame.segmentation)
            calls.append((id(prediction), id(frame), representation))
            if representation == 'mask' and prediction.mask is None:
                return Localization(prediction, None, reason='missing_mask')
            point = (.03, .04, 1.02) if prediction is predictions[0] else (1., 0., 1.)
            return Localization(prediction, point, depth_pixels=10)
        predictor = Predictor()
        with patch('robosort.localization_evaluation.SimScene', Scene), patch('robosort.localization_evaluation.localize', locate):
            report = evaluate_localization([SceneSpec('test', 42, ())], predictor)
        self.assertTrue(predictor.asserted_no_truth); self.assertEqual(predictor.calls, 1)
        summary = report['summary']
        self.assertEqual((summary['total_targets'], summary['matched_targets'], summary['missed_targets'], summary['false_positive_predictions']), (3, 2, 1, 1))
        self.assertEqual(summary['classification_correct_matches'], 1)
        self.assertEqual(summary['classification_accuracy_matched'], .5)
        self.assertAlmostEqual(summary['representations']['box']['valid_localization_coverage_all_targets'], 2/3)
        mask = summary['representations']['mask']
        self.assertAlmostEqual(mask['valid_localization_coverage_all_targets'], 1/3)
        self.assertAlmostEqual(mask['xy_error_m']['mean'], .05)
        self.assertAlmostEqual(mask['z_error_m']['p95'], .02)
        self.assertEqual(mask['error_3d_m']['count'], 1)
        self.assertEqual(len({frame_id for _, frame_id, _ in calls}), 1)
        self.assertEqual(report['scenes'][0]['missed'][0]['reason'], 'not_visible')
        with tempfile.TemporaryDirectory() as folder:
            path = save_localization_results(report, folder, {'checkpoint_sha256': 'test'})
            self.assertEqual(json.loads(path.read_text())['metadata']['checkpoint_sha256'], 'test')
            self.assertTrue((Path(folder)/'localization.csv').is_file())
            with self.assertRaises(FileExistsError):
                save_localization_results(report, folder)


if __name__ == '__main__': unittest.main()
