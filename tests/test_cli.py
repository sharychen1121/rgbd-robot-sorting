import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robosort.cli import main, parser
from robosort.evaluation import load_catalogue
from robosort.types import CameraConfig, SceneSpec


class CLITests(unittest.TestCase):
    def test_frozen_catalogue_from_cli_is_replayable(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder)/"scenes.json"
            self.assertEqual(main(["scenes", "--output", str(path), "--count", "2", "--seed", "1300"]), 0)
            scenes = load_catalogue(path)
            self.assertEqual(len(scenes), 2)
            self.assertEqual(scenes[0].seed, 1300)
            self.assertEqual(main(["scenes", "--output", str(path)]), 1)

    def test_missing_checkpoint_does_not_start_inference(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["demo", "--method", "yolo_mask", "--output", folder]), 1)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_gui_is_explicit_and_default_evaluation_is_paired_reference(self):
        args = parser().parse_args(["evaluate", "--output", "results/example"])
        self.assertFalse(args.gui)
        self.assertEqual(args.methods, ["oracle", "hsv"])

    def test_invalid_counts_and_budgets_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()), patch('robosort.cli._build_predictors') as build:
            for arguments in (["evaluate", "--count", "0"], ["evaluate", "--max-attempts", "0"],
                              ["demo", "--max-observations", "-1"], ["evaluate", "--conf", "nan"],
                              ["evaluate", "--methods", "hybrid", "--conf", ".8", "--hybrid-high", ".8"]):
                with self.subTest(arguments=arguments):
                    self.assertEqual(main([*arguments, "--output", folder]), 1)
            build.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_existing_file_output_is_reported_without_traceback_or_changes(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            output = Path(folder)/'precious.txt'; output.write_text('keep')
            self.assertEqual(main(['demo', '--method', 'hsv', '--output', str(output)]), 1)
            self.assertEqual(output.read_text(), 'keep')

    def test_prediction_output_records_scene_model_and_timing_provenance(self):
        class Scene:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def capture(self): return SimpleNamespace(rgb=np.zeros((16, 16, 3), np.uint8))
        predictor = SimpleNamespace(predict=lambda frame: [])
        model_info = {'yolo_mask': {'checkpoint_sha256': 'trained-model-hash', 'model_load_s': .2, 'warmup_s': .1}}
        spec = SceneSpec('frozen', 42, (), CameraConfig(width=16, height=16))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch('robosort.cli._catalogue', return_value=[spec]), \
             patch('robosort.cli._build_predictors', return_value=({'yolo_mask': predictor}, model_info)), \
             patch('robosort.cli._environment', return_value={'code_commit': 'test'}), \
             patch('robosort.sim.SimScene', Scene):
            self.assertEqual(main(['predict', '--weights', 'unused-by-stub.pt', '--output', folder]), 0)
            metadata = json.loads((Path(folder)/'metadata.json').read_text())
            self.assertEqual(metadata['models'], model_info)
            self.assertEqual(load_catalogue(Path(folder)/'scenes.json'), [spec])
            self.assertEqual(set(metadata['files_sha256']), {'predictions.png', 'coordinates.json', 'scenes.json'})
            for value in metadata['timing_s'].values():
                self.assertGreaterEqual(value, 0)


if __name__ == "__main__":
    unittest.main()
