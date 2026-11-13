"""Regression tests for experiment denominators and immutable run storage."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from robosort.evaluation import (catalogue_fingerprint, load_catalogue, load_results,
    save_catalogue, save_results, summarize)
from robosort.types import ObjectOutcome, ObjectSpec, RunResult, SceneSpec


def result(method="mask", scene="scene-1", **kwargs):
    values=dict(scene_id=scene,method=method,status="stopped",reason="no_valid_targets",total_objects=3,
                outcomes=[ObjectOutcome(3,"cube","cube","placed","contact_verified","A",True)],
                rejections=[{"stage":"localization","reason":"no_valid_depth"}],
                total_s=2.0,perception_s=None,localization_s=.2,motion_s=1.0,initialization_s=None)
    values.update(kwargs)
    return RunResult(**values)


class EvaluationTests(unittest.TestCase):
    def test_partial_placement_is_not_complete_sorting(self):
        report=summarize([result()])["methods"]["mask"]
        self.assertEqual(report["completed_trials"],0)
        self.assertEqual(report["at_least_one_correct_placement_trials"],1)
        self.assertAlmostEqual(report["object_correct_placement_rate"],1/3)
        self.assertEqual(report["rejection_events_by_stage"],{"localization":1})

    def test_misclassification_cannot_be_reported_as_correct_placement(self):
        outcome=ObjectOutcome(3,"box","cube","placed","contact_verified","C",True)
        report=summarize([result(outcomes=[outcome])])["methods"]["mask"]
        self.assertEqual(report["correct_placements"],0)
        self.assertEqual(report["mis_sorted_objects"],1)
        self.assertEqual(report["lift_only_objects"],0)

    def test_retry_does_not_double_count_one_object(self):
        first=ObjectOutcome(3,"cube","cube","failed","attach_failed",None,True)
        second=ObjectOutcome(3,"cube","cube","placed","contact_verified","A",True)
        report=summarize([result(outcomes=[first,second])])["methods"]["mask"]
        self.assertEqual(report["correct_placements"],1)
        self.assertEqual(report["grasp_attempts"],2)
        self.assertEqual(report["lifted_objects"],1)

    def test_lift_only_is_separate_from_placement(self):
        outcome=ObjectOutcome(3,"cube","cube","failed","placement_failed",None,True)
        report=summarize([result(outcomes=[outcome])])["methods"]["mask"]
        self.assertEqual(report["lift_only_objects"],1)
        self.assertEqual(report["correct_placements"],0)

    def test_missing_timing_remains_missing(self):
        report=summarize([result()])["methods"]["mask"]
        self.assertIsNone(report["mean_time_s"]["perception_s"])
        self.assertEqual(report["timing_measured_trials"]["perception_s"],0)
        self.assertEqual(report["mean_time_s"]["total_s"],2.0)

    def test_completed_status_requires_all_targets(self):
        with self.assertRaisesRegex(ValueError,"every object"):
            summarize([result(status="completed")])

    def test_mismatched_scene_sets_fail_paired_comparison(self):
        with self.assertRaisesRegex(ValueError,"different scene sets"):
            summarize([result("box","scene-1"),result("mask","scene-2")])

    def test_same_scene_id_requires_same_target_count_for_every_method(self):
        with self.assertRaisesRegex(ValueError, "target count"):
            summarize([result("box", total_objects=1), result("mask", total_objects=3)])

    def test_empty_scene_is_recorded_without_becoming_a_terminal_failure(self):
        report = summarize([result(total_objects=0, outcomes=[], status="completed", reason="no_objects")])["methods"]["mask"]
        self.assertEqual(report["empty_scene_trials"], 1)
        self.assertEqual(report["terminal_failures"], {})
        self.assertIsNone(report["full_success_rate"])

    def test_one_physical_body_cannot_change_truth_class_between_attempts(self):
        left = ObjectOutcome(3, "cube", "cube", "failed", "attach_failed")
        right = ObjectOutcome(3, "box", "box", "placed", "contact_verified", "C")
        with self.assertRaisesRegex(ValueError, "true class"):
            summarize([result(outcomes=[left, right])])

    def test_duplicate_method_scene_and_mixed_run_ids_fail(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            summarize([result(),result()])
        left=result("box").to_dict();left["run_id"]="a"
        right=result("mask").to_dict();right["run_id"]="b"
        with self.assertRaisesRegex(ValueError,"run IDs"):
            summarize([left,right])

    def test_results_roundtrip_does_not_silently_append(self):
        with tempfile.TemporaryDirectory() as td:
            path=save_results([result("box"),result("mask")],td,{"run_id":"test-run"})
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_results([result()],td,{"run_id":"other-run"})
            self.assertEqual(before,path.read_bytes())
            loaded=load_results(td)
            self.assertEqual({r["run_id"] for r in loaded},{"test-run"})
            self.assertIsNone(loaded[0]["perception_s"])
            self.assertTrue((Path(td)/"trial_results.csv").is_file())
            metadata=json.loads((Path(td)/"metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["record_count"],2)

    def test_result_fingerprint_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            path=save_results([result()],td,{"run_id":"one"})
            text=path.read_text(encoding="utf-8").replace('"total_s":2.0','"total_s":3.0')
            path.write_text(text,encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                load_results(td)

    def test_missing_hash_and_mismatched_metadata_cannot_disable_integrity_checks(self):
        with tempfile.TemporaryDirectory() as td:
            save_results([result()], td, {"run_id": "one"})
            metadata_path = Path(td)/"metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["results_sha256"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_results(td)
            save_results([result()], td, {"run_id": "one"}, overwrite=True)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")); metadata["record_count"] = 90
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not describe"):
                load_results(td)

    def test_incomplete_bundle_and_skips_are_visible(self):
        with tempfile.TemporaryDirectory() as td:
            raw=result().to_dict();raw["run_id"]="incomplete"
            (Path(td)/"results.jsonl").write_text(json.dumps(raw)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"metadata.json"):
                load_results(td)
        complete=result(scene="complete",status="completed",reason="all_objects_placed",total_objects=1)
        skipped=result(scene="not-run",status="skipped",reason="unavailable",outcomes=[])
        report=summarize([complete,skipped])["methods"]["mask"]
        self.assertEqual(report["evaluation_coverage"],.5)
        self.assertEqual(report["full_success_rate"],.5)
        self.assertEqual(report["skipped_trials"],1)

    def test_plots_render_measured_and_missing_times(self):
        from robosort.evaluation import plot_results
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            files=plot_results([result("box"),result("mask")],td)
            self.assertEqual(len(files),3)
            for path in files:
                with Image.open(path) as image:
                    self.assertGreater(image.width,600)
                    self.assertGreater(image.height,400)

    def test_catalogue_roundtrip_and_tamper_detection(self):
        scene=SceneSpec("test-1",42,(ObjectSpec("cube",(.4,.1)),))
        with tempfile.TemporaryDirectory() as td:
            path=save_catalogue([scene],Path(td)/"scenes.json",{"purpose":"test"})
            self.assertEqual(catalogue_fingerprint(load_catalogue(path)),catalogue_fingerprint([scene]))
            with self.assertRaises(FileExistsError):
                save_catalogue([scene],path)
            data=json.loads(path.read_text(encoding="utf-8"));data["scenes"][0]["seed"]=99
            path.write_text(json.dumps(data),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                load_catalogue(path)


if __name__ == "__main__":
    unittest.main()
