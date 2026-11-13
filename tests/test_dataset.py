import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from robosort.dataset import _split_indices, _validate_output, generate_dataset, labels_from_segmentation, sha256_file, validate_dataset
from robosort.types import CLASS_IDS, CameraConfig, ObjectSpec, SceneSpec


def recorded_dataset(root):
    """Small fixed byte-level fixture; no renderer/model/network is required."""
    records = []
    for index, split in enumerate(('train', 'val', 'test')):
        scene_id = f'scene_{index}'
        scene = SceneSpec(scene_id, index, (ObjectSpec('cube', (.5, .1)),), CameraConfig(width=48, height=32), split=split)
        for task in ('segment', 'detect'):
            images = root/task/'images'/split; images.mkdir(parents=True)
            labels = root/task/'labels'/split; labels.mkdir(parents=True)
            cv2.imwrite(str(images/(scene_id+'.png')), np.zeros((32, 48, 3), np.uint8))
            line = '0 .25 .25 .75 .25 .75 .75 .25 .75\n' if task == 'segment' else '0 .5 .5 .5 .5\n'
            (labels/(scene_id+'.txt')).write_text(line, encoding='utf-8')
        records.append(dict(scene=scene.to_dict(), negative=False, instances=1,
                            image=f'segment/images/{split}/{scene_id}.png',
                            image_sha256=sha256_file(root/'segment'/'images'/split/(scene_id+'.png')),
                            segment_label_sha256=sha256_file(root/'segment'/'labels'/split/(scene_id+'.txt')),
                            detect_label_sha256=sha256_file(root/'detect'/'labels'/split/(scene_id+'.txt'))))
    for task in ('segment', 'detect'):
        config = {split: f'{task}/images/{split}' for split in ('train', 'val', 'test')}
        config['names'] = {value: key for key, value in CLASS_IDS.items()}
        (root/f'{task}.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    manifest = dict(owner='robosort', schema_version=2, scene_count=3, negative_count=0,
                    split_counts=dict(train=1, val=1, test=1), records=records)
    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


class DatasetTests(unittest.TestCase):
    def test_body_ids_not_links_or_background_become_labels(self):
        seg=np.full((50,80),-1,np.int64)
        seg[10:30,20:50]=7+(2<<24)
        seg[2:6,2:6]=9  # non-target must not become a label
        polygon,box,audit=labels_from_segmentation(seg,{7:'box'})
        self.assertEqual(len(polygon),1)
        self.assertTrue(polygon[0].startswith('2 '))
        self.assertEqual(len(box[0].split()),5)
        self.assertTrue(audit[0]['visible'])

    def test_negative_images_have_valid_empty_labels(self):
        polygon,box,audit=labels_from_segmentation(np.zeros((20,20),np.int32),{})
        self.assertEqual((polygon,box,audit),([],[],[]))

    def test_split_is_deterministic_and_contains_every_scene_once(self):
        a=_split_indices(30,42);b=_split_indices(30,42)
        self.assertEqual(a,b)
        self.assertEqual(set(a),set(range(30)))
        self.assertEqual(set(a.values()),{'train','val','test'})

    def test_existing_unrecognized_directory_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'precious.txt').write_text('keep')
            with self.assertRaises(FileExistsError): _validate_output(root,False)
            with self.assertRaises(ValueError): _validate_output(root,True)
            self.assertEqual((root/'precious.txt').read_text(),'keep')

    def test_validator_flags_malformed_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for task in ('segment','detect'):
                for split in ('train','val','test'):
                    (root/task/'images'/split).mkdir(parents=True)
                    (root/task/'labels'/split).mkdir(parents=True)
                cv2.imwrite(str(root/task/'images'/'train'/'scene_0.png'),np.zeros((20,20,3),np.uint8))
                (root/task/'labels'/'train'/'scene_0.txt').write_text('0 .2 .3 .4 .4\n')
            (root/'manifest.json').write_text(json.dumps({'scene_count':1,'negative_count':0}))
            report=validate_dataset(root)
            self.assertFalse(report['valid'])
            self.assertTrue(any('Malformed' in x for x in report['errors']))

    def test_tiny_simulation_dataset_round_trip(self):
        try:
            from robosort.sim import SimScene
        except ImportError:
            self.skipTest('Simulation module not yet installed')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'sample'
            manifest_path=generate_dataset(output,count=6,seed=11,width=160,height=120,negative_fraction=.5)
            manifest=json.loads(manifest_path.read_text())
            self.assertEqual(manifest['negative_count'],3)
            report=validate_dataset(output)
            self.assertTrue(report['valid'],report['errors'])
            self.assertEqual(sum(x['images'] for x in report['splits']['segment'].values()),6)
            # Exercise the actual generated YAML after its original path has
            # disappeared. Only font download is disabled; path resolution is
            # the installed Ultralytics implementation used by training.
            moved = Path(tmp)/'relocated'; output.rename(moved)
            from ultralytics.data.utils import check_det_dataset
            with patch('ultralytics.data.utils.check_font'):
                for task in ('segment', 'detect'):
                    config = check_det_dataset(str((moved/f'{task}.yaml').resolve()), autodownload=False)
                    for split in ('train', 'val', 'test'):
                        self.assertEqual(Path(config[split]), moved/task/'images'/split)
            self.assertTrue(validate_dataset(moved)['valid'])
            self.assertEqual(manifest['segment_yaml'], 'segment.yaml')

    def test_valid_but_changed_detection_label_fails_hash_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); recorded_dataset(root)
            self.assertTrue(validate_dataset(root)['valid'])
            label = root/'detect'/'labels'/'train'/'scene_0.txt'
            label.write_text('0 .4 .4 .2 .2\n', encoding='utf-8')
            report = validate_dataset(root)
            self.assertFalse(report['valid'])
            self.assertTrue(any('SHA256' in error and 'detect' in error for error in report['errors']))

    def test_split_moves_and_orphan_labels_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); recorded_dataset(root)
            for task in ('segment', 'detect'):
                for kind, extension in (('images', '.png'), ('labels', '.txt')):
                    (root/task/kind/'train'/('scene_0'+extension)).rename(root/task/kind/'val'/('scene_0'+extension))
            orphan = root/'detect'/'labels'/'test'/'orphan.txt'; orphan.write_text('')
            report = validate_dataset(root)
            self.assertFalse(report['valid'])
            self.assertTrue(any('scene IDs differ' in error for error in report['errors']))
            self.assertTrue(any('Orphan label' in error for error in report['errors']))


if __name__=='__main__': unittest.main()
