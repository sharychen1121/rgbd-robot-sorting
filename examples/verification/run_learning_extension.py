"""Continue the pilot with a separate fresh optimizer: 5 epochs, then 25.

This is deliberately not represented as an uninterrupted 30-epoch run.
"""
from pathlib import Path
import csv
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from robosort.dataset import sha256_file,validate_dataset
from robosort.training import train_model,validate_model,package_versions
from robosort.perception import UltralyticsPredictor,localize,draw_predictions
from robosort.sim import SimScene,sample_scene
from robosort.types import CameraConfig
from examples.verification.run_learning_smoke import portable
import cv2


def save(report):
    target=ROOT/'examples'/'verification'/'learning-extension.json'
    target.write_text(json.dumps(portable(report),indent=2),encoding='utf-8')


def main():
    started=time.perf_counter()
    initial=ROOT/'artifacts'/'pilot-yolo26'/'best.pt'
    data=ROOT/'datasets'/'pilot-v1'
    run=ROOT/'artifacts'/'yolo26-smoke-30'
    report={'status':'running','purpose':'Two-stage small-data learning verification; not a formal method comparison',
            'versions':package_versions(),
            'training_lineage':{'stage1_epochs':5,'stage1_weights':str(initial),'stage1_sha256':sha256_file(initial),
                'stage2_requested_max_epochs':25,'stage2_optimizer':'fresh Ultralytics optimizer and learning-rate schedule',
                'equivalent_to_uninterrupted_30_epochs':False},
            'configuration':{'data':str(data/'segment.yaml'),'imgsz':320,'batch':2,'device':'cpu','seed':241114}}
    save(report)
    try:
        audit=validate_dataset(data)
        report['dataset_validation']=audit
        if not audit['valid']:raise RuntimeError(audit['errors'])
        checkpoint=run/'best.pt'
        if not checkpoint.is_file():
            checkpoint=train_model(data/'segment.yaml',model=initial,run_dir=run,
                epochs=25,imgsz=320,batch=2,device='cpu',seed=241114,allow_download=False)
        report['checkpoint']={'path':str(checkpoint),'sha256':sha256_file(checkpoint),'bytes':checkpoint.stat().st_size}
        history=run/'fit'/'results.csv'
        if history.is_file():
            with history.open(newline='',encoding='utf-8') as file:
                rows=list(csv.DictReader(file))
            report['training_lineage']['stage2_completed_epochs']=len(rows)
            report['training_lineage']['stage2_requested_max_epochs']=25
        save(report)
        valdir=run/'heldout-test'
        if (valdir/'metrics.json').is_file():
            metrics=json.loads((valdir/'metrics.json').read_text())
        else:
            metrics=validate_model(checkpoint,data/'segment.yaml',valdir,split='test',imgsz=320,batch=2,device='cpu')
        report['heldout_image_metrics']=metrics
        predictor=UltralyticsPredictor(checkpoint,conf=.25,device='cpu')
        report['fresh_scenes']=[]
        for seed in (241116,241117,241118,241119):
            negative=seed==241119
            scene=sample_scene(seed,scene_id=f'learning-extension-unseen-{seed}',
                camera=CameraConfig(width=320,height=240),random_colors=True,
                min_objects=0 if negative else 3,max_objects=0 if negative else 3)
            with SimScene(scene,gui=False) as sim:
                frame=sim.capture(include_truth=False)
                predictions=predictor.predict(frame)
                observations=[]
                for prediction in predictions:
                    point=localize(prediction,frame,'mask')
                    observations.append({'class_name':prediction.class_name,'confidence':prediction.confidence,
                        'has_mask':prediction.mask is not None,'mask_pixels':int(prediction.mask.sum()) if prediction.mask is not None else 0,
                        'mask_localization':point.point_world,'localization_reason':point.reason,
                        'box_localization':localize(prediction,frame,'box').point_world})
                report['fresh_scenes'].append({'seed':seed,'scene':scene.to_dict(),
                    'negative_scene':negative,'runtime_segmentation_present':frame.segmentation is not None,
                    'confidence_threshold':.25,'predictions':observations})
                overlay=draw_predictions(frame,predictions)
                cv2.imencode('.png',cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))[1].tofile(
                    str(ROOT/'examples'/'verification'/f'learning-extension-{seed}.png'))
        report.update(status='completed',elapsed_seconds=time.perf_counter()-started)
        report['limitations']=['90-scene pilot only; no formal statistical method comparison.',
            'Two-stage training resets the optimizer and learning-rate schedule at the stage boundary; actual epochs are recorded separately.',
            'The 15-image test partition was already inspected after stage 1; this is a retained-image retest, not a pristine final benchmark.',
            'Image metrics and valid localizations do not prove completed mechanical sorting.']
        save(report)
        values=metrics['metrics']
        (ROOT/'examples'/'verification'/'learning-extension.md').write_text(
            '# Two-stage learning verification\n\n'
            'YOLO26n-seg was first fine-tuned for 5 epochs and its best checkpoint initialized a separate run of up to 25 epochs. '
            f"The second stage completed {report['training_lineage'].get('stage2_completed_epochs','unknown')} epochs. "
            'The second stage starts a fresh optimizer and schedule; it is not an uninterrupted 30-epoch experiment. '
            'Both stages use the same 90-scene small pilot dataset and validation split.\n\n'
            f"Retained 15-image test partition retest: box mAP50 {values['metrics/mAP50(B)']:.4f}, mask mAP50 {values['metrics/mAP50(M)']:.4f}, "
            f"mask mAP50-95 {values['metrics/mAP50-95(M)']:.4f}. These are pilot image metrics, not final comparative claims.\n\n"
            'This test partition was previously inspected after stage 1; a formal study still needs a new frozen test set. '
            'Fresh seeds 241116, 241117, and 241118 contain three objects; seed 241119 is a negative scene. '
            'All use confidence 0.25 and camera frames with no simulator segmentation. '
            'Predicted masks and box/mask RGB-D positions are recorded in learning-extension.json.\n',encoding='utf-8')
        print(json.dumps(portable({'status':'completed','checkpoint':report['checkpoint'],'metrics':values,
            'fresh_prediction_counts':[len(s['predictions']) for s in report['fresh_scenes']]}),indent=2))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started)
        save(report)
        raise


if __name__=='__main__':main()
