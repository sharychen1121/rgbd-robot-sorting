"""Run the explicitly small CPU learning integration check; not a benchmark."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from robosort.dataset import generate_dataset,validate_dataset
from robosort.training import train_model,validate_model,package_versions
from robosort.perception import UltralyticsPredictor,localize,draw_predictions
from robosort.sim import SimScene,sample_scene
from robosort.types import CameraConfig
import cv2


def save(report):
    (ROOT/'examples'/'verification'/'learning-smoke.json').write_text(json.dumps(portable(report),indent=2),encoding='utf-8')


def portable(value):
    """Public verification files use repository-relative artifact references."""
    if isinstance(value, dict): return {key:portable(item) for key,item in value.items()}
    if isinstance(value, list): return [portable(item) for item in value]
    if isinstance(value, str):
        try:
            path=Path(value)
            if path.is_absolute() and path.is_relative_to(ROOT): return path.relative_to(ROOT).as_posix()
        except (ValueError,OSError): pass
        return value.replace(str(ROOT)+'\\','').replace(str(ROOT)+'/', '')
    return value


def write_summary(report):
    metrics=report['heldout_image_metrics']['metrics']
    unseen=report['unseen_scene']
    text=(
        '# Learning pipeline integration check\n\n'
        'This is a small functional check, not a finished accuracy study or proof that mask-based sorting is better. '
        'It verifies data generation, five CPU fine-tuning epochs of YOLO26n-seg, explicit held-out test validation, '
        'checkpoint reload, predicted masks, and mask/box RGB-D localization.\n\n'
        'Dataset: 90 independently seeded 320 x 240 scenes, randomized colors and lighting; '
        '62 train / 13 validation / 15 test images, including 14 explicitly empty scenes. '
        'Training: 5 epochs, image size 320, batch 2, CPU, seed 241114.\n\n'
        f"Held-out test box mAP50: {metrics['metrics/mAP50(B)']:.4f}; "
        f"mask mAP50: {metrics['metrics/mAP50(M)']:.4f}; "
        f"mask mAP50-95: {metrics['metrics/mAP50-95(M)']:.4f}. "
        'These low pilot metrics indicate that the model needs substantially more training and evaluation.\n\n'
        f"The unseen image uses scene seed 241115, with {len(unseen['predictions'])} candidates at diagnostic confidence 0.10 "
        f"and {unseen.get('default_confidence_prediction_count', 'not measured')} at the default confidence 0.25. "
        'The diagnostic candidates include duplicate detections and incorrect class predictions; a valid 3D point only '
        'means that a predicted region can be localized, not that its category or grasp is correct. '
        'The runtime camera frame contains no simulator segmentation buffer.\n\n'
        'Full metrics, checkpoint SHA256, model versions, configuration, and candidate details are in learning-smoke.json. '
        'The checkpoint is a pilot artifact and should not be presented as the final project model.\n'
    )
    (ROOT/'examples'/'verification'/'learning-smoke.md').write_text(text,encoding='utf-8')


def main():
    started=time.perf_counter()
    report={'purpose':'Small CPU end-to-end integration check, not a formal accuracy or robot-sorting benchmark',
            'status':'running','versions':package_versions()}
    save(report)
    try:
        data=ROOT/'datasets'/'pilot-v1'
        if not (data/'manifest.json').is_file():
            generate_dataset(data,count=90,seed=241114,width=320,height=240,
                             color_randomization=True,lighting_randomization=True,negative_fraction=.15)
        audit=validate_dataset(data)
        report['dataset_validation']=audit
        if not audit['valid']: raise RuntimeError(audit['errors'])
        report['dataset_manifest_sha256']=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
        save(report)
        checkpoint=ROOT/'artifacts'/'pilot-yolo26'/'best.pt'
        if not checkpoint.is_file():
            checkpoint=train_model(data/'segment.yaml',model='yolo26n-seg.pt',
                                  run_dir=ROOT/'artifacts'/'pilot-yolo26',epochs=5,imgsz=320,
                                  batch=2,device='cpu',seed=241114,allow_download=True)
        report['checkpoint']={'path':str(checkpoint.relative_to(ROOT)),
                              'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              'bytes':checkpoint.stat().st_size}
        save(report)
        valdir=ROOT/'artifacts'/'pilot-yolo26'/'heldout-test'
        if (valdir/'metrics.json').is_file():
            metrics=json.loads((valdir/'metrics.json').read_text())
        else:
            metrics=validate_model(checkpoint,data/'segment.yaml',valdir,split='test',imgsz=320,batch=2,device='cpu')
        report['heldout_image_metrics']=metrics
        predictor=UltralyticsPredictor(checkpoint,conf=.10,device='cpu')
        scene=sample_scene(241115,scene_id='learning-smoke-unseen',camera=CameraConfig(width=320,height=240),random_colors=True)
        with SimScene(scene,gui=False) as sim:
            frame=sim.capture(include_truth=False)
            predictions=predictor.predict(frame)
            default_predictions=UltralyticsPredictor(checkpoint,conf=.25,device='cpu',model=predictor.model).predict(frame)
            report['unseen_scene']={'scene':scene.to_dict(),'confidence_threshold':.10,
                'runtime_segmentation_present':frame.segmentation is not None,
                'default_confidence_threshold':.25,'default_confidence_prediction_count':len(default_predictions),'predictions':[
                {'class_name':p.class_name,'confidence':p.confidence,'has_mask':p.mask is not None,
                 'mask_localization':localize(p,frame,'mask').point_world,
                 'mask_reason':localize(p,frame,'mask').reason,
                 'box_localization':localize(p,frame,'box').point_world} for p in predictions]}
            overlay=draw_predictions(frame,predictions)
            cv2.imencode('.png',cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))[1].tofile(str(ROOT/'examples'/'verification'/'learning-smoke-overlay.png'))
        report.update(status='completed',elapsed_seconds=time.perf_counter()-started)
        report['configuration']={'scene_count':90,'width':320,'height':240,'color_randomization':True,
            'lighting_randomization':True,'negative_fraction_requested':.15,'epochs':5,'imgsz':320,'batch':2,
            'device':'cpu','seed':241114,'inference_seed':241115}
        initial=ROOT/'yolo26n-seg.pt'
        if initial.is_file():
            report['pretrained_checkpoint']={'source':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt',
                'sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'bytes':initial.stat().st_size}
        report['limitations']=['Five-epoch pilot is under-trained; not a formal benchmark.',
            'Diagnostic confidence 0.10 exposes low-confidence candidates, including duplicates and incorrect classes.',
            'Successful RGB-D localization does not imply correct classification or completed sorting.']
        save(report)
        write_summary(report)
        print(json.dumps({'status':report['status'],'checkpoint':report['checkpoint'],'seconds':report['elapsed_seconds']},indent=2))
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started)
        save(report)
        raise


if __name__=='__main__':main()
