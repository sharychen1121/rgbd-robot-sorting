import math
from pathlib import Path
import unittest

import numpy as np

from robosort.perception import HSVPredictor, UltralyticsPredictor, localize, unproject_pixels
from robosort.types import CameraConfig, Frame, Prediction, DIMENSIONS, TABLE_Z


def frame(width=160, height=120):
    camera=CameraConfig(width=width,height=height)
    f=1/math.tan(math.radians(camera.fov)/2)
    near,far=camera.near,camera.far
    projection=np.array([[f/(width/height),0,0,0],[0,f,0,0],
                         [0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]],float)
    view=np.eye(4);view[2,3]=-2
    return Frame(np.zeros((height,width,3),np.uint8),np.ones((height,width),np.float32),view,projection,camera)


class Tensor:
    def __init__(self,array): self.array=np.asarray(array)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class FakeModel:
    def predict(self,**kwargs):
        self.input=kwargs
        class Boxes: pass
        class Masks: pass
        class Result: pass
        r=Result();r.boxes=Boxes();r.masks=Masks()
        r.boxes.xyxy=Tensor([[20,20,50,50],[60,20,80,40]])
        r.boxes.conf=Tensor([.9,.95]);r.boxes.cls=Tensor([0,1]);r.names={0:'cube',1:'person'}
        mask=np.zeros((2,120,160));mask[0,20:50,20:50]=1
        r.masks.data=Tensor(mask)
        return [r]


class PerceptionTests(unittest.TestCase):
    def test_unprojection_axis_depth_and_pixel_centres(self):
        image=frame()
        point=unproject_pixels(image,np.array([79.5]),np.array([59.5]),np.array([1.]))[0]
        np.testing.assert_allclose(point,[0,0,1],atol=1e-7)
        off=unproject_pixels(image,np.array([119.5]),np.array([59.5]),np.array([1.]))[0]
        self.assertAlmostEqual(off[0],.5/image.projection[0,0],places=7)
        self.assertAlmostEqual(off[2],1.,places=7)

    def test_localization_uses_predictions_not_gt_segmentation(self):
        image=frame();mask=np.zeros((120,160),bool);mask[45:75,65:95]=True
        prediction=Prediction('box',.8,(65,45,95,75),mask)
        image.segmentation=np.full((120,160),-1)
        a=localize(prediction,image)
        image.segmentation=np.full((120,160),999)
        b=localize(prediction,image)
        self.assertTrue(a.valid)
        np.testing.assert_allclose(a.point_world,b.point_world)
        np.testing.assert_allclose(a.point_world,[0,0,1],atol=1e-6)

    def test_missing_mask_and_invalid_depth_are_rejected(self):
        image=frame();pred=Prediction('cube',.9,(20,20,40,40))
        self.assertEqual(localize(pred,image).reason,'missing_or_misaligned_mask')
        image.depth_m[:]=np.nan
        self.assertFalse(localize(pred,image,representation='box').valid)

    def test_offscreen_box_cannot_wrap_numpy_negative_indices(self):
        image = frame()
        for box in ((-30, -30, -2, -2), (170, 130, 190, 150), (40, 40, 20, 20)):
            pred = Prediction('cube', .9, box)
            location = localize(pred, image, representation='box')
            self.assertFalse(location.valid)
            self.assertEqual(location.depth_pixels, 0)

    def test_probability_mask_and_invalid_mask_are_handled_explicitly(self):
        image = frame()
        mask = np.zeros((120, 160), float); mask[45:75, 65:95] = .9
        pred = Prediction('box', .9, (65, 45, 95, 75), mask)
        self.assertTrue(localize(pred, image).valid)
        pred.mask[0, 0] = np.nan
        self.assertEqual(localize(pred, image).reason, 'invalid_mask')

    def test_invalid_camera_matrices_and_pixel_coordinates_cannot_yield_valid_points(self):
        image = frame()
        with self.assertRaisesRegex(ValueError, 'pixel coordinates'):
            unproject_pixels(image, np.array([np.nan]), np.array([1]), np.array([1]))
        image.view[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'invertible'):
            unproject_pixels(image, np.array([1]), np.array([1]), np.array([1]))
        image = frame(); image.projection = np.eye(4)
        with self.assertRaisesRegex(ValueError, 'perspective'):
            localize(Prediction('cube', .9, (20, 20, 40, 40)), image, representation='box')

    def test_same_model_prediction_adapter_preserves_frame_rgb(self):
        image=frame();image.rgb[0,0]=[255,0,0];model=FakeModel()
        result=UltralyticsPredictor('unused.pt',model=model).predict(image)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].class_name,'cube')
        np.testing.assert_array_equal(model.input['source'][0,0],[0,0,255])
        np.testing.assert_array_equal(image.rgb[0,0],[255,0,0])
        self.assertEqual(result[0].mask.shape,(120,160))

    def test_hsv_reference_does_not_read_gt(self):
        image=frame();image.rgb[30:55,45:70]=[230,20,20]
        image.segmentation=np.zeros((120,160),np.int32)
        output=HSVPredictor(min_area=30).predict(image)
        self.assertEqual([x.class_name for x in output],['cube'])
        self.assertGreater(output[0].mask.sum(),400)

    def test_missing_checkpoint_fails_before_network_import(self):
        with self.assertRaises(FileNotFoundError):
            UltralyticsPredictor(Path('definitely-no-trained-model.pt')).load()

    def test_renderer_depth_reconstructs_visible_top_in_diagnostic(self):
        # This test uses truth ONLY to define a diagnostic input region. The
        # deployed predictor and localizer never access simulator segmentation.
        try:
            from robosort.sim import SimScene, sample_scene
            import pybullet as p
        except ImportError:
            self.skipTest('Simulation unavailable')
        spec=sample_scene(3291,camera=CameraConfig(width=320,height=240))
        with SimScene(spec,gui=False) as sim:
            image=sim.capture(include_truth=True)
            ids=image.segmentation.astype(np.int64)&0xFFFFFF
            for body,name in sim.truth.items():
                mask=ids==body
                rows,cols=np.nonzero(mask)
                self.assertGreater(len(rows),10)
                pred=Prediction(name,1.,(float(cols.min()),float(rows.min()),float(cols.max()+1),float(rows.max()+1)),mask)
                location=localize(pred,image)
                self.assertTrue(location.valid,location.reason)
                position,_=p.getBasePositionAndOrientation(body,physicsClientId=sim.client_id)
                self.assertLess(math.dist(location.point_world[:2],position[:2]),.012)
                self.assertAlmostEqual(location.point_world[2],TABLE_Z+DIMENSIONS[name][2],delta=.004)


if __name__=='__main__': unittest.main()
