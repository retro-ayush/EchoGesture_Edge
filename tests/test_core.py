import unittest
import numpy as np
try:
    import torch
except ImportError:  # runtime-only install (requirements.txt)
    torch = None
from echogesture.features import frame_features, encode_sequence, augment, BASE_DIM, FEATURE_DIM, COORD_DIM, coord_mask
if torch is not None:
    from echogesture.model import TinyTemporalNet


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.zeros((33,2),np.float32);self.pose[11]=[.4,.4];self.pose[12]=[.6,.4]
        self.vis=np.ones(33)
        self.hand=np.stack([np.linspace(0,.1,21),np.linspace(0,.2,21),np.linspace(0,.03,21)],axis=1)
        self.image=np.tile([.5,.5],(21,1))

    def test_translation_and_scale_invariance(self):
        a=frame_features(self.pose,self.vis,{"Left":(self.hand,self.image)})
        b=frame_features(self.pose*2+.1,self.vis,{"Left":(self.hand*3+7,self.image*2+.1)})
        np.testing.assert_allclose(a,b,atol=1e-4)

    def test_missing_frames_cannot_create_motion(self):
        a=frame_features(self.pose,self.vis,{"Left":(self.hand,self.image)})
        z=np.zeros(BASE_DIM,np.float32)
        x=encode_sequence(np.stack([a,z,a]),frames=3)
        np.testing.assert_array_equal(x[COORD_DIM:2*COORD_DIM],0)
        self.assertEqual(x.shape,(FEATURE_DIM,3))

    def test_hand_location_preserved(self):
        a=frame_features(self.pose,self.vis,{"Left":(self.hand,self.image)})
        b=frame_features(self.pose,self.vis,{"Left":(self.hand,self.image+.1)})
        np.testing.assert_allclose(a[26:89],b[26:89])
        self.assertFalse(np.allclose(a[152:154],b[152:154]))

    def test_invalid_input(self):
        for x in [np.zeros((0,BASE_DIM)),np.full((3,BASE_DIM),np.nan)]:
            with self.assertRaises(ValueError):encode_sequence(x)

    @unittest.skipIf(torch is None, "training test needs requirements-train.txt")
    def test_augmentation_and_backprop(self):
        torch.set_num_threads(1)
        base=np.stack([frame_features(self.pose,self.vis,{"Left":(self.hand,self.image)})]*64)
        x=encode_sequence(augment(base,np.random.default_rng(3)))
        net=TinyTemporalNet(5)
        logits=net(torch.tensor(np.stack([x,x])))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]));loss.backward()
        self.assertEqual(tuple(logits.shape),(2,5))
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(sum(v.numel() for v in net.parameters()),200000)


if __name__=="__main__":unittest.main()
