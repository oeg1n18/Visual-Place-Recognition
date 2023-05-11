
import unittest
import cv2
import numpy as np

import data.utils as dataset

class TestDataModule(unittest.TestCase):
    def test_dm_setup(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')


    def test_Q(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert isinstance(dm.Q, list)

    def test_Q2(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert len(dm.Q) > 0

    def test_M(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert isinstance(dm.M, list)

    def test_M2(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert len(dm.M) > 0

    def test_gt(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert type(dm.GT) == type(np.zeros(shape=(3,3)))

    def test_gt2(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert dm.GT.shape[0] > 0
        assert dm.GT.shape[1] > 0

    def test_gt3(self):
        dm = dataset.DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        assert dm.GT.dtype == np.zeros(1, dtype=np.uint8).dtype



if __name__ == '__main__':
    unittest.main()
