import unittest
import numpy as np
import torch
from data.utils import DataModule
from python.vpr.resnet.vpr import VPR


class TestResnetVPR(unittest.TestCase):

    def setUp(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dm = DataModule(dataset='GardensPointWalking', session_type='ms', gt_type='hard')
        self.vpr = VPR(device=device, batch_size=32)

    def test_map(self):
        m_desc = self.vpr.compute_map_features(self.dm.M[:10])

    def test_map2(self):
        m_desc = self.vpr.compute_map_features(self.dm.M[:10])
        assert 10 == m_desc.shape[0]
        assert m_desc.dtype == np.float32

    def test_query(self):
        q_desc = self.vpr.compute_query_desc(self.dm.Q[:10])
        assert q_desc.shape[0] == 10
        assert type(q_desc) == type(np.zeros((3,3)))

    def test_query2(self):
        q_desc = self.vpr.compute_query_desc(self.dm.Q[:10])
        assert q_desc.dtype == np.float32

    def test_sim_matrix(self):
        m_desc = self.vpr.compute_map_features(self.dm.M[:20])
        S = self.vpr.similarity_matrix(self.dm.Q[:10])
        assert S.shape[0] == 10
        assert S.shape[1] == 20
        assert S.dtype == np.float32


if __name__ == '__main__':
    unittest.main()
