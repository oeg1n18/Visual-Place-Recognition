
import unittest
import cv2
import numpy as np

import vpr.data.datasets.GardensPointWalking as gp

class TestStringMethods(unittest.TestCase):

    def test_query_type(self):
        q_paths = gp.get_query_paths(session_type='ms')
        assert isinstance(q_paths, list)

    def test_query_dtype(self):
        q_paths = gp.get_query_paths(session_type='ms')
        assert isinstance(q_paths[0], str)

    def test_query_read(self):
        q_paths = gp.get_query_paths(session_type='ms')
        im = cv2.imread(q_paths[0])
        assert isinstance(im, type(np.zeros(1)))

    def test_map_type(self):
        m_paths = gp.get_map_paths(session_type='ms')
        assert isinstance(m_paths, list)

    def test_map_dtype(self):
        m_paths = gp.get_map_paths(session_type='ms')
        assert isinstance(m_paths[0], str)

    def test_map_read(self):
        m_paths = gp.get_query_paths(session_type='ms')
        im = cv2.imread(m_paths[0])
        assert isinstance(im, type(np.zeros(1)))

    def test_gtmatrix_type(self):
        gtmatrix = gp.get_gtmatrix(session_type='ms', gt_type='hard')
        assert isinstance(gtmatrix, type(np.zeros(1)))

    def test_gtmatrix_dtype(self):
        gtmatrix = gp.get_gtmatrix(session_type='ms', gt_type='hard')
        assert gtmatrix.dtype == np.zeros(1, dtype=np.uint8).dtype

    def test_matrix_size(self):
        gtmatrix = gp.get_gtmatrix(session_type='ms', gt_type='hard')
        m_paths = gp.get_map_paths(session_type='ms')
        q_paths = gp.get_query_paths(session_type='ms')
        assert gtmatrix.shape == np.empty(shape=(len(m_paths), len(q_paths))).shape




if __name__ == '__main__':
    unittest.main()