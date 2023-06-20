import pytest
from vpr.data.datasets import GardensPointWalking
from vpr.vpr_techniques import switchCNN
import config
import numpy as np
from glob import glob

dataset = GardensPointWalking
method = switchCNN
config.batch_size = 10

def test_contains_name():
    assert type(method.NAME) == type("abc")

def test_query_desc_shape():
    Q = dataset.get_query_paths()
    q_desc = method.compute_query_desc(Q[:config.batch_size - 1])
    assert len(q_desc[1]) == config.batch_size - 1
    assert len(q_desc[0]) == config.batch_size - 1

def test_query_desc_shape2():
    Q = dataset.get_query_paths()
    q_desc = method.compute_query_desc(Q[:config.batch_size + 1])
    assert len(q_desc[1]) == config.batch_size + 1
    assert len(q_desc[0]) == config.batch_size + 1
def test_map_desc_shape():
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:config.batch_size - 1])
    assert len(m_desc) == 5
    assert len(m_desc[0]) == config.batch_size - 1
    assert len(m_desc[1]) == config.batch_size - 1
    assert len(m_desc[2]) == config.batch_size - 1
    assert len(m_desc[3]) == config.batch_size - 1
    assert len(m_desc[4]) == config.batch_size - 1

def test_map_desc_shape2():
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:config.batch_size + 1])
    assert len(m_desc) == 5
    assert len(m_desc[0]) == config.batch_size + 1
    assert len(m_desc[1]) == config.batch_size + 1
    assert len(m_desc[2]) == config.batch_size + 1
    assert len(m_desc[3]) == config.batch_size + 1
    assert len(m_desc[4]) == config.batch_size + 1

def test_query_desc_type():
    Q = dataset.get_query_paths()
    q_desc = method.compute_query_desc(Q[:1])
    assert q_desc[0][0].dtype == np.float32

def test_map_desc_shape():
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:1])
    assert m_desc[0].dtype == np.float32
    assert m_desc[1].dtype == np.float32
    assert m_desc[2].dtype == np.float32
    assert m_desc[3].dtype == np.float32
    assert m_desc[4].dtype == np.float32






def test_matching_method():
    Q = dataset.get_query_paths()
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:10])
    q_desc = method.compute_query_desc(Q[:3])
    S = method.matching_method(q_desc, m_desc)
    assert S.shape[0] == 3
    assert S.shape[1] == 10
    assert S.max() < 1.001
    assert S.min() > 0.

def test_retrieval():
    Q = dataset.get_query_paths()
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:50])
    pr = method.PlaceRecognition(m_desc)
    I, S = pr.perform_vpr(Q[0])
    assert S.dtype == np.float32
    assert I.dtype == np.int

def test_retrieval1():
    Q = dataset.get_query_paths()
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:50])
    pr = method.PlaceRecognition(m_desc)
    I, S = pr.perform_vpr(Q[:10])
    assert S.dtype == np.float32
    assert I.dtype == np.int
    assert len(S) == 10
    assert S.max() <= 1.0
    assert S.min() >= 0.0
