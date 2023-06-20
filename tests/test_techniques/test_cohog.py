import pytest
from vpr.data.datasets import GardensPointWalking
from vpr.vpr_techniques import cohog
import config
import numpy as np
from glob import glob

dataset = GardensPointWalking
method = cohog
config.batch_size = 5


def test_contains_name():
    assert type(method.NAME) == type("abc")


def test_query_desc_shape():
    Q = dataset.get_query_paths()
    q_desc = method.compute_query_desc(Q[:config.batch_size - 1])
    assert len(q_desc) == config.batch_size - 1

def test_query_desc_shape2():
    Q = dataset.get_query_paths()
    q_desc = method.compute_query_desc(Q[:config.batch_size + 1])
    assert len(q_desc) == config.batch_size + 1
def test_map_desc_shape():
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:config.batch_size - 1])
    assert len(m_desc) == config.batch_size-1

def test_map_desc_shape2():
    M = dataset.get_map_paths()
    m_desc = method.compute_map_features(M[:config.batch_size + 1])
    assert len(m_desc) == config.batch_size + 1


def test_perform_vpr():
    query_img = glob(config.root_dir + '/tests/test_techniques/resources/test_query_img/*')[0]
    db_img = sorted(glob(config.root_dir + '/tests/test_techniques/resources/test_db_img/*'))
    m_desc = method.compute_map_features(db_img[:20])
    idx, score = method.perform_vpr(query_img, m_desc)
    print(idx, score)
    assert idx == 0
    assert isinstance(idx, int)
    assert isinstance(score, float)
    assert 0 <= score <= 1.


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
