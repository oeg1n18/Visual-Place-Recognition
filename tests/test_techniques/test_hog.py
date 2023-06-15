import pytest
from vpr.data.datasets import GardensPointWalking
from vpr.vpr_techniques import hog
import config
import numpy as np
from glob import glob

dataset = GardensPointWalking
method = hog
config.batch_size = 5


def test_contains_name():
    assert type(method.NAME) == type("abc")

def test_query_desc_shape():
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    q_desc = method.compute_query_desc(Q[:config.batch_size - 1])
    assert q_desc.shape[0] == config.batch_size - 1

def test_query_desc_shape2():
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    q_desc = method.compute_query_desc(Q[:config.batch_size + 1])
    assert q_desc.shape[0] == config.batch_size + 1
def test_map_desc_shape():
    M = dataset.get_map_paths(rootdir=config.root_dir)
    m_desc = method.compute_map_features(M[:config.batch_size - 1])
    assert m_desc.shape[0] == config.batch_size - 1

def test_map_desc_shape2():
    M = dataset.get_map_paths(rootdir=config.root_dir)
    m_desc = method.compute_map_features(M[:config.batch_size + 1])
    assert m_desc.shape[0] == config.batch_size + 1

def test_query_desc_type():
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    q_desc = method.compute_query_desc(Q[:1])
    assert q_desc.dtype == np.float32

def test_map_desc_shape():
    M = dataset.get_map_paths(rootdir=config.root_dir)
    m_desc = method.compute_map_features(M[:1])
    assert m_desc.dtype == np.float32


def test_perform_vpr():
    query_img = glob(config.root_dir + '/tests/test_techniques/resources/test_query_img/*')[0]
    db_img = sorted(glob(config.root_dir + '/tests/test_techniques/resources/test_db_img/*'))
    m_desc = method.compute_map_features(db_img)
    idx, score = method.perform_vpr(query_img, m_desc)
    assert idx == 0


def test_perform_vpr():
    query_img = glob(config.root_dir + '/tests/test_techniques/resources/test_query_img/*')[0]
    db_img = sorted(glob(config.root_dir + '/tests/test_techniques/resources/test_db_img/*'))
    m_desc = method.compute_map_features(db_img)
    idx, score = method.perform_vpr(query_img, m_desc)
    assert idx == 0
    assert isinstance(idx, int)
    assert isinstance(score, float)
    assert 0 <= score <= 1.01

def test_matching_method():
    Q = dataset.get_query_paths(rootdir=config.root_dir)
    M = dataset.get_map_paths(rootdir=config.root_dir)
    m_desc = method.compute_map_features(M[:10])
    q_desc = method.compute_query_desc(Q[:3])
    S = method.matching_method(q_desc, m_desc)
    assert S.shape[0] == 3
    assert S.shape[1] == 10