import pytest
import importlib
from vpr.vpr_techniques import hog, mixvpr, cosplace, netvlad, patchnetvlad, delf, cohog
from vpr.data.datasets import SFU
from vpr.vpr_techniques.utils import load_descriptors
import numpy as np
from glob import glob
import config
from vpr.vpr_techniques.utils import load_descriptors

dataset = SFU
methods = ["hog", "netvlad", "mixvpr", "cosplace", "switchCNN"]
methods = ["netvlad", "cosplace", "mixvpr", "hog"]


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_contains_name(method_name):
    method = importlib.import_module(method_name)
    assert isinstance(method.NAME, str)
    del method

@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_computes_query_long(method_name):
    method = importlib.import_module(method_name)
    Q = SFU.get_query_paths()[:10]
    q_desc = method.compute_query_desc(Q, disable_pbar=True)
    assert q_desc is not None
    del method

@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_computes_query_short(method_name):
    method = importlib.import_module(method_name)
    Q = SFU.get_query_paths()[0]
    q_desc = method.compute_query_desc([Q], disable_pbar=True)
    assert q_desc is not None
    del method

@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_computes_map_long(method_name):
    method = importlib.import_module(method_name)
    M = SFU.get_query_paths()[:10]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    assert m_desc is not None
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_computes_map_short(method_name):
    method = importlib.import_module(method_name)
    M = SFU.get_query_paths()[0]
    m_desc = method.compute_map_features([M], disable_pbar=True)
    assert m_desc is not None
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_save_query_descriptors(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:3]
    M = dataset.get_map_paths()[:3]
    m_desc = method.compute_map_features(M, dataset.NAME, disable_pbar=True)
    q_desc = method.compute_query_desc(Q, dataset.NAME, disable_pbar=True)
    q_desc_load, m_desc_load = load_descriptors(dataset.NAME, method.NAME)
    if isinstance(q_desc, np.ndarray):
        assert (q_desc == q_desc_load).all()
        assert (m_desc == m_desc_load).all()
    elif q_desc:
        for i in range(len(q_desc[0])):
            assert (q_desc[0][i] == q_desc_load[0][i]).all()
        assert q_desc[1] == q_desc_load[1]

        for i in range(len(m_desc)):
            assert (m_desc[i] == m_desc_load[i]).all()
    del method



@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_repeatability_query(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:3]
    q_desc = method.compute_query_desc(Q, disable_pbar=True)
    q_desc2 = method.compute_query_desc(Q, disable_pbar=True)
    if isinstance(q_desc, np.ndarray):
        assert (q_desc == q_desc2).all()
    elif isinstance(q_desc, tuple):
        for i in range(len(q_desc[0])):
            assert (q_desc[0][i] == q_desc2[0][i]).all()
        assert q_desc[1] == q_desc2[1]
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_repeatability_map(method_name):
    method = importlib.import_module(method_name)
    M = dataset.get_map_paths()[:5]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    m_desc2 = method.compute_map_features(M, disable_pbar=True)
    if isinstance(m_desc, np.ndarray):
        assert (m_desc == m_desc2).all()
    elif isinstance(m_desc, list):
        for i in range(len(m_desc)):
            assert (m_desc[i] == m_desc2[i]).all()
    del method

@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_matching(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:3]
    M = dataset.get_map_paths()[:5]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    q_desc = method.compute_query_desc(Q, disable_pbar=True)
    S = method.matching_method(q_desc, m_desc)
    assert S.min() >= 0.
    assert S.max() <= 1.
    assert S.shape[0] == 5
    assert S.shape[1] == 3
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_retrieval_short(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:3]
    M = dataset.get_map_paths()[:5]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    q_desc = method.compute_query_desc(Q, disable_pbar=True)
    S = method.matching_method(q_desc, m_desc)
    assert S.min() >= 0.
    assert S.max() <= 1.
    assert S.shape[0] == 3
    assert S.shape[1] == 5
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_retrieval_short(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[0]
    M = dataset.get_map_paths()[:5]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    pr = method.PlaceRecognition(m_desc)
    matches, scores = pr.perform_vpr([Q])
    assert isinstance(matches, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert len(matches) == 1
    assert len(scores) == 1
    assert matches.dtype == int
    assert scores.dtype == np.float32
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_retrieval_longs(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:5]
    M = dataset.get_map_paths()[:5]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    pr = method.PlaceRecognition(m_desc)
    matches, scores = pr.perform_vpr(Q)
    assert isinstance(matches, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert matches.dtype == int
    assert scores.dtype == np.float32
    assert len(matches) == 5
    assert len(scores) == 5
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_retrieval_example(method_name):
    method = importlib.import_module(method_name)
    Q = sorted(glob(config.root_dir + "/tests/resources/test_query_img/*"))
    M = sorted(glob(config.root_dir + "/tests/resources/test_db_img/*"))[:30]
    m_desc = method.compute_map_features(M, disable_pbar=True)
    pr = method.PlaceRecognition(m_desc)
    matches, scores = pr.perform_vpr(Q)
    assert matches[0] == 0
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_retrieval_longs(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:5]
    M = dataset.get_map_paths()[:10]
    m_desc = method.compute_map_features(M, disable_pbar=True, dataset_name=dataset.NAME)
    q_desc = method.compute_query_desc(Q, disable_pbar=True, dataset_name=dataset.NAME)
    q_load, m_load = load_descriptors(dataset_name=dataset.NAME, method_name=method.NAME)
    if isinstance(q_desc, np.ndarray):
        assert (q_desc == q_load).all()
        assert (m_desc == m_load).all()
    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_matching_repeatability(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:5]
    M = dataset.get_map_paths()[:10]
    q_desc = method.compute_query_desc(Q)
    m_desc = method.compute_map_features(M)
    S = method.matching_method(q_desc, m_desc)
    matches = np.argmax(S, axis=0).flatten()
    q_desc2 = method.compute_query_desc(Q)
    m_desc2 = method.compute_map_features(M)
    S2 = method.matching_method(q_desc2, m_desc2)
    matches2 = np.argmax(S2, axis=0).flatten()
    assert (matches == matches2).all()
    del method



@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_descriptor_repeatability(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:5]
    M = dataset.get_map_paths()[:10]
    q_desc1 = method.compute_query_desc(Q)
    m_desc1 = method.compute_map_features(M)
    q_desc2 = method.compute_query_desc(Q)
    m_desc2 = method.compute_map_features(M)

    if isinstance(q_desc1, np.ndarray):
        assert (q_desc1 == q_desc2).all()
    elif isinstance(q_desc1, tuple):
        q_desc1, selections1 = q_desc1
        q_desc2, selections2 = q_desc2
        for i in range(len(q_desc1)):
            assert (q_desc2[i] == q_desc1[i]).all()
        assert selections1 == selections2
    else:
        raise Exception("Do not know how to check for this descriptor type")

    if isinstance(m_desc1, np.ndarray):
        assert (m_desc1 == m_desc2).all()
    elif isinstance(m_desc1, list):
        for i in range(len(m_desc1)):
            assert (m_desc1[i] == m_desc2[i]).all()

    del method


@pytest.mark.parametrize("method_name", ["vpr.vpr_techniques." + method for method in methods])
def test_descriptor_equivalence(method_name):
    method = importlib.import_module(method_name)
    Q = dataset.get_query_paths()[:10]
    q_desc = method.compute_query_desc(Q)
    q_desc2 = np.vstack(method.compute_query_desc([q]) for q in Q)
    if method.NAME != "switchCNN":
        assert (q_desc == q_desc2).all()
    del method




















