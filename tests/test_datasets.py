import pytest
import importlib
from PIL import Image
import config
import numpy as np

datasets = ["ESSEX3IN1", "Nordlands", "GardensPointWalking", "SFU", "SPED_V2", "berlin_kudamm", "StLucia", "RobotCars_short"]

@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_name(dataset_name):
    ds = importlib.import_module(dataset_name)
    assert isinstance(ds.NAME, str)
    del ds

@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_query_load(dataset_name):
    ds = importlib.import_module(dataset_name)
    Q = ds.get_query_paths()
    for q in Q:
        Image.open(q)
    del ds

@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_map_load(dataset_name):
    ds = importlib.import_module(dataset_name)
    M = ds.get_map_paths()
    for m in M:
        Image.open(m)
    del ds


@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_consistensy(dataset_name):
    ds = importlib.import_module(dataset_name)
    Q = ds.get_query_paths()
    M = ds.get_map_paths()
    Q1 = ds.get_query_paths()
    M1 = ds.get_map_paths()
    assert Q == Q1
    assert M == M1


@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_gt_shape(dataset_name):
    ds = importlib.import_module(dataset_name)
    Q = ds.get_query_paths()
    M = ds.get_map_paths()
    GT = ds.get_gtmatrix()
    assert GT.shape[0] == len(M)
    assert GT.shape[1] == len(Q)


@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_gt_repeatability(dataset_name):
    ds = importlib.import_module(dataset_name)
    GT = ds.get_gtmatrix()
    GT1 = ds.get_gtmatrix()
    assert (GT == GT1).all()


@pytest.mark.parametrize("dataset_name", ["vpr.data.datasets." + dataset for dataset in datasets])
def test_gt_dtype(dataset_name):
    ds = importlib.import_module(dataset_name)
    GT = ds.get_gtmatrix()
    assert isinstance(GT, np.ndarray)
    assert GT.dtype == np.uint8
