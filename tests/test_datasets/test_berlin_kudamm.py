import nptyping
import pytest
from vpr.data.datasets import berlin_kudamm
import config
import numpy as np

dataset = berlin_kudamm
def test1():
    assert type(dataset.NAME) == type('abc')

def test2():
    Q = dataset.get_query_paths()
    assert len(Q) > 0

def test3():
    M = dataset.get_map_paths()
    assert len(M) > 0

def test4():
    GT = dataset.get_gtmatrix()
    assert GT.shape[0] > 0
    assert GT.shape[1] > 0
    assert GT.dtype == np.zeros_like(GT).astype(np.uint8).dtype
def test5():
    Q, M = dataset.get_query_paths(), dataset.get_map_paths()
    GT = dataset.get_gtmatrix()
    assert GT.shape[0] == len(Q)
    assert GT.shape[1] == len(M)

def test6():
    GT = dataset.get_gtmatrix()
    assert 1 in np.unique(GT)
    assert 0 in np.unique(GT)
    assert len(np.unique(GT)) == 2

def test7():
    GT = dataset.get_gtmatrix()
    Q, M = dataset.get_query_paths(), dataset.get_map_paths()
    h, _ = np.histogram(GT, bins=[0, 0.5, 1.5])
    assert h[0] > len(M)
    assert h[1] >= len(Q)
