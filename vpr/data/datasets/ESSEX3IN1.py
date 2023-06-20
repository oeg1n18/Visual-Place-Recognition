
import config
from glob import glob
import numpy as np

NAME = 'ESSEX3IN1'
def get_query_paths(session_type='ms'):
    if session_type=='ms':
        return sorted(glob(config.root_dir + '/vpr/data/raw_data/essex3IN1/query_combined/*.png'))


def  get_map_paths(session_type='ms'):
    if session_type=='ms':
        return sorted(glob(config.root_dir + '/vpr/data/raw_data/essex3IN1/reference_combined/*.png'))

def get_gtmatrix(session_type='ms', gt_type='hard'):
    Q = get_query_paths()
    M = get_map_paths()
    assert len(Q) == len(M)
    GT = np.eye(len(Q)).astype(np.uint8)
    return GT


def download():
    print("Seek permission of dataset owner")
