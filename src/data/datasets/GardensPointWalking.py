import os
import glob
import numpy as np
from scipy.signal import convolve2d


def get_query_paths(session_type='ms', rootdir='/home/ollie/Documents/Github/Visual-Place-Recognition'):
    if session_type=='ms':
        path = rootdir + '/src/data/raw_data/GardensPointWalking'
        query_paths = sorted(glob.glob(path + "/night_right/*"))
        return query_paths
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")



def get_map_paths(session_type='ms', rootdir='/home/ollie/Documents/Github/Visual-Place-Recognition'):
    if session_type=='ms':
        path = rootdir + '/src/data/raw_data/GardensPointWalking'
        test_paths = sorted(glob.glob(path + "/day_right/*"))
        return test_paths
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")



def get_gtmatrix(session_type='ms', gt_type='hard', rootdir='/home/ollie/Documents/Github/Visual-Place-Recognition'):
    if session_type=='ms':
        query_paths = get_query_paths(session_type=session_type, rootdir=rootdir)
        map_paths = get_map_paths(session_type=session_type, rootdir=rootdir)
        gtmatrix = np.eye(len(map_paths)).astype('bool')
        if gt_type == 'soft':
            gtmatrix = convolve2d(gtmatrix.astype(int), np.ones((17,1), 'int'), mode='same').astype('bool')
        return gtmatrix
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")
