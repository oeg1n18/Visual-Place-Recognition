import os
import glob
import urllib
import zipfile
import urllib.request
import numpy as np
from scipy.signal import convolve2d

NAME = 'GardensPointWalking'
def get_query_paths(session_type='ms', rootdir='/Users/olivergrainge/Documents/github/Visual-Place-Recognition'):
    if session_type=='ms':
        path = rootdir + '/vpr/data/raw_data/GardensPointWalking'
        query_paths = sorted(glob.glob(path + "/night_right/*"))
        return query_paths
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")



def get_map_paths(session_type='ms', rootdir='/Users/olivergrainge/Documents/github/Visual-Place-Recognition'):
    if session_type=='ms':
        path = rootdir + '/vpr/data/raw_data/GardensPointWalking'
        test_paths = sorted(glob.glob(path + "/day_right/*"))
        return test_paths
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")

#

def get_gtmatrix(session_type='ms', gt_type='hard', rootdir='/Users/olivergrainge/Documents/github/Visual-Place-Recognition'):
    if session_type=='ms':
        query_paths = get_query_paths(session_type=session_type, rootdir=rootdir)
        map_paths = get_map_paths(session_type=session_type, rootdir=rootdir)
        gtmatrix = np.eye(len(map_paths)).astype('bool')
        if gt_type == 'soft':
            gtmatrix = convolve2d(gtmatrix.astype(int), np.ones((17,1), 'int'), mode='same').astype('bool')
        return gtmatrix
    else:
        raise Exception("Only session_type=ms available for this GardensPoint Dataset")


def download(rootdir=None):
    destination = rootdir + '/vpr/data/raw_data/GardensPointWalking'
    print('===== GardensPoint dataset does not exist. Download to ' + destination + '...')

    fn = 'GardensPoint_Walking.zip'
    url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

    # create folders
    path = os.path.expanduser(destination)
    os.makedirs(path, exist_ok=True)

    # download
    urllib.request.urlretrieve(url, path + fn)

    # unzip
    with zipfile.ZipFile(path + fn, 'r') as zip_ref:
        zip_ref.extractall(destination)

    # remove zipfile
    os.remove(destination + fn)
