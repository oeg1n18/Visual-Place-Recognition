import os
import urllib
import zipfile
import urllib.request
from os.path import isdir

import numpy as np
import glob

NAME = 'SFU'
def get_query_paths(session_type='ms', rootdir=None):
    if isdir(rootdir + '/vpr/data/raw_data/SFU'):
        return sorted(glob.glob(rootdir + '/vpr/data/raw_data/SFU/jan/*.jpg'))
    else:
        print("must download dataset first into rootdir + /vpr/data/raw_data/ directory")


def get_map_paths(session_type='ms', rootdir=None):
    if isdir(rootdir + '/vpr/data/raw_data/SFU'):
        return sorted(glob.glob(rootdir + '/vpr/data/raw_data/SFU/dry/*.jpg'))
    else:
        print("must download dataset first into rootdir + /vpr/data/raw_data/ directory")

def get_gtmatrix(session_type='ms', gt_type='hard', rootdir=None):
    gt_data = np.load(rootdir + '/vpr/data/raw_data/SFU/GT.npz')
    if gt_type=='hard':
        GT = gt_data['GThard'].astype('bool')
    else:
        GT = gt_data['GTsoft'].astype('bool')
    return GT

def download(rootdir=None):
    destination = rootdir + '/vpr/data/raw_data/SFU'
    print('===== SFU dataset does not exist. Download to ' + destination + '...')
    fn = 'SFU.zip'
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