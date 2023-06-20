import glob
import os
import urllib
import urllib.request
import zipfile
import config

import numpy as np

NAME = 'StLucia'
def get_query_paths(session_type='ms'):
    return sorted(glob.glob(config.root_dir + '/vpr/data/raw_data/StLucia_small/180809_1545/*.jpg'))


def get_map_paths(session_type='ms'):
    return sorted(glob.glob(config.root_dir + '/vpr/data/raw_data/StLucia_small/100909_0845/*.jpg'))

def get_gtmatrix(session_type='ms', gt_type='soft'):
    gt_data = np.load(config.root_dir + '/vpr/data/raw_data/StLucia_small/GT.npz')
    if gt_type=='hard':
        GT = gt_data['GThard'].astype('bool')
    else:
        GT = gt_data['GTsoft'].astype('bool')
    return GT.astype(np.uint8)

def download():
    destination = config.root_dir + '/vpr/data/raw_data/StLucia_small'
    print('===== StLucia dataset does not exist. Download to ' + destination + '...')

    fn = 'StLucia_small.zip'
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
