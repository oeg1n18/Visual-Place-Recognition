
from src.vpr_techniques.patchnetvlad import PATCHNETVLAD_ROOT_DIR
import configparser
from src.vpr_techniques.patchnetvlad import PatchNetVLADFeatureExtractor
import os
import PIL.Image as Image
import numpy as np


NAME = 'NetVLAD'
configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')

assert os.path.isfile(configfile)
config = configparser.ConfigParser()
config.read(configfile)
feature_extractor = PatchNetVLADFeatureExtractor(config)

def compute_query_desc(Q, dataset_name=None):
    q_imgs = [np.array(Image.open(img_path)) for img_path in Q]
    q_desc = feature_extractor.compute_features(q_imgs)
    q_desc = q_desc / np.linalg.norm(q_desc, axis=1, keepdims=True)

    if dataset_name is not None:
        pth = config.root_dir + '/src/descriptors/' + dataset_name
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + NAME):
                np.save(config.root_dir + '/src/descriptors/python/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', q_desc)
            else:
                os.mkdir(pth + '/' + NAME)
                np.save(config.root_dir + '/src/descriptors/python/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', q_desc)

    return q_desc

def compute_map_features(M, dataset_name=None):
    m_imgs = [np.array(Image.open(img_path)) for img_path in M]
    m_desc = feature_extractor.compute_features(m_imgs)
    m_desc = m_desc / np.linalg.norm(m_desc, axis=1, keepdims=True)

    if dataset_name is not None:
        pth = config.root_dir + '/src/descriptors/' + dataset_name
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + NAME):
                np.save(config.root_dir + '/src/descriptors/python/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', m_desc)
            else:
                os.mkdir(pth + '/' + NAME)
                np.save(config.root_dir + '/src/descriptors/python/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', m_desc)

    return m_desc

def perform_vpr(q_path, M):
    q_img = [np.array(Image.open(q_path))]
    q_desc = feature_extractor.compute_features(q_img)
    S = np.matmul(q_desc, M.T)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return j, S[i, j]