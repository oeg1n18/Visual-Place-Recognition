from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
import configparser
from src.vpr_techniques.python.techniques.patchnetvlad.feature_extractor_patchnetvlad import \
    PatchNetVLADFeatureExtractor
import os
import PIL.Image as Image
import numpy as np

NAME = 'PatchNetVLAD'
configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')

assert os.path.isfile(configfile)
config = configparser.ConfigParser()
config.read(configfile)
feature_extractor = PatchNetVLADFeatureExtractor(config)


def compute_query_desc(Q, dataset_name=None):
    q_imgs = [np.array(Image.open(img_path)) for img_path in Q]
    _, q_patch = feature_extractor.compute_features(q_imgs)

    if dataset_name is not None:
        pth = config.root_dir + '/src/descriptors/' + dataset_name
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + NAME):
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', q_patch)
            else:
                os.mkdir(pth + '/' + NAME)
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', q_patch)
    return q_patch


def compute_map_features(M, dataset_name=None):
    m_imgs = [np.array(Image.open(img_path)) for img_path in M]
    _, m_patch = feature_extractor.compute_features(m_imgs)

    if dataset_name is not None:
        pth = config.root_dir + '/src/descriptors/' + dataset_name
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + NAME):
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', m_patch)
            else:
                os.mkdir(pth + '/' + NAME)
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', m_patch)
    return m_patch


def matching_function(q_desc_patches, m_desc_patches):
    S = feature_extractor.local_matcher_from_numpy_single_scale(q_desc_patches, m_desc_patches)
    return S


def perform_vpr(q_path, m_desc):
    q_img = [np.array(Image.open(q_path))]
    q_desc = feature_extractor.compute_features(q_img)
    S = matching_function(q_desc, m_desc)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return j, S[i, j]
