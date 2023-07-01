
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
import configparser
from vpr.vpr_techniques.techniques.patchnetvlad.feature_extractor_patchnetvlad import \
    PatchNetVLADFeatureExtractor
import os
import PIL.Image as Image
import numpy as np
from vpr.vpr_techniques.utils import save_descriptors
import torch

NAME = 'PatchNetVLAD'
batch_size=12

configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')

assert os.path.isfile(configfile)
config = configparser.ConfigParser()
config.read(configfile)
feature_extractor = PatchNetVLADFeatureExtractor(config)

@torch.no_grad()
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    q_imgs = [np.array(Image.open(img_path)) for img_path in Q]
    _, q_patch = feature_extractor.compute_features(q_imgs, disable_pbar)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_patch, type='query')
    return q_patch

@torch.no_grad()
def compute_map_features(M, dataset_name=None, disable_pbar=False):
    m_imgs = [np.array(Image.open(img_path)) for img_path in M]
    _, m_patch = feature_extractor.compute_features(m_imgs, disable_pbar)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_patch, type='map')
    return m_patch


def matching_method(q_desc_patches, m_desc_patches):
    S = feature_extractor.local_matcher_from_numpy_single_scale(q_desc_patches, m_desc_patches)
    return S

# These needs making consistent with the others
class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc

    def perform_vpr(self, q_path, disable_pbar=True):
        q_img = [np.array(Image.open(q_path))]
        _, q_desc = feature_extractor.compute_features(q_img, disable_pbar=disable_pbar)
        S = matching_method(q_desc, self.m_desc)
        i, j = np.unravel_index(S.argmax(), S.shape)
        return int(j), float(S[i, j])