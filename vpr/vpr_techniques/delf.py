from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vpr.vpr_techniques.techniques.delf.feature_extractor_holistic import HDCDELF
from vpr.vpr_techniques.utils import save_descriptors
import torch
NAME = 'HDC-DELF'

feature_extractor = HDCDELF()
@torch.no_grad()
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    q_imgs = [np.array(Image.open(img_path)) for img_path in Q]
    q_desc = feature_extractor.compute_features(q_imgs, disable_pbar=disable_pbar)
    q_desc = q_desc / np.linalg.norm(q_desc, axis=1, keepdims=True)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc
@torch.no_grad()
def compute_map_features(M, dataset_name=None, disable_pbar=False):
    m_imgs = [np.array(Image.open(img_path)) for img_path in M]
    m_desc = feature_extractor.compute_features(m_imgs, disable_pbar=disable_pbar)
    m_desc = m_desc / np.linalg.norm(m_desc, axis=1, keepdims=True)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc
@torch.no_grad()
def perform_vpr(q_path, M):
    q_img = [np.array(Image.open(q_path))]
    q_desc = feature_extractor.compute_features(q_img)
    q_desc = q_desc / np.linalg.norm(q_desc, axis=1, keepdims=True)
    S = np.matmul(q_desc, M.T)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return int(j), float(S[i, j])

def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc)

