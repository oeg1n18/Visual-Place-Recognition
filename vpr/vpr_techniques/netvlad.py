from sklearn.metrics.pairwise import cosine_similarity

from vpr.vpr_techniques.patchnetvlad import PATCHNETVLAD_ROOT_DIR
import configparser
from vpr.vpr_techniques.patchnetvlad import PatchNetVLADFeatureExtractor
from vpr.vpr_techniques.utils import save_descriptors
import os
import PIL.Image as Image
import torch
import torch
import torch
import numpy as np
import faiss


NAME = 'NetVLAD'
configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')

assert os.path.isfile(configfile)
config = configparser.ConfigParser()
config.read(configfile)
feature_extractor = PatchNetVLADFeatureExtractor(config)
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

class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc
        self.index = faiss.IndexFlatL2(m_desc.shape[1])
        self.index.add(m_desc)

    def perform_vpr(self, q_path):
        if len(q_path) == 1:
            q_desc = compute_query_desc(q_path, disable_pbar=True)
            D, I = self.index.search(q_desc.astype(np.float32), 1)
            I = I[0][0]
            score = cosine_similarity(self.m_desc[I][None, :], q_desc)
            return I, score
        else:
            q_desc = compute_query_desc(q_path)
            D, I = self.index.search(q_desc.astype(np.float32),1)
            scores = cosine_similarity(self.m_desc[I].squeeze(), q_desc)
            return I.squeeze(), scores

def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc)
