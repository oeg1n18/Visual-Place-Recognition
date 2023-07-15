from sklearn.metrics.pairwise import cosine_similarity

from vpr.vpr_techniques.patchnetvlad import PATCHNETVLAD_ROOT_DIR
import configparser
from vpr.vpr_techniques.patchnetvlad import PatchNetVLADFeatureExtractor
from vpr.vpr_techniques.utils import save_descriptors
import os
from tqdm import tqdm
from vpr.data.utils import VprDataset
import numpy as np
import faiss
from torch.utils.data import DataLoader
import config as my_config

NAME = 'NetVLAD'
configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')

assert os.path.isfile(configfile)
config = configparser.ConfigParser()
config.read(configfile)
feature_extractor = PatchNetVLADFeatureExtractor(config)


def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    ds = VprDataset(Q, transform=None)
    dl = DataLoader(ds, batch_size=my_config.batch_size, shuffle=False)
    all_q_desc = []
    for batch in tqdm(dl, desc="Computing Query Descriptors", disable=disable_pbar):
        q_imgs = list(batch.numpy())
        all_q_desc.append(feature_extractor.compute_features(q_imgs, disable_pbar=True))
    q_desc = np.vstack(all_q_desc)
    q_desc = q_desc / np.linalg.norm(q_desc, axis=1, keepdims=True)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc


def compute_map_features(M, dataset_name=None, disable_pbar=False):
    ds = VprDataset(M, transform=None)
    dl = DataLoader(ds, batch_size=my_config.batch_size, shuffle=False)
    all_m_desc = []
    for batch in tqdm(dl, desc="Computing map Descriptors", disable=disable_pbar):
        q_imgs = list(batch.numpy())
        all_m_desc.append(feature_extractor.compute_features(q_imgs, disable_pbar=True))
    m_desc = np.vstack(all_m_desc)
    m_desc = m_desc / np.linalg.norm(m_desc, axis=1, keepdims=True)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc


def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc).transpose()


class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc
        self.index = faiss.IndexFlatIP(m_desc.shape[1])
        faiss.normalize_L2(m_desc)
        self.index.add(m_desc)

    def perform_vpr(self, q_path):
        q_desc = compute_query_desc(q_path, disable_pbar=True)
        q_desc = q_desc.astype(np.float32)
        faiss.normalize_L2(q_desc)
        D, I = self.index.search(q_desc, 1)
        temp_mdesc = self.m_desc[I].squeeze() if self.m_desc[I].squeeze().ndim == 2 else self.m_desc[I][0]
        scores = cosine_similarity(q_desc, temp_mdesc).diagonal()
        return I.flatten(), scores
