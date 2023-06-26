import random

import pytorch_lightning as pl
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from vpr.vpr_techniques.techniques.switchCNN.models import mobilenetModule, resnet9Module
from vpr.vpr_techniques.techniques.switchCNN.train import resnet_transforms_test
from vpr.vpr_techniques import netvlad, hog, cosplace, mixvpr
from vpr.data.datasets import Nordlands
import torch
import numpy as np
from PIL import Image
import faiss
from numpy.linalg import norm
import torch.nn.functional as F

from vpr.vpr_techniques.utils import save_descriptors

selection_model_pth = "//home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/switchCNN/weights/resnet9-epoch=632-val_loss=0.00-other_metric=0.00.ckpt"
selection_transform = resnet_transforms_test
selection_model = resnet9Module.load_from_checkpoint(checkpoint_path=selection_model_pth)
selection_model.eval()

NAME = "switchCNN"
techniques = [netvlad, hog, cosplace, mixvpr]


def logits2selections(logits):
    selection_probs = F.sigmoid(logits)
    selections = torch.argmax(selection_probs, dim=1).numpy()
    return selections.flatten()


@torch.no_grad()
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    # Computing VPR technqiue selections
    X = torch.stack([resnet_transforms_test(Image.open(q)) for q in Q])
    selections = []
    for batch in torch.chunk(X, int(max(len(Q) / config.batch_size, 1))):
        logits = selection_model(batch.to(config.device)).detach().cpu()
        sample_selections = logits2selections(logits)
        selections += list(sample_selections)
    selections = np.array(selections)
    # Computing the descriptors with VPR method selections

    q_desc = []
    for i, s in enumerate(selections):
        q_desc.append(techniques[s].compute_query_desc([Q[i]]).squeeze())
    """
    all_queries = [[] for _ in range(len(techniques))]
    for i, query in enumerate(Q):
        all_queries[selections[i]].append(query)
    all_desc = []
    for i, technique in enumerate(techniques):
        if len(all_queries[i]) == 0:
            all_desc.append([])
        else:
            all_desc.append(technique.compute_query_desc(all_queries[i]))

    q_desc = []
    selection_idxs = [0 for _ in range(len(techniques))]
    for i in range(len(Q)):
        q_desc.append(all_desc[selections[i]][selection_idxs[selections[i]]])
        selection_idxs[selections[i]] += 1
    """
    # saving and returning both query vpr technique selections and descriptors
    all_desc = (q_desc, list(selections))
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, all_desc, type='query')

    return all_desc


@torch.no_grad()
def compute_map_features(M, dataset_name=None, disable_pbar=False):
    all_desc = [technique.compute_map_features(M, disable_pbar=disable_pbar) for technique in techniques]
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, all_desc, type='map')
    return all_desc


def matching_method(q_desc, m_desc):
    desc, select_idx = q_desc
    S = np.vstack([cosine_similarity(q[None, :], m_desc[select_idx[i]]) for i, q in enumerate(desc)]).transpose()
    #S = (S - np.mean(S, axis=0, keepdims=True)) / np.std(S, axis=0, keepdims=True)
    S = S/np.linalg.norm(S, axis=0, keepdims=True)
    return S


@torch.no_grad()
class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc
        self.indexes = [faiss.IndexFlatL2(m.shape[1]) for m in m_desc]
        for i, desc in enumerate(m_desc):
            self.indexes[i].add(desc)

    def perform_vpr(self, q_path):
        q_desc, selections = compute_query_desc(q_path, disable_pbar=True)
        Is = []
        all_scores = []
        for i, desc in enumerate(q_desc):
            D, I = self.indexes[selections[i]].search(desc[None, :].astype(np.float32), 1)
            temp_mdesc = self.m_desc[selections[i]].squeeze() if self.m_desc[selections[i]].squeeze().ndim == 2 else \
                self.m_desc[selections[i]][0]
            scores = cosine_similarity(q_desc[i][None, :], temp_mdesc).diagonal()
            Is += list(I.flatten())
            all_scores.append(scores)

        return np.array(Is), np.array(all_scores).flatten()

