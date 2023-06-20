import pytorch_lightning as pl
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from vpr.data.utils import VprDataset
from vpr.vpr_techniques.techniques.switchCNN.models import mobilenetModule, resnet9Module, resnet18_regression_Module
from vpr.vpr_techniques.techniques.switchCNN.train import resnet_transforms_test, mobilenet_transform
from vpr.vpr_techniques.techniques.switchCNN.train import mobilenet_transform
import torch
import numpy as np
from PIL import Image
import faiss

from vpr.vpr_techniques.utils import save_descriptors

selection_model_pth = "/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/switchCNN/weights/resnet18-epoch=198-val_loss=1.34-other_metric=0.00.ckpt"
selection_transform = resnet_transforms_test
selection_model = resnet18_regression_Module.load_from_checkpoint(checkpoint_path=selection_model_pth)

NAME = "switchCNN"

@torch.no_grad()
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    if len(Q) > config.batch_size:
        selection_dl = DataLoader(VprDataset(Q, transform=selection_transform), batch_size=config.batch_size,
                                  shuffle=False)
        selections = []
        for batch in tqdm(selection_dl, desc='Computing Query Descriptors', disable=disable_pbar):
            selection_probs = selection_model(batch.to(config.device)).detach().cpu()
            selection_probs = selection_probs.softmax(dim=1)
            select = torch.argmax(selection_probs, dim=1)
            selections += list(np.array(select))
        selections = np.array(selections)
    else:
        imgs = torch.stack([selection_transform(Image.open(pth)) for pth in Q])
        selection_probs = selection_model(imgs.to(config.device)).detach().cpu()
        selection_probs = selection_probs.softmax(dim=1)
        selections = torch.argmax(selection_probs, dim=1)
        selections = np.array(selections)

    q_desc = list(np.zeros(len(Q)))
    cosplace_idx = np.argwhere(selections == 0)
    netvlad_idx = np.argwhere(selections == 1)
    delf_idx = np.argwhere(selections == 2)
    mixvpr_idx = np.argwhere(selections == 3)
    hog_idx = np.argwhere(selections == 4)

    from vpr.vpr_techniques import cosplace
    select_queries = np.array(Q)[cosplace_idx].squeeze()
    if select_queries.size > 0:
        if select_queries.size == 1:
            select_queries = np.array([select_queries])
        q_desc_cosplace = cosplace.compute_query_desc(list(select_queries))
        if q_desc_cosplace.shape[0] == 1:
            q_desc[cosplace_idx.squeeze()] = q_desc_cosplace[0]
        else:
            for i, desc in enumerate(q_desc_cosplace):
                q_desc[cosplace_idx.squeeze()[i]] = desc
        del cosplace

    from vpr.vpr_techniques import netvlad
    select_queries = np.array(Q)[netvlad_idx].squeeze()
    if select_queries.size > 0:
        if select_queries.size == 1:
            select_queries = np.array([select_queries])
        q_desc_netvlad = netvlad.compute_query_desc(list(select_queries))
        if q_desc_netvlad.shape[0] == 1:
            q_desc[netvlad_idx.squeeze()] = q_desc_netvlad[0]
        else:
            for i, desc in enumerate(q_desc_netvlad):
                q_desc[netvlad_idx.squeeze()[i]] = desc
        del netvlad

    from vpr.vpr_techniques import delf
    select_queries = np.array(Q)[delf_idx].squeeze()
    if select_queries.size > 0:
        if select_queries.size == 1:
            select_queries = np.array([select_queries])
        q_desc_delf = delf.compute_query_desc(list(select_queries))
        if q_desc_delf.shape[0] == 1:
            q_desc[delf_idx.squeeze()] = q_desc_delf[0]
        else:
            for i, desc in enumerate(q_desc_delf):
                q_desc[delf_idx.squeeze()[i]] = desc
        del delf

    from vpr.vpr_techniques import mixvpr
    select_queries = np.array(Q)[mixvpr_idx].squeeze()
    if select_queries.size > 0:
        if select_queries.size == 1:
            select_queries = np.array([select_queries])
        q_desc_mixvpr = mixvpr.compute_query_desc(list(select_queries))
        if q_desc_mixvpr.shape[0] == 1:
            q_desc[mixvpr_idx.squeeze()] = q_desc_mixvpr[0]
        else:
            for i, desc in enumerate(q_desc_mixvpr):
                q_desc[mixvpr_idx.squeeze()[i]] = desc
        del mixvpr

    from vpr.vpr_techniques import hog
    select_queries = np.array(Q)[hog_idx].squeeze()
    if select_queries.size > 0:
        if select_queries.size == 1:
            select_queries = np.array([select_queries])
        q_desc_hog = hog.compute_query_desc(list(select_queries))
        if q_desc_hog.shape[0] == 1:
            q_desc[hog_idx.squeeze()] = q_desc_hog[0]
        else:
            for i, desc in enumerate(q_desc_hog):
                q_desc[hog_idx.squeeze()[i]] = desc

    all_desc = (q_desc, selections)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, all_desc, type='query')
    return all_desc


@torch.no_grad()
def compute_map_features(M, dataset_name=None, disable_pbar=False):
    from vpr.vpr_techniques import cosplace
    m_desc_cosplace = cosplace.compute_map_features(M, disable_pbar=disable_pbar)
    del cosplace
    from vpr.vpr_techniques import netvlad
    m_desc_netvlad = netvlad.compute_map_features(M, disable_pbar=disable_pbar)
    del netvlad
    from vpr.vpr_techniques import delf
    m_desc_delf = delf.compute_map_features(M, disable_pbar=disable_pbar)
    del delf
    from vpr.vpr_techniques import mixvpr
    m_desc_mixvpr = mixvpr.compute_map_features(M, disable_pbar=disable_pbar)
    del mixvpr
    from vpr.vpr_techniques import hog
    m_desc_hog = hog.compute_map_features(M, disable_pbar=disable_pbar)
    del hog
    all_desc = (m_desc_cosplace, m_desc_netvlad, m_desc_delf, m_desc_mixvpr, m_desc_hog)

    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, all_desc, type='map')
    return all_desc


def matching_method(q_desc, m_desc):
    q_desc, select_idx = q_desc
    S = []
    for i in range(len(select_idx)):
        S.append(cosine_similarity(q_desc[i][None, :], m_desc[select_idx[i]]))
    return np.vstack(S)
@torch.no_grad()
class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc
        self.indexes = [faiss.IndexFlatL2(m.shape[1]) for m in m_desc]
        for i, desc in enumerate(m_desc):
            self.indexes[i].add(desc)

    def perform_vpr(self, q_path):
        if isinstance(q_path, str):
            q_desc, selections = compute_query_desc([q_path], disable_pbar=True)
            D, I = self.indexes[selections[0]].search(q_desc.astype(np.float32), 1)
            I = I[0][0]
            score = cosine_similarity(self.m_desc[I][None, :], q_desc)
            return I.squeeze(), score.squeeze()
        else:
            q_desc, selections = compute_query_desc(q_path, disable_pbar=True)
            all_idxs = [np.argwhere(selections == i) for i in range(np.max(selections))]
            matches = np.zeros(len(q_desc)).astype(np.int)
            scores = np.zeros(len(q_desc)).astype(np.float32)
            for i, idxs in enumerate(all_idxs):
                if idxs.size != 0:
                    if idxs.size == 1:
                        select_desc = np.array(q_desc)[idxs.squeeze()].squeeze()
                        D, I = self.indexes[i].search(select_desc[None, :].astype(np.float32), 1)
                        scores[idxs.squeeze()] = cosine_similarity(self.m_desc[i][I].squeeze()[None, :],
                                                                   select_desc[None, :]).diagonal().astype(np.float32)
                        matches[idxs.squeeze()] = D.squeeze().astype(np.int)
                    else:

                        select_desc = np.vstack([q_desc[i] for i in idxs.squeeze()])
                        D, I = self.indexes[i].search(np.array(select_desc).astype(np.float32), 1)
                        scores[idxs.squeeze()] = np.array([cosine_similarity(self.m_desc[i][I][j].squeeze()[None, :], desc[None, :]).diagonal().astype(np.float32) for j, desc in enumerate(select_desc)]).squeeze()
                        matches[idxs.squeeze()] = D.squeeze().astype(np.int)
            return np.array(matches), np.array(scores)
    def match(self, q_desc):
        D, I = self.index.search(q_desc.astype(np.float32), 1)
        scores = cosine_similarity(self.m_desc[I].squeeze(), q_desc).diagonal()
        return I.squeeze(), scores