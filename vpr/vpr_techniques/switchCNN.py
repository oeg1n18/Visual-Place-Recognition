import pytorch_lightning as pl
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from vpr.data.utils import VprDataset
from vpr.vpr_techniques.techniques.switchCNN.models import mobilenetModule
from vpr.vpr_techniques.techniques.switchCNN.train import mobilenet_transform
import torch
import numpy as np
from PIL import Image

selection_model = mobilenetModule.load_from_checkpoint(checkpoint_path="techniques/switchCNN/weights/***")


@torch.no_grad()
def compute_query_desc(Q, dataset_name=None):
    if len(Q) > config.batch_size:
        selection_dl = DataLoader(VprDataset(Q, transform=mobilenet_transform), batch_size=config.batch_size,
                                  shuffle=False)
        selections = []
        for batch in selection_dl:
            selections = torch.argmax(selection_model(batch.to(config.device)).detach().cpu(), dim=1)
            selections += list(np.array(selections))
        selections = np.array(selections)
    else:
        imgs = torch.stack([mobilenet_transform(Image.open(pth)) for pth in Q])
        selections = torch.argmax(selection_model(imgs.to(config.device)).detach().cpu(), dim=0)
        selections = np.array(selections)

    q_desc = list(np.zeros(len(Q)))
    cosplace_idx = np.argwhere(selections == 0)
    netvlad_idx = np.argwhere(selections == 1)
    delf_idx = np.argwhere(selections == 2)
    mixvpr_idx = np.argwhere(selections == 3)

    from vpr.vpr_techniques import cosplace
    select_queries = np.array(Q)[cosplace_idx]
    q_desc_cosplace = cosplace.compute_query_desc(list(select_queries))
    for i, desc in enumerate(q_desc_cosplace):
        q_desc[cosplace_idx.squeeze()[i]] = desc
    del cosplace

    from vpr.vpr_techniques import netvlad
    select_queries = np.array(Q)[netvlad_idx]
    q_desc_netvlad = netvlad.compute_query_desc(list(select_queries))
    for i, desc in enumerate(q_desc_netvlad):
        q_desc[netvlad_idx.squeeze()[i]] = desc
    del netvlad

    from vpr.vpr_techniques import delf
    select_queries = np.array(Q)[delf_idx]
    q_desc_delf = delf.compute_query_desc(list(select_queries))
    for i, desc in enumerate(q_desc_delf):
        q_desc[delf_idx.squeeze()[i]] = desc
    del delf

    from vpr.vpr_techniques import mixvpr
    select_queries = np.array(Q)[mixvpr_idx]
    q_desc_mixvpr = mixvpr.compute_query_desc(list(select_queries))
    for i, desc in enumerate(q_desc_mixvpr):
        q_desc[mixvpr_idx.squeeze()[i]] = desc
    del mixvpr

    return (q_desc, selections)


@torch.no_grad()
def compute_map_features(M, dataset_name=None):
    from vpr.vpr_techniques import cosplace
    m_desc_cosplace = cosplace.compute_map_features(M)
    del cosplace
    from vpr.vpr_techniques import netvlad
    m_desc_netvlad = netvlad.compute_map_features(M)
    del netvlad
    from vpr.vpr_techniques import delf
    m_desc_delf = delf.compute_map_features(M)
    del delf
    from vpr.vpr_techniques import mixvpr
    m_desc_mixvpr = mixvpr.compute_map_features(M)
    del mixvpr
    return (m_desc_cosplace, m_desc_netvlad, m_desc_delf, m_desc_mixvpr)


def matching_method(q_desc, m_desc):
    q_desc, select_idx = q_desc
    S = cosine_similarity(q_desc, m_desc[select_idx])
    return S


@torch.no_grad()
def perform_vpr(q_path, m_desc):
    q_desc = compute_query_desc([q_path])
    S = matching_method(q_desc, m_desc)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return int(j), float(S[i, j])
