import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import torch
import numpy as np

import config
from vpr.data.utils import VprDataset
from vpr.vpr_techniques.utils import save_descriptors
import faiss

NAME = "CosPlace"

model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to(
    config.device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    if len(Q) > config.batch_size:
        dl = DataLoader(VprDataset(Q, transform=preprocess), batch_size=config.batch_size)
        all_desc = []
        for batch in tqdm(dl, desc='Computing Query Descriptors', disable=disable_pbar):
            all_desc.append(model(batch.to(config.device)).detach().cpu().numpy())
        q_desc = np.vstack(all_desc)
    else:
        imgs = torch.stack([preprocess(Image.open(pth)) for pth in Q])
        q_desc = model(imgs.to(config.device)).detach().cpu().numpy()
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc


def compute_map_features(M, dataset_name=None, disable_pbar=False):
    if len(M) > config.batch_size:
        dl = DataLoader(VprDataset(M, transform=preprocess), batch_size=config.batch_size)
        all_desc = []
        for batch in tqdm(dl, desc='Computing Map Descriptors', disable=disable_pbar):
            all_desc.append(model(batch.to(config.device)).detach().cpu().numpy())
        m_desc = np.vstack(all_desc)
    else:
        imgs = torch.stack([preprocess(Image.open(pth)) for pth in M])
        m_desc = model(imgs.to(config.device)).detach().cpu().numpy()
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
            D, I = self.index.search(q_desc.astype(np.float32), 1)
            scores = cosine_similarity(self.m_desc[I].squeeze(), q_desc)
            return I.squeeze(), scores


def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc)
