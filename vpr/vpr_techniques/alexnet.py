import numpy as np
from typing import List
from abc import ABC, abstractmethod

from sklearn.metrics.pairwise import cosine_similarity
import faiss
import config
from vpr.data.utils import VprDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from vpr.vpr_techniques.utils import save_descriptors

class FeatureExtractor(ABC):

    @abstractmethod
    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        pass

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224, 224), antialias=True)
])

# preprocess images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


import torch
from torchvision import transforms

nDims = 4096
# load alexnet
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

# select conv3
model = model.features[:7]
model.to(config.device)
model.eval()

NAME = "alexnet"
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    ds = VprDataset(Q, transform=transform)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    all_desc = []
    for batch in tqdm(dl, desc="Computing query Descriptors", disable=disable_pbar):
        with torch.no_grad():
            output = model(batch.to(config.device)).detach().cpu().numpy()
        Ds = output.reshape([batch.shape[0], -1])
        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
        Ds = Ds @ Proj
        all_desc.append(Ds)
    q_desc = np.vstack(all_desc)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc

def compute_map_features(M, dataset_name=None, disable_pbar=False):
    ds = VprDataset(M, transform=transform)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    all_desc = []
    for batch in tqdm(dl, desc="Computing query Descriptors", disable=disable_pbar):
        with torch.no_grad():
            output = model(batch.to(config.device)).detach().cpu().numpy()
        Ds = output.reshape([batch.shape[0], -1])
        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
        Ds = Ds @ Proj
        all_desc.append(Ds)
    m_desc = np.vstack(all_desc)
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
