import numpy as np
from typing import List
from abc import ABC, abstractmethod

from sklearn.metrics.pairwise import cosine_similarity

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

class AlexNetConv3Extractor(FeatureExtractor):
    def __init__(self, nDims: int = 4096):
        import torch
        from torchvision import transforms

        self.nDims = nDims
        # load alexnet
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

        # select conv3
        self.model = self.model.features[:7]

        # preprocess images
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if torch.cuda.is_available():
            print('Using GPU')
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print('Using MPS')
            self.device = torch.device("mps")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")

        self.model.to(self.device)


    def compute_features(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        import torch

        imgs_torch = [self.preprocess(img) for img in imgs]
        imgs_torch = torch.stack(imgs_torch, dim=0)

        imgs_torch = imgs_torch.to(self.device)

        with torch.no_grad():
            output = self.model(imgs_torch)

        output = output.to('cpu').numpy()
        Ds = output.reshape([len(imgs), -1])

        rng = np.random.default_rng(seed=0)
        Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
        Proj = Proj / np.linalg.norm(Proj , axis=1, keepdims=True)

        Ds = Ds @ Proj

        return Ds



feature_extractor = AlexNetConv3Extractor()

NAME = "AlexNet"
def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    ds = VprDataset(Q, transform=transform)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    all_desc = []
    for batch in tqdm(dl, desc="Computing query Descriptors", disable=disable_pbar):
        desc = feature_extractor.compute_features(batch)
        all_desc.append(desc)
    q_desc = np.vstack(all_desc)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc

def compute_map_features(M, dataset_name=None, disable_pbar=False):
    ds = VprDataset(M, transform=transform)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    all_desc = []
    for batch in tqdm(dl, desc="Computing Map Descriptors", disable=disable_pbar):
        desc = feature_extractor.compute_features(batch)
        all_desc.append(desc)
    m_desc = np.vstack(all_desc)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc


def matching_method(q_desc, m_desc):
    return cosine_similarity(q_desc, m_desc).transpose()


class PlaceRecognition:
    def __init__(self, m_desc):
        self.m_desc = m_desc
        self.index = faiss.IndexFlatL2(m_desc.shape[1])
        self.index.add(m_desc)

    def perform_vpr(self, q_path):
        q_desc = compute_query_desc(q_path, disable_pbar=True)
        D, I = self.index.search(q_desc.astype(np.float32), 1)
        temp_mdesc = self.m_desc[I].squeeze() if self.m_desc[I].squeeze().ndim == 2 else self.m_desc[I][0]
        scores = cosine_similarity(q_desc, temp_mdesc).diagonal()
        return I.flatten(), scores
