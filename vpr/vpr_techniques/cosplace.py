import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import torch
import numpy as np

import config
from vpr.vpr_techniques.utils import save_descriptors
import faiss

NAME = "CosPlace"

model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to(
    config.device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    X = torch.stack([preprocess(Image.open(q_pth)) for q_pth in Q])
    all_desc = []
    for batch in torch.chunk(X, int(max(X.shape[0] / config.batch_size, 1))):
        with torch.no_grad():
            desc = model(batch.to(config.device)).detach().cpu().numpy().squeeze()
        all_desc.append(desc)
    q_desc = np.vstack(all_desc)

    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc


def compute_map_features(M, dataset_name=None, disable_pbar=False):
    X = torch.stack([preprocess(Image.open(m_pth)) for m_pth in M]).to('cpu')
    all_desc = []
    for batch in torch.chunk(X, int(max(X.shape[0] / config.batch_size, 1))):
        with torch.no_grad():
            desc = model(batch.to(config.device)).detach().cpu().numpy().squeeze()
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


