import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from vpr.data.utils import VprDataset

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
    ds = VprDataset(Q, transform=preprocess)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=16)
    all_desc = []
    for batch in tqdm(dl, desc="Computing Query Descriptors", disable=disable_pbar):
        with torch.no_grad():
            desc = model(batch.to(config.device)).detach().cpu().numpy().squeeze()
            all_desc.append(desc.astype(np.float32))
        q_desc = np.vstack(all_desc).astype(np.float32)

    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc


def compute_map_features(M, dataset_name=None, disable_pbar=False):
    ds = VprDataset(M, transform=preprocess)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=16)
    all_desc = []
    for batch in tqdm(dl, desc="Computing Map Features", disable=disable_pbar):
        with torch.no_grad():
            desc = model(batch.to(config.device)).detach().cpu().numpy().squeeze()
        all_desc.append(desc.astype(np.float32))
    m_desc = np.vstack(all_desc).astype(np.float32)

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


