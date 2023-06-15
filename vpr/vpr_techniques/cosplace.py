import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import torch
import numpy as np

import config
from vpr.data.utils import VprDataset
from vpr.vpr_techniques.utils import save_descriptors

NAME = "CosPlace"

model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to(config.device)

preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
def compute_query_desc(Q, dataset_name=None):
    if len(Q) > config.batch_size:
        dl = DataLoader(VprDataset(Q, transform=preprocess), batch_size=config.batch_size)
        pbar = tqdm(dl)
        all_desc = []
        for batch in pbar:
            pbar.set_description("Computing Query Descriptors")
            all_desc.append(model(batch.to(config.device)).detach().cpu().numpy())
        q_desc = np.vstack(all_desc)
    else:
        imgs = torch.stack([preprocess(Image.open(pth)) for pth in Q])
        q_desc = model(imgs.to(config.device)).detach().cpu().numpy()
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc

def compute_map_features(M, dataset_name=None):
    if len(M) > config.batch_size:
        dl = DataLoader(VprDataset(M, transform=preprocess), batch_size=config.batch_size)
        pbar = tqdm(dl)
        all_desc = []
        for batch in pbar:
            pbar.set_description("Computing Query Descriptors")
            all_desc.append(model(batch.to(config.device)).detach().cpu().numpy())
        m_desc = np.vstack(all_desc)
    else:
        imgs = torch.stack([preprocess(Image.open(pth)) for pth in M])
        m_desc = model(imgs.to(config.device)).detach().cpu().numpy()
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, m_desc, type='map')
    return m_desc


def perform_vpr(q_path, M):
    img = preprocess(Image.open(q_path))
    q_desc = model(img[None, :].to(config.device)).detach().cpu()
    S = np.matmul(q_desc, M.T)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return int(j), float(S[i, j])

matching_method = None