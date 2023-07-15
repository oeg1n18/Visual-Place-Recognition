import os

from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from vpr.data.utils import VprDataset
from vpr.vpr_techniques.techniques.mixvpr.main import VPRModel
import torch
import config as config
from torchvision.models import ResNet50_Weights
from PIL import Image
import tqdm
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import faiss
from torch.utils.data import Dataset

from vpr.vpr_techniques.utils import save_descriptors

# Note that images must be resized to 320x320
model = VPRModel(backbone_arch='resnet50',
                 layers_to_crop=[4],
                 agg_arch='mixvpr',
                 agg_config={'in_channels': 1024,
                             'in_h': 20,
                             'in_w': 20,
                             'out_channels': 1024,
                             'mix_depth': 4,
                             'mlp_ratio': 1,
                             'out_rows': 4},
                 ).to(config.device)

weight_pth = config.root_dir + '/vpr/vpr_techniques/techniques/mixvpr/weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'

state_dict = torch.load(weight_pth)
model.load_state_dict(state_dict)
model.eval()

preprocess = torch.nn.Sequential(
    ResNet50_Weights.DEFAULT.transforms(),
    transforms.Resize(320, antialias=True))

class MixVPRDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = np.array(Image.open(path)).astype(np.uint8)[:, :, :3]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img


NAME = 'MixVPR'


def compute_query_desc(Q, dataset_name=None, disable_pbar=False):
    model.eval()
    ds = MixVPRDataset(Q, transform=preprocess)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=16)
    all_desc = []
    for batch in tqdm(dl, desc="Computing Query Features", disable=disable_pbar):
        desc = model(batch.to(config.device)).detach().cpu()
        all_desc.append(desc)
    q_desc = np.vstack(all_desc)
    if dataset_name is not None:
        save_descriptors(dataset_name, NAME, q_desc, type='query')
    return q_desc


def compute_map_features(M, dataset_name=None, disable_pbar=False):
    model.eval()
    ds = MixVPRDataset(M, transform=preprocess)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=16)
    all_desc = []
    for batch in tqdm(dl, desc="Computing Map Features", disable=disable_pbar):
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
        self.index = faiss.IndexFlatIP(m_desc.shape[1])
        faiss.normalize_L2(m_desc)
        self.index.add(m_desc)

    def perform_vpr(self, q_path):
        q_desc = compute_query_desc(q_path)
        q_desc = q_desc.astype(np.float32)
        faiss.normalize_L2(q_desc)
        D, I = self.index.search(q_desc.astype(np.float32), 1)
        temp_mdesc = self.m_desc[I].squeeze() if self.m_desc[I].squeeze().ndim == 2 else self.m_desc[I][0]
        scores = cosine_similarity(q_desc, temp_mdesc).diagonal()
        return I.flatten(), scores
