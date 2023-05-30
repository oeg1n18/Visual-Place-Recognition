import os

from torch.utils.data import DataLoader

from src.data.utils import VprDataset
from src.vpr_techniques.techniques.mixvpr.main import VPRModel
import torch
import src.config as config
from torchvision.models import ResNet50_Weights
from PIL import Image
import tqdm
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm


# Note that images must be resized to 320x320
model = VPRModel(backbone_arch='resnet50',
                 layers_to_crop=[4],
                 agg_arch='mixvpr',
                 agg_config={'in_channels' : 1024,
                             'in_h' : 20,
                             'in_w' : 20,
                             'out_channels' : 1024,
                             'mix_depth' : 4,
                             'mlp_ratio' : 1,
                             'out_rows' : 4},
                ).to(config.device)

weight_pth = config.root_dir + '/src/vpr_techniques/techniques/mixvpr/weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'

state_dict = torch.load(weight_pth)
model.load_state_dict(state_dict)
model.eval()


preprocess = torch.nn.Sequential(
    ResNet50_Weights.DEFAULT.transforms(),
    transforms.Resize(320))


NAME = 'mixvpr'


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
        pth = config.root_dir + '/src/descriptors/' + dataset_name
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + NAME):
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', q_desc)
            else:
                os.mkdir(pth + '/' + NAME)
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', q_desc)

    return q_desc





def compute_map_features(M, dataset_name=None):
    if len(M) > config.batch_size:
        dl = DataLoader(VprDataset(M, transform=preprocess), batch_size=config.batch_size)
        pbar = tqdm(dl)
        all_desc = []
        for batch in pbar:
            pbar.set_description("Computing Map Descriptors")
            all_desc.append(model(batch.to(config.device)).detach().cpu().numpy())
        m_desc = np.vstack(all_desc)
    else:
        imgs = torch.stack([preprocess(Image.open(pth)) for pth in Q])
        m_desc = model(imgs.to(config.device)).detach().cpu().numpy()

    if dataset_name is not None:
        pth = config.root_dir + '/src/descriptors/' + dataset_name
        if os.path.exists(pth):
            if os.path.exists(pth + '/' + NAME):
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', m_desc)
            else:
                os.mkdir(pth + '/' + NAME)
                np.save(config.root_dir + '/src/descriptors/' + dataset_name
                        + '/' + NAME + '/q_desc.npy', m_desc)

    return m_desc



def perform_vpr(q_path, M):
    img = preprocess(Image.open(q_path))
    q_desc = model(img.to(config.device))
    q_desc = (q_desc / np.linalg.norm(q_desc) + 1e-8)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / norms
    S = np.matmul(q_desc, M.T)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return j, S[i, j]


