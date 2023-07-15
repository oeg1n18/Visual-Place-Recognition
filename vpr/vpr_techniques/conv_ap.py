from sklearn.metrics.pairwise import cosine_similarity

from vpr.vpr_techniques.techniques.conv_ap.main import VPRModel
import torch
import config
import faiss
from torchvision import transforms
from vpr.data.utils import VprDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from vpr.vpr_techniques.utils import save_descriptors

# Note that these models have been trained with images resized to 320x320
# Also, either use BILINEAR or BICUBIC interpolation when resizing.
# The model with 4096-dim output has been trained with images resized with bicubic interpolation
# The model with 8192-dim output with bilinear interpolation
# ConvAP works with all image sizes, but best performance can be achieved when resizing to the training resolution

NAME = "Conv-AP"

model = VPRModel(backbone_arch='resnet50',
                 layers_to_crop=[],
                 agg_arch='ConvAP',
                 agg_config={'in_channels': 2048,
                             'out_channels': 512,
                             's1': 2,
                             's2': 2},
                 )

state_dict = torch.load(
    config.root_dir + '/vpr/vpr_techniques/techniques/conv_ap/weights/resnet50_ConvAP_512_2x2.ckpt')
model.load_state_dict(state_dict)
model.eval()
model.to(config.device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320), antialias=True),
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
        self.index = faiss.IndexFlatIP(m_desc.shape[1])
        faiss.normalize_L2(m_desc)
        self.index.add(m_desc)

    def perform_vpr(self, q_path):
        q_desc = compute_query_desc(q_path, disable_pbar=True)
        D, I = self.index.search(q_desc.astype(np.float32), 1)
        q_desc = q_desc.astype(np.float32)
        faiss.normalize_L2(q_desc)
        temp_mdesc = self.m_desc[I].squeeze() if self.m_desc[I].squeeze().ndim == 2 else self.m_desc[I][0]
        scores = cosine_similarity(q_desc, temp_mdesc).diagonal()
        return I.flatten(), scores
