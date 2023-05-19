import PIL.Image as Image
from torch.utils.data import DataLoader

from src.vpr_techniques.python.techniques.netvlad import netvlad
import torchvision.models as models
import torch.nn as nn
from os.path import join, exists, isfile
from src.data.utils import VprDataset
import argparse
import torch
from torchvision import transforms
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 5
M = None


transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])

encoder = models.vgg16(pretrained=True)
layers = list(encoder.features.children())[:-2]
encoder = nn.Sequential(*layers)
net_vlad = netvlad.NetVLAD(num_clusters=64, dim=512, vladv2=False)
model = nn.Module()
model.add_module('encoder', encoder)
model.add_module('pool', net_vlad)

resume_ckpt = join('vpr_techniques/python/techniques/netvlad/dataPath/May16_11-08-14_vgg16_netvlad', 'checkpoints', 'model_best.pth.tar')
if isfile(resume_ckpt):
    print("Loading Model")
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

def compute_query_desc(Q):
    model.eval()
    if len(Q) > batch_size:
        ds = VprDataset(Q, transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        q_desc = []
        for i, imgs in enumerate(dl):
            img_embed = model.encoder(imgs.to(device))
            vlad_embed = model.pool(img_embed).detach().cpu().numpy()
            q_desc.append(vlad_embed)
        q_desc = np.vstack(q_desc).astype('float32')
    else:
        imgs = torch.stack([transform(Image.open(img_path)) for img_path in Q])
        img_embed = model.encoder(imgs.to(device))
        q_desc = model.pool(img_embed).detach().cpu().numpy().astype('float32')
    return q_desc

def compute_map_features(M):
    model.eval()
    if len(M) > batch_size:
        ds = VprDataset(M, transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        m_desc = []
        for i, imgs in enumerate(dl):
            img_embed = model.encoder(imgs.to(device))
            vlad_embed = model.pool(img_embed).detach().cpu().numpy()
            m_desc.append(vlad_embed)
        m_desc = np.vstack(m_desc).astype('float32')
    else:
        imgs = torch.stack([transform(Image.open(img_path)) for img_path in M])
        img_embed = model.encoder(imgs.to(device))
        m_desc = model.pool(img_embed).detach().cpu().numpy().astype('float32')
    return m_desc

def perform_vpr(q_path, M):
    model.eval()
    q_img = Image.open(q_path)
    q_img = transform(q_img)[None, :]
    img_embed = model.encoder(q_img.to(device))
    q_desc = model.pool(img_embed).detach().cpu().numpy()
    S = np.matmul(q_desc, M.T)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return j, S[i, j]




