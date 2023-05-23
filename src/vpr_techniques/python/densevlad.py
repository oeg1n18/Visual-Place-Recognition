from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from src.data.datasets import GardensPointWalking
import os
from torch.utils.data import DataLoader
from src.data.utils import VprDataset
from tqdm import tqdm

BATCH_SIZE = 32
DESCRIPTOR_SIZE = 128
BASE_MODEL = 'resnet50'
RECLUSTER = False
CODEBOOK_SIZE = 100
NAME = 'DenseVLAD'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_ROOT = '/home/ollie/Documents/Github/Visual-Place-Recognition/src/'

def get_base_model(base_model='resnet50'):
    if base_model == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(model.children())[:-2]
        model = nn.Sequential(*modules)
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        return model, preprocess


def get_descriptors(feature_maps, desc_size):
    N, D, H, W = feature_maps.shape

    assert D % desc_size == 0, "Feature Map depth must be divisible by descriptor size. Map depth is " + str(D)
    feature_maps = feature_maps.flatten(start_dim=1)
    descriptors = feature_maps.view(-1, D * H * W // desc_size, desc_size)
    return descriptors.detach().cpu().numpy()


def cluster_descriptors(desc, codebook_size=100):
    if desc.ndim == 3:
        desc = desc.reshape(desc.shape[0] * desc.shape[1], desc.shape[2])
    print("=> Clustering " + str(desc.shape[0]) + " descriptors of size " + str(desc.shape[1]))
    kmeans = KMeans(n_clusters=codebook_size, random_state=0, n_init="auto").fit(desc)
    return kmeans.cluster_centers_


def compute_vlad(desc, centers, codebook_size=100):
    all_distances = np.stack([np.linalg.norm((desc - center), axis=1) for center in centers]).T
    codes = all_distances.argmin(axis=1)
    residuals = desc - centers[codes]
    masks = [(codes == i).astype(int) for i in range(codebook_size)]
    vlads = np.array([np.sum(mask[:, None] * residuals, axis=0) for mask in masks])
    vlads_centered = vlads - vlads.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vlads_centered, axis=1, keepdims=True)
    norms[norms==0.] = 1.
    vlads_centered_normed = vlads_centered/norms
    vlads_flattened = vlads_centered_normed.flatten()
    vlad = (vlads_flattened - vlads_flattened.mean())/vlads_flattened.std()
    norm = np.linalg.norm(vlad)
    norm = 1. if norm == 0. else norm
    return vlad/norm


def recluster(max_images=200):
    db_imgs = GardensPointWalking.get_map_paths(rootdir=PROJECT_ROOT)
    if len(db_imgs) > max_images:
        db_imgs = db_imgs[:max_images]
    if len(db_imgs) > BATCH_SIZE:
        dl = DataLoader(VprDataset(db_imgs, transform=preprocess), batch_size=BATCH_SIZE)
        feature_maps = torch.vstack([feature_extractor(batch) for batch in dl])
    else:
        imgs = torch.stack([preprocess(Image.open(pth)) for pth in db_imgs])
        feature_maps = feature_extractor(imgs)
    desc = get_descriptors(feature_maps, desc_size=DESCRIPTOR_SIZE)
    clusters = cluster_descriptors(desc, codebook_size=100)
    np.save(PROJECT_ROOT + "vpr_techniques/python/techniques/densevlad/clusters.npy", clusters)


## ===================================== VPR Functions ==============================================================
feature_extractor, preprocess = get_base_model(base_model=BASE_MODEL)


if RECLUSTER:
    recluster(max_images=200)

centers = np.load(PROJECT_ROOT + "vpr_techniques/python/techniques/densevlad/clusters.npy")

def compute_query_desc(Q):
    if len(Q) > BATCH_SIZE:
        dl = DataLoader(VprDataset(Q, transform=preprocess), batch_size=BATCH_SIZE)
        vlads = []
        pbar = tqdm(dl)
        for batch in pbar:
            pbar.set_description("Computing Query Descriptors")
            feature_maps = feature_extractor(batch)
            desc = get_descriptors(feature_maps, DESCRIPTOR_SIZE)
            vlad = np.vstack([compute_vlad(d, centers, codebook_size=CODEBOOK_SIZE) for d in desc])
            vlads.append(vlad)
        vlads = np.vstack(vlads)
        return vlads
    else:
        imgs = np.stack([preprocess(Image.open(pth)) for pth in Q])
        feature_maps = feature_extractor(imgs)
        desc = get_descriptors(feature_maps, DESCRIPTOR_SIZE)
        vlads = np.vstack([compute_vlad(d, centers, codebook_size=CODEBOOK_SIZE) for d in desc])
        return vlads





def compute_map_features(M):
    if len(M) > BATCH_SIZE:
        dl = DataLoader(VprDataset(M, transform=preprocess), batch_size=BATCH_SIZE)
        vlads = []
        pbar = tqdm(dl)
        for batch in pbar:
            pbar.set_description("Computing Map Descriptors")
            feature_maps = feature_extractor(batch)
            desc = get_descriptors(feature_maps, DESCRIPTOR_SIZE)
            vlad = np.vstack([compute_vlad(d, centers, codebook_size=CODEBOOK_SIZE) for d in desc])
            vlads.append(vlad)
        vlads = np.vstack(vlads)
        return vlads
    else:
        imgs = np.stack([preprocess(Image.open(pth)) for pth in M])
        feature_maps = feature_extractor(imgs)
        desc = get_descriptors(feature_maps, DESCRIPTOR_SIZE)
        vlads = np.vstack([compute_vlad(d, centers, codebook_size=CODEBOOK_SIZE) for d in desc])
        return vlads


def perform_vpr(q_path, M):
    img = preprocess(Image.open(q_path))
    feature_map = feature_extractor(img.to(device))
    q_desc = get_descriptors(feature_map, DESCRIPTOR_SIZE)
    q_desc = q_desc / np.linalg.norm(q_desc)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms==0.] = 1.
    M = M / norms
    S = np.matmul(q_desc, M.T)
    i, j = np.unravel_index(S.argmax(), S.shape)
    return j, S[i, j]


