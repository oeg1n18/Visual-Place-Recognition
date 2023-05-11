import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from data.utils import VprDataset
from torch.utils.data import DataLoader


class ResNet:
    def __init__(self, device='cpu', batch_size=64):
        self.device = device
        self.resnet = self.get_model()
        self.resnet.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def resnet_descriptors(self, imgs):
        features = self.resnet(imgs.to(self.device))
        features = features.view(imgs.shape[0], -1).detach().cpu().numpy()
        return features

    def get_model(self):
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        resnet_extractor = nn.Sequential(*modules)
        return resnet_extractor


class VPR(ResNet):
    def __init__(self, device='cpu', batch_size=128):
        super().__init__(device=device)
        self.M = None
        self.batch_size = batch_size

    def compute_query_desc(self, Q):
        if len(Q) > self.batch_size:
            ds = VprDataset(Q, transform=self.preprocess)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            q_desc = np.vstack([self.resnet_descriptors(img_batch) for img_batch in dl])
        else:
            imgs = torch.stack([self.preprocess(Image.open(img_path)) for img_path in Q])
            q_desc = self.resnet_descriptors(imgs.to(self.device))
        q_desc = q_desc / np.linalg.norm(q_desc, axis=1, keepdims=True)
        return q_desc

    def compute_map_desc(self, M):
        if len(M) > self.batch_size:
            ds = VprDataset(M, transform=self.preprocess)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            m_desc = np.vstack([self.resnet_descriptors(img_batch) for img_batch in dl])
        else:
            imgs = torch.stack([self.preprocess(Image.open(img_path)) for img_path in M])
            m_desc = self.resnet_descriptors(imgs.to(self.device))
        m_desc = m_desc / np.linalg.norm(m_desc, axis=1, keepdims=True)
        self.M = m_desc
        return m_desc

    def perform_vpr(self, q_path):
        q_img = Image.open(q_path)
        q_img = self.preprocess(q_img)[None, :]
        m_desc = self.resnet_descriptors(q_img)
        m_desc = m_desc / np.linalg.norm(m_desc, axis=1, keepdims=True)
        S = np.matmul(m_desc, self.M.T)
        i, j = np.unravel_index(S.argmax(), S.shape)
        return j, S[i, j]

    def similarity_matrix(self, Q):
        q_desc = self.compute_query_desc(Q)
        if not type(self.M) == type(np.zeros((3,3))):
            raise Exception("Must first compute match features")
        return np.matmul(q_desc, self.M.T)
