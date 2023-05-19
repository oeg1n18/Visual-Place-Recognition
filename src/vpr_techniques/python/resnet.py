from torch.utils.data import DataLoader
import numpy as np
import torch
from techniques.resnet.resnet_base import ResNet, VprDataset
from PIL import Image

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

    def compute_map_features(self, M):
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
        q_desc = self.resnet_descriptors(q_img)
        q_desc = q_desc / np.linalg.norm(q_desc, axis=1, keepdims=True)
        S = np.matmul(q_desc, self.M.T)
        i, j = np.unravel_index(S.argmax(), S.shape)
        return j, S[i, j]
