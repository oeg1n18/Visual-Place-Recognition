import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from src.data.utils import VprDataset
from torch.utils.data import DataLoader


class ResNet:
    def __init__(self, device='cpu', batch_size=64):
        self.device = device
        self.resnet = self.get_model()
        self.resnet.to(self.device)

        self.method = 'resnet'

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



