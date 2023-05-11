import data.datasets.Nordlands as Nordlands
import data.datasets.GardensPointWalking as GardensPointWalking
import torch
from torch.utils.data import Dataset
import PIL.Image as Image


class DataModule:
    def __init__(self, dataset='Nordlands', session_type='ms', gt_type='hard'):

        assert gt_type in ['hard', 'soft']
        assert session_type in ['ss', 'ms']

        self.session_type = session_type
        self.gt_type = gt_type

        self.Q = self.get_query_paths(dataset)
        self.M = self.get_map_paths(dataset)
        self.GT = self.get_gtmatrix(dataset)

    def get_query_paths(self, dataset):
        if dataset == 'Nordlands':
            return Nordlands.get_query_paths(session_type=self.session_type)
        elif dataset == 'GardensPointWalking':
            return GardensPointWalking.get_query_paths(session_type=self.session_type)

    def get_map_paths(self, dataset):
        if dataset == 'Nordlands':
            return Nordlands.get_map_paths(session_type=self.session_type)
        elif dataset == 'GardensPointWalking':
            return GardensPointWalking.get_map_paths(session_type=self.session_type)

    def get_gtmatrix(self, dataset, gt_type='hard'):
        if dataset == 'Nordlands':
            return Nordlands.get_gtmatrix(session_type=self.session_type, gt_type=gt_type)
        elif dataset == 'GardensPointWalking':
            return GardensPointWalking.get_gtmatrix(session_type=self.session_type, gt_type=self.gt_type)


class VprDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img
