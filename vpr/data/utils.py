import vpr.data.datasets.GardensPointWalking as GardensPointWalking
from torch.utils.data import Dataset
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import random


class VprDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path)
        if img is not None:
            img = self.transform(img)
        return img


def view_dataset_matches(dataset):
    M = dataset.get_map_paths()
    Q = dataset.get_query_paths()
    GT = dataset.get_gtmatrix()
    print(len(Q), len(M))

    Q_sample = random.sample(Q, k=1)
    print(Q_sample)
    Q_sample_idx = Q.index(Q_sample[0])
    ref_idx = [[i] for i, x in enumerate(GT[:, Q_sample_idx]) if x == 1]
    n_correct_matches = len(ref_idx)
    print("Number of correct matches", n_correct_matches)
    M = np.array(M)
    Q_ref = M[ref_idx]
    if len(Q_ref) == 0:
        raise Exception("No reference Images")
    plt.imshow(np.array(Image.open(Q_sample[0])))
    plt.title("Query")
    plt.axis('off')
    plt.show()
    for i in range(0, max(n_correct_matches, 2)):
        plt.imshow(np.array(Image.open(Q_ref[i][0])))
        plt.title("Ref")
        plt.axis('off')
        plt.show()
