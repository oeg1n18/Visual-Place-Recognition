from vpr.data.datasets import Nordlands, berlin_kudamm, SPED_V2
import config
from vpr.vpr_techniques.techniques.switchCNN.models import mobilenetModule, resnet9Module
from vpr.vpr_techniques.techniques.switchCNN.train_accuracy import resnet_transforms_test
from vpr.vpr_techniques import switchCNNprec, hog, netvlad, mixvpr, cosplace
import pandas as pd
import torch
from PIL import Image
import torch.nn.functional as F
from vpr.vpr_techniques.utils import load_descriptors
dataset = SPED_V2
import numpy as np

Q = dataset.get_query_paths()
M = dataset.get_map_paths()
GT = dataset.get_gtmatrix()

q_desc = switchCNN.compute_query_desc(Q)
m_desc = switchCNN.compute_map_features(M)
S = switchCNN.matching_method(q_desc, m_desc)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4)


ax[0].set_title("Control")
ax[0].imshow(S)
ax[0].set_xlabel("Queries")
ax[0].set_ylabel("References")
plt.axis('off')

#
ax[1].set_title("unit norm")
ax[1].imshow(S/np.linalg.norm(S, axis=0, keepdims=True))
ax[1].set_xlabel("Queries")
ax[1].set_ylabel("References")
plt.axis('off')


ax[2].set_title("Scaling")
mins = np.min(S, axis=0, keepdims=True)
max = np.max(S, axis=0, keepdims=True)
ax[2].imshow((S-mins)/(max - mins))
ax[2].set_xlabel("Queries")
ax[2].set_ylabel("References")
plt.axis('off')


ax[3].set_title("Normalization")
ax[3].imshow((S - np.mean(S, axis=0, keepdims=True))/np.std(S, axis=0, keepdims=True))
ax[3].set_xlabel("Queries")
ax[3].set_ylabel("References")
ax[3].set_axis_off()
ax[2].set_axis_off()
ax[1].set_axis_off()
ax[0].set_axis_off()
plt.show()

