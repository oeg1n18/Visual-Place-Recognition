
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from vpr.vpr_techniques import switchCNNprec
from vpr.vpr_techniques.techniques.selectCNN.models import resnet9Module
from vpr.vpr_techniques.techniques.selectCNN.train_accuracy import resnet_transforms_test
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from vpr.data.datasets import SPED_V2, GardensPointWalking, Nordlands

dataset = SPED_V2

Q = dataset.get_query_paths()
selections = switchCNN.compute_query_desc(Q)[1]

hist = np.histogram(selections, bins=[0, 1, 2, 3, 4])
plt.bar(np.arange(4), hist[0])
plt.xticks(np.arange(4), ["NetVLAD", "Hog", "CosPlace", "MixVPR"])
plt.title("Selection Preferece for Dataset: " + dataset.NAME)
plt.show()