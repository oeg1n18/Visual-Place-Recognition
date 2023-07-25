
import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
import time
import torch
from vpr.data.utils import view_dataset_matches

import time
from sklearn.metrics.pairwise import cosine_similarity






u = np.random.rand(100, 5000)
M = np.random.rand(100, 5000)

sim = torch.nn.functional.cosine_similarity(torch.tensor(u), torch.tensor(M), dim=1)
print(sim.shape)