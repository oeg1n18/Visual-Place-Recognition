import torch

#from vpr.vpr_techniques import conv_ap
#from vpr.data.datasets import Nordlands_passes
from vpr.vpr_techniques.utils import load_descriptors
import numpy as np
from fastdist import fastdist
from numba import jit
import time
#ds = Nordlands_passes
#method = conv_ap

x = np.random.rand(5000, 512)
y = np.random.rand(1000, 512)

from sklearn.metrics.pairwise import cosine_similarity
import time

st = time.time()
t = cosine_similarity(x, y)
sklearn_time = time.time() - st

print(t.shape)

import numpy as np
import numba

@numba.jit(target='cpu', nopython=True, parallel=True)
def fast_cosine_matrix(u, M):
    scores = np.zeros(M.shape[0])
    for i in numba.prange(M.shape[0]):
        v = M[i]
        m = u.shape[0]
        udotv = 0
        u_norm = 0
        v_norm = 0
        for j in range(m):
            if (np.isnan(u[j])) or (np.isnan(v[j])):
                continue

            udotv += u[j] * v[j]
            u_norm += u[j] * u[j]
            v_norm += v[j] * v[j]

        u_norm = np.sqrt(u_norm)
        v_norm = np.sqrt(v_norm)

        if (u_norm == 0) or (v_norm == 0):
            ratio = 1.0
        else:
            ratio = udotv / (u_norm * v_norm)
        scores[i] = ratio
    return scores


u = np.random.rand(100)
M = np.random.rand(100000, 100)

fast_cosine_matrix(u, M)
