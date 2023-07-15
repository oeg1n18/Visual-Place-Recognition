from vpr.vpr_techniques import netvlad, cosplace, mixvpr, hog, switchCNNprec, selectCNN
from vpr.data.datasets import SPED_V2, Nordlands_passes
import matplotlib.pyplot as plt

ds = Nordlands_passes
Q = ds.get_query_paths()
M = ds.get_map_paths()
GT = ds.get_gtmatrix()

methods = [switchCNNf1, netvlad, hog, cosplace, mixvpr]

qs = [m.compute_query_desc(Q) for m in methods]
ms = [m.compute_map_features(M) for m in methods]
Ss = [m.matching_method(qs[i], ms[i]) for i, m in enumerate(methods)]

titles = ["switchCNNf1", "switchCNNprec", "NetVLAD", "HOG", "CosPlace", "MixVPR"]

fig, ax = plt.subplots(1, 6)

for i in range(6):
    ax[i].set_title(titles[i])
    ax[i].imshow(Ss[i])
    ax[i].set_axis_off()

plt.show()

