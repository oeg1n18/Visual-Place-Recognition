from vpr.vpr_techniques import netvlad, cosplace, mixvpr, hog, switchCNN
from vpr.data.datasets import SPED_V2, Nordlands
import matplotlib.pyplot as plt

ds = SPED_V2
Q = ds.get_query_paths()
M = ds.get_map_paths()
GT = ds.get_gtmatrix()

methods = [switchCNN, netvlad, hog, cosplace, mixvpr]

qs = [m.compute_query_desc(Q) for m in methods]
ms = [m.compute_map_features(M) for m in methods]
Ss = [m.matching_method(qs[i], ms[i]) for i, m in enumerate(methods)]

titles = ["switchCNN", "NetVLAD", "HOG", "CosPlace", "MixVPR"]

fig, ax = plt.subplots(1, 5)

for i in range(5):
    ax[i].set_title(titles[i])
    ax[i].imshow(Ss[i])
    ax[i].set_axis_off()

plt.show()

