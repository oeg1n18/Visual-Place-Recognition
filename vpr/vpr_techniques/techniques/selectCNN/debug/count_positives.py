from vpr.data.datasets import Nordlands_passes
from vpr.vpr_techniques import selectCNN, mixvpr, cosplace, netvlad, hog
from vpr.vpr_techniques.utils import load_descriptors
from vpr.evaluate.matching import thresholding
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(3)
ds = Nordlands_passes

Q = ds.get_query_paths()
M = ds.get_map_paths()

q_desc, m_desc = load_descriptors(ds.NAME, selectCNN.NAME)
S = selectCNN.matching_method(q_desc, m_desc)
S = thresholding(S, thresh='auto')
selections = q_desc[1]
masks = [np.where(np.array(selections)==i) for i in range(4)]
sel_preds = [S[:, mask[0]] for mask in masks]
for pred in sel_preds:
    print(pred.shape, pred[0][:10])
average_number_positives = [np.sum(preds)/preds.shape[1] for preds in sel_preds]
#ax[1].set_title("selectCNN Similarity Matrix")
print(average_number_positives)


ax[0].imshow(S)


q_desc, m_desc = load_descriptors(ds.NAME, mixvpr.NAME)
S = mixvpr.matching_method(q_desc, m_desc)
preds = thresholding(S, thresh='auto')
ax[1].imshow(preds)
#ax[1].set_title("MixVPR Similarity Matrix")


q_desc, m_desc = load_descriptors(ds.NAME, cosplace.NAME)
S = cosplace.matching_method(q_desc, m_desc)
preds = thresholding(S, thresh='auto')
ax[2].imshow(preds)
#ax[2].set_title("Cosplace Similarity Matrix")


plt.tight_layout()
plt.show()
