import pandas as pd
import numpy as np
from vpr.data.datasets import Nordlands
from vpr.vpr_techniques import hog, netvlad, mixvpr, cosplace
import config

method = hog
dataset = Nordlands

df = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')


Q = dataset.get_query_paths()
M = dataset.get_map_paths()
GTmat = dataset.get_gtmatrix()
GT = [df.loc[df["query_images"] == q].to_numpy()[0][2:] for q in Q]
q_desc = method.compute_query_desc(Q)
m_desc = method.compute_map_features(M)
S = method.matching_method(q_desc, m_desc)
matches = np.argmax(S, axis=0).flatten()
q_desc2 = method.compute_query_desc(Q)
m_desc2 = method.compute_map_features(M)
S2 = method.matching_method(q_desc2, m_desc2)
matches2 = np.argmax(S2, axis=0).flatten()


for i in range(len(GT)):
    print(GTmat[matches[i], i], GTmat[matches2[i], i], GT[i][1])