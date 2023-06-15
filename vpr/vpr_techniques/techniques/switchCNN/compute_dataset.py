from vpr.vpr_techniques import cosplace
from vpr.data.datasets import StLucia, SFU, GardensPointWalking
import config
import numpy as np

import pandas as pd

datasets = [StLucia, SFU, GardensPointWalking]

def compute_accuracy(technique):
    images = []
    results = []
    for dataset in datasets:
        M = dataset.get_map_paths(rootdir=config.root_dir)
        Q = dataset.get_query_paths(rootdir=config.root_dir)
        GTsoft = dataset.get_gtmatrix(rootdir=config.root_dir, gt_type='soft')
        map_descriptors = technique.compute_map_features(M)
        for i, q_path in enumerate(Q):
            images.append(q_path)
            j = technique.perform_vpr(q_path, map_descriptors)[0]
            if GTsoft[i, j] == 1:
                results.append(1)
            else:
                results.append(0)
    return images, results


from vpr.vpr_techniques import cosplace
images, cosplace_results = compute_accuracy(cosplace)
del cosplace

from vpr.vpr_techniques import netvlad
_, netvlad_results = compute_accuracy(netvlad)
del netvlad

from vpr.vpr_techniques import mixvpr
_, mixvpr_results = compute_accuracy(mixvpr)
del mixvpr

from vpr.vpr_techniques import delf
_, delf_results = compute_accuracy(delf)
del delf

data = {"query_images": images,
        "cosplace": cosplace_results,
        "netvlad": netvlad_results,
        "delf": delf_results,
        "mixvpr": mixvpr_results}

df = pd.DataFrame.from_dict(data)
df = df.sample(frac = 1)
df.to_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')


