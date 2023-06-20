from vpr.vpr_techniques import cosplace
from vpr.data.datasets import StLucia, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, Nordlands, RobotCars_short, pittsburgh30k
import config
import numpy as np
from tqdm import tqdm
import pandas as pd

datasets = [pittsburgh30k, StLucia, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, Nordlands, RobotCars_short]

def compute_accuracy(technique):
    images = []
    results = []
    for dataset in tqdm(datasets, desc="Compute on Datasets"):
        M = dataset.get_map_paths()
        Q = dataset.get_query_paths()
        GTsoft = dataset.get_gtmatrix(gt_type='soft')
        map_descriptors = technique.compute_map_features(M, disable_pbar=False)
        for i, q_path in enumerate(Q):
            images.append(q_path)
            j = technique.perform_vpr(q_path, map_descriptors)[0]
            if GTsoft[i, j] == 1:
                results.append(1)
            else:
                results.append(0)
    return images, results


print("============ patchNet-VLAD ================")
from vpr.vpr_techniques import patchnetvlad
_, patchnetvlad_results = compute_accuracy(patchnetvlad)
del patchnetvlad
print("============ CosPlace ================")
from vpr.vpr_techniques import cosplace
images, cosplace_results = compute_accuracy(cosplace)
del cosplace
print("============ netvlad ================")
from vpr.vpr_techniques import netvlad
_, netvlad_results = compute_accuracy(netvlad)
del netvlad
print("============ mixvpr ================")
from vpr.vpr_techniques import mixvpr
_, mixvpr_results = compute_accuracy(mixvpr)
del mixvpr
print("============ delf ================")
from vpr.vpr_techniques import delf
_, delf_results = compute_accuracy(delf)
del delf



data = {"query_images": images,
        "cosplace": cosplace_results,
        "netvlad": netvlad_results,
        "delf": delf_results,
        "mixvpr": mixvpr_results,
        "patchnetvlad": patchnetvlad_results}

df = pd.DataFrame.from_dict(data)
df = df.sample(frac = 1)
df.to_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')


