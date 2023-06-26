from sklearn.metrics import precision_score, recall_score

from vpr.vpr_techniques import cosplace
from vpr.data.datasets import StLucia, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, Nordlands, RobotCars_short, pittsburgh30k
import config
import numpy as np
from tqdm import tqdm
import pandas as pd

datasets = [Nordlands, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, StLucia, RobotCars_short]
datasets = [ESSEX3IN1]
def compute_accuracy(technique):
    all_images = []
    all_results = []
    for dataset in datasets:
        M = dataset.get_map_paths()
        Q = dataset.get_query_paths()

        GT = dataset.get_gtmatrix()
        map_descriptors = technique.compute_map_features(M, disable_pbar=False)
        query_descriptors = technique.compute_query_desc(Q, disable_pbar=True)
        S = technique.matching_method(query_descriptors, map_descriptors)
        matches = np.argmax(S, axis=0).flatten()
        results = [GT[matches[i], i] for i in range(len(Q))]
        assert isinstance(results, list)
        all_images += Q
        all_results += results

    return all_images, all_results

def compute_recallAT100p(technique):
    all_images = []
    for dataset in datasets:
        M = dataset.get_map_paths()
        Q = dataset.get_query_paths()
        GT = dataset.get_gtmatrix()
        map_descriptors = technique.compute_map_features(M, disable_pbar=False)
        query_descriptors = technique.compute_query_desc(Q, disable_pbar=True)

        all_images += Q

        all_recallAt100 = []
        for i, q in enumerate(query_descriptors):
            recallAt100 = 0.
            S0 = technique.matching_method(q[None, :], map_descriptors).flatten()
            for t in np.flip(np.linspace(0, 1, 400)):
                S = np.copy(S0)
                S[S>t] = 1
                S[S<t] = 0
                prec = precision_score(GT[:, i].flatten().astype(int), S.flatten().astype(int))
                recal = recall_score(GT[:, i].flatten().astype(int), S.flatten().astype(int))
                print(prec, recal)
                if prec == 1.:
                    recallAt100 = recal
                if prec < 1.:
                    break
            all_recallAt100.append(recallAt100)
    return all_images, all_recallAt100

from vpr.vpr_techniques import mixvpr
images, metrics = compute_recallAT100p(mixvpr)
print(metrics)

"""

print("============ netvlad ================")
from vpr.vpr_techniques import netvlad
images, netvlad_results = compute_accuracy(netvlad)
print(images[:10], netvlad_results[:10])
del netvlad

print("============ cosplace ================")
from vpr.vpr_techniques import cosplace
_, cosplace_results = compute_accuracy(cosplace)
del cosplace
print("============ mixvpr ================")
from vpr.vpr_techniques import mixvpr
_, mixvpr_results = compute_accuracy(mixvpr)
del mixvpr


print("============ hog ================")
from vpr.vpr_techniques import hog
_, hog_results = compute_accuracy(hog)
del hog

print(len(images))
print(len(netvlad_results), len(hog_results), len(cosplace_results), len(mixvpr_results))

data = {"query_images": images,
        "netvlad": netvlad_results,
        "hog": hog_results,
        "cosplace": cosplace_results,
        "mixvpr": mixvpr_results}

df = pd.DataFrame.from_dict(data)
df = df.sample(frac = 1)
df.to_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')


"""