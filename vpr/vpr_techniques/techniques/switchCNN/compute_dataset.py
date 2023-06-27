from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score

from vpr.vpr_techniques import cosplace, mixvpr, netvlad, hog
from vpr.data.datasets import StLucia, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, Nordlands, \
    RobotCars_short, pittsburgh30k
import config
import numpy as np
from vpr.evaluate.matching import thresholding
from tqdm import tqdm
import pandas as pd


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


def compute_f1(technique, datasets):
    all_images = []
    all_f1 = []
    for dataset in datasets:
        M = dataset.get_map_paths()
        Q = dataset.get_query_paths()
        GT = dataset.get_gtmatrix()
        map_descriptors = technique.compute_map_features(M, disable_pbar=False)
        query_descriptors = technique.compute_query_desc(Q, disable_pbar=True)
        S0 = technique.matching_method(query_descriptors, map_descriptors)
        all_images += Q
        for i, q in enumerate(query_descriptors):
            S = S0[:, i].flatten()
            gt = GT[:, i].flatten().astype(int)
            preds = thresholding(S, thresh='auto')
            all_f1.append(f1_score(gt, preds))
    return all_images, all_f1


def build_dataset(criteria, dataset_name, techniques, datasets):
    data = {}
    flag = 1
    for technique in techniques:
        images, results = criteria(technique, datasets)
        if flag == 1:
            data["query_images"] = images
            flag = 0
        data[technique.NAME] = results

    df = pd.DataFrame.from_dict(data)
    df = df.sample(frac=1)
    df.to_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/' + dataset_name)


datasets = [Nordlands, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, StLucia, RobotCars_short]
techniques = [netvlad, hog, cosplace, mixvpr]
criteria = compute_f1
dataset_name = "f1_dataset.csv"

build_dataset(criteria, dataset_name, techniques, datasets)
