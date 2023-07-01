from vpr.vpr_techniques import netvlad, hog, cosplace, mixvpr
from vpr.data.datasets import pittsburgh30k, Nordlands
from vpr.evaluate.matching import best_match_per_query
import config
import numpy as np
import pickle
import importlib
from PIL import Image
datasets = [pittsburgh30k]
method_names = ["vpr.vpr_techniques.netvlad", "vpr.vpr_techniques.hog", "vpr.vpr_techniques.cosplace", "vpr.vpr_techniques.mixvpr"]


def build_dataset(dataset, method_name):
    technique = importlib.import_module(method_name)
    Q = dataset.get_query_paths()
    M = dataset.get_map_paths()
    GT = dataset.get_gtmatrix()

    q_desc = technique.compute_query_desc(Q)
    m_desc = technique.compute_map_features(M)

    S = technique.matching_method(q_desc, m_desc)

    import matplotlib.pyplot as plt
    plt.imshow(S)
    plt.show()
    del q_desc
    del m_desc

    preds = best_match_per_query(S)
    matches = np.argmax(preds, axis=0).flatten()

    # do not collect failed retreivals on distractor images
    distractors = np.sum(GT, axis=0).flatten().astype(int)
    D = np.zeros_like(distractors)
    D[distractors == 0] = 1

    failed_images = []
    # collect failed retrievals
    accuracy = []
    for i, q_idx in enumerate(range(len(Q))):
        accuracy.append(GT[matches[i], q_idx])
        if GT[matches[i], q_idx] != 1 and D[i] != 1:
            failed_images.append(Q[q_idx])

    pth = config.root_dir + '/vpr/vpr_techniques/techniques/selectVAE/data/'
    with open(pth + technique.NAME + '_dataset.pkl', 'wb') as f:
        pickle.dump(failed_images, f)
    print("accuracy", np.sum(accuracy)/len(Q))
    print("Number of failed Images on " + technique.NAME + ": ", len(failed_images))
    return None

# build and save the datasets
build_dataset(Nordlands, "vpr.vpr_techniques.netvlad")
build_dataset(Nordlands, "vpr.vpr_techniques.hog")
build_dataset(Nordlands, "vpr.vpr_techniques.cosplace")
build_dataset(Nordlands, "vpr.vpr_techniques.mixvpr")