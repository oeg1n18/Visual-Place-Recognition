from vpr.vpr_techniques import netvlad, hog, cosplace, mixvpr
from vpr.data.datasets import StLucia, SFU, GardensPointWalking, ESSEX3IN1, SPED_V2, berlin_kudamm, Nordlands, RobotCars_short, pittsburgh30k
import config
import numpy as np
from tqdm import tqdm
import pandas as pd
from vpr.vpr_techniques.techniques.switchCNN.models import resnet9Module
from vpr.vpr_techniques.techniques.switchCNN.train_accuracy import resnet_transforms_test
import torch.nn.functional as F
from PIL import Image
techniques = [netvlad,hog,cosplace,mixvpr]
datasets = [Nordlands]



model_pth = ""
model = resnet9Module.load_from_checkpoint(model_pth)
df = pd.read_csv("/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv").to_numpy()
def compute_accuracy():
    images = []
    results = []
    for dataset in datasets:
        M = dataset.get_map_paths()
        Q = dataset.get_query_paths()
        GTsoft = dataset.get_gtmatrix(gt_type='soft')
        map_descriptors = [technique.compute_map_features(M, disable_pbar=False) for technique in techniques]
        prs = [technique.PlaceRecognition(map_descriptors[i]) for i, technique in enumerate(techniques)]

        for record in tqdm(df, desc='image matching'):
            q_path = record[1]
            results = record[2:]
            js = [pr.perform_vpr(q_path)[0] for pr in prs]
            gts = []
            for j in js:
                if GTsoft[Q.index(q_path), j] == 1:
                    gts.append(1)
                else:
                    gts.append(0)
            X = resnet_transforms_test(Image.open(q_path))
            probs = F.sigmoid(model(X[None, :].to('cuda'))).deatch().cpu().numpy()
            print("probs", probs, "GT_computed", gts, "GT_dataset", results)
    return None

compute_accuracy()