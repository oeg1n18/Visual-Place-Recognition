from vpr.data.datasets import Nordlands, SFU
import config
from vpr.vpr_techniques.techniques.switchCNN.models import mobilenetModule, resnet9Module
from vpr.vpr_techniques.techniques.switchCNN.train import resnet_transforms_test
from vpr.vpr_techniques import switchCNN
import pandas as pd
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

dataset = SFU

df = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')

selection_model_pth = "/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/switchCNN/weights/resnet9-epoch=1232-val_loss=0.00-other_metric=0.00.ckpt"
selection_transform = resnet_transforms_test
selection_model = resnet9Module.load_from_checkpoint(checkpoint_path=selection_model_pth)
selection_model.eval()

def get_matches(Q, M, method):
    m_desc = method.compute_map_features(M)
    q_desc = method.compute_query_desc(Q)
    S = method.matching_method(q_desc, m_desc)
    matches = np.argmax(S, axis=0).flatten()
    return matches

from vpr.vpr_techniques import hog, mixvpr, cosplace, netvlad

Q = dataset.get_query_paths()[:120]
M = dataset.get_map_paths()
GT = dataset.get_gtmatrix()


switchcnn_match = get_matches(Q, M, switchCNN)
hog_match = get_matches(Q, M, hog)
mixvpr_match = get_matches(Q, M, mixvpr)
cosplace_match = get_matches(Q, M, cosplace)
netvlad_match = get_matches(Q, M, netvlad)

X = torch.stack([resnet_transforms_test(Image.open(pth)) for pth in Q])
probs = F.sigmoid(selection_model(X.to(config.device))).detach().cpu().numpy()
desc, selections = switchCNN.compute_query_desc(Q)


gt = [df.loc[df["query_images"]==q].to_numpy()[0][2:] for q in Q]

counts = [0, 0, 0, 0, 0]
for i in range(len(Q)):

    print("probs", (probs[i]+0.5).astype(int), "good_selection", gt[i][selections[i]], "gt", gt[i], "switchCNN", GT[switchcnn_match[i], i], "nv, hog, cp, mv", GT[netvlad_match[i], i], GT[hog_match[i], i],  GT[cosplace_match[i], i],  GT[mixvpr_match[i], i])
    counts[0] += GT[switchcnn_match[i], i]
    counts[1] += GT[netvlad_match[i], i]
    counts[2] += GT[hog_match[i], i]
    counts[3] += GT[cosplace_match[i], i]
    counts[4] += GT[mixvpr_match[i], i]

    print(counts)