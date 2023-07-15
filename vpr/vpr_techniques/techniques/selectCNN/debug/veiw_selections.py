from vpr.data.datasets import Nordlands, SFU, GardensPointWalking, Nordlands_passes
import config
from vpr.vpr_techniques.techniques.selectCNN.models import mobilenetModule, resnet9Module
from vpr.vpr_techniques.techniques.selectCNN.train_accuracy import resnet_transforms_test
from vpr.data.utils import VprDataset
from torch.utils.data import DataLoader
from vpr.vpr_techniques import selectCNN
import pandas as pd
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F

dataset = Nordlands_passes

df = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/selectCNN/data/nordlands_f1_dataset.csv')

selection_model_pth = "/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/weights/nordlands_f1-v3.ckpt"
selection_transform = resnet_transforms_test
selection_model = resnet9Module.load_from_checkpoint(checkpoint_path=selection_model_pth)
selection_model.eval()


Q = list(df["query_images"].to_numpy())[:100]

X = torch.stack([resnet_transforms_test(Image.open(q)) for q in Q])
selections = []
for batch in torch.chunk(X, int(max(len(Q) / config.batch_size, 1))):
    logits = selection_model(batch.to(config.device)).detach().cpu()
    sample_selections = logits
    selections += list(sample_selections)
selections = np.array(selections).flatten()

accuracy = 0
for i in range(len(Q)):
    select = np.argmax(np.array(selections[i]))
    if np.max(df.iloc[i].to_numpy()[2:]) == df.iloc[i].to_numpy()[2:][select]:
        gt = True
    else:
        gt = False

    if gt:
        accuracy += 1
    print("Preds: ", gt)

print("selection accuracy: ", accuracy/len(Q))
