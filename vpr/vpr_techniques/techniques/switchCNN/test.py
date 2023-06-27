from torch.utils.data import DataLoader

from models import resnet9Module
from train_accuracy import accuracyDataModule, accuracyDataset
import pytorch_lightning as pl
from vpr.vpr_techniques.techniques.switchCNN.train_accuracy import resnet_transforms_test
from vpr.data.datasets import Nordlands
import torch.nn.functional as F
import torch
from PIL import Image
import pandas as pd
import numpy as np

model_pth = "/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/switchCNN/weights/resnet9-epoch=632-val_loss=0.00-other_metric=0.00.ckpt"
model = resnet9Module.load_from_checkpoint(model_pth)


def lightning_test():
    dm = accuracyDataModule()
    dm.setup("train")
    trainer = pl.Trainer()
    trainer.test(model, dm.train_dataloader())



lightning_test()
