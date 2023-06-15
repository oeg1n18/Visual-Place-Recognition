import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint

import config
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from vpr.vpr_techniques.techniques.switchCNN.models import mobilenetModule
from pytorch_lightning.loggers import TensorBoardLogger



mobilenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class accuracyDataset(Dataset):
    def __init__(self, pd_dataset, transform=None, device='cuda'):
        self.data = pd_dataset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        record = self.data.iloc[idx].to_numpy()
        img = Image.open(record[1])
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(record[2:].astype(np.float32))
        return img.to(self.device), y.to(self.device)


class accuracyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32,):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        data = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')
        max_idx = len(data.index)
        self.train_ds = accuracyDataset(data[:int(max_idx*0.8)], transform=mobilenet_transform)
        self.val_ds = accuracyDataset(data[int(max_idx*0.8):], transform=mobilenet_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


datamodule = accuracyDataModule()
mobilenetModel = mobilenetModule()
checkpoint_callback = ModelCheckpoint(dirpath="weights/",
                                      filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                                      monitor="val_loss", save_top_k=1,
                                      mode='min')

tb_logger = TensorBoardLogger(save_dir="logs/")
trainer = pl.Trainer(max_epochs=30, accelerator='gpu', logger=tb_logger, callbacks=[checkpoint_callback])
trainer.fit(model=mobilenetModel, datamodule=datamodule)
trainer.test(model=mobilenetModel, datamodule=datamodule)
