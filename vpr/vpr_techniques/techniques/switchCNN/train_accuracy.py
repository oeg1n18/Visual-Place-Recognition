import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint

import config
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from vpr.vpr_techniques.techniques.switchCNN.models import resnet9Module
from pytorch_lightning.loggers import TensorBoardLogger

resnet_transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),  # must same as here
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalization
])

resnet_transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),  # must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class accuracyDataset(Dataset):
    def __init__(self, pd_dataset, transform=None):
        self.data = pd_dataset
        self.transform = transform
        self.images = [Image.open(self.data.iloc[i].to_numpy()[1]) for i in range(len(self.data.index))]
        if transform:
            self.images = [transform(img) for img in self.images]

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        record = self.data.iloc[idx].to_numpy()
        img = self.images[idx]
        y = torch.tensor(record[2:].astype(np.float32))
        return img, y


class accuracyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 512):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str):
        data = pd.read_csv(config.root_dir + '/vpr/vpr_techniques/techniques/switchCNN/data/accuracy_dataset.csv')
        max_idx = len(data.index)
        self.train_ds = accuracyDataset(data, transform=resnet_transforms_test)
        self.val_ds = accuracyDataset(data[int(max_idx * 0.95):], transform=resnet_transforms_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    datamodule = accuracyDataModule()
    model = resnet9Module()
    checkpoint_callback = ModelCheckpoint(dirpath="weights/",
                                          filename='resnet9-{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                                          monitor="train_loss", save_top_k=1,
                                          mode='min')

    tb_logger = TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(max_epochs=2000, accelerator='gpu', logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
