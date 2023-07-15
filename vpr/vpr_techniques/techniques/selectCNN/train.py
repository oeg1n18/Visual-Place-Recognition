import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint

import config
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from vpr.vpr_techniques.techniques.selectCNN.models import resnet18Module
from pytorch_lightning.loggers import TensorBoardLogger

TRAIN_DATASET_PATH = '/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/data/Nordlands_passes_F-beta_train.csv'
VAL_DATASET_PATH = '/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/selectCNN/data/Nordlands_passes_F-beta_test.csv'
MAX_DATA_SIZE = 10000 # any datasets over this size will be read into ram at runtime, otherwise it will be preloaded


resnet_transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),  # must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class selectCNN_Dataset(Dataset):
    def __init__(self, pd_dataset, transform=None):
        self.data = pd_dataset
        self.transform = transform
        if len(self.data.index) < MAX_DATA_SIZE:
            self.images = [Image.open(self.data.iloc[i].to_numpy()[1]) for i in range(len(self.data.index))]
            self.images = [np.array(img).astype(np.uint8)[:, :, :3] for img in self.images]
            self.images = [Image.fromarray(img) for img in self.images]
        if transform and len(self.data.index) < MAX_DATA_SIZE:
            self.images = [transform(img) for img in self.images]

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        record = self.data.iloc[idx].to_numpy()
        if len(self.data.index) > MAX_DATA_SIZE:
            img = Image.fromarray(np.array(Image.open(record[1])).astype(np.uint8)[:, :, :3])
            if self.transform:
                img = self.transform(img)
        else:
            img = self.images[idx]
        y = torch.tensor(record[2:].astype(np.float32))
        return img, y


class TrainingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 512):
        super().__init__()
        self.batch_size = batch_size

    def get_output_spec(self):
        data = pd.read_csv(TRAIN_DATASET_PATH)
        record = data.iloc[0].to_numpy()
        data_type = None
        if isinstance(record[2:][0], float):
            data_type = 'continuous'
        elif isinstance(record[2:][0], np.int64):
            data_type = 'discrete'
        y = record[2:].astype(np.float32)
        return y.shape[0], data_type


    def setup(self, stage: str):
        train_data = pd.read_csv(TRAIN_DATASET_PATH)
        val_data = pd.read_csv(VAL_DATASET_PATH)
        self.train_ds = selectCNN_Dataset(train_data, transform=resnet_transforms_test)
        self.val_ds = selectCNN_Dataset(val_data, transform=resnet_transforms_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=12)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    datamodule = TrainingDataModule()
    size = datamodule.get_output_spec()
    model = resnet18Module(output_dim=size[0], output_type=size[1])
    checkpoint_callback = ModelCheckpoint(dirpath="weights/",
                                          filename=TRAIN_DATASET_PATH.split('/')[-1][:-4],
                                          monitor="val_loss", save_top_k=1,
                                          mode='min')

    tb_logger = TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(max_epochs=2000, accelerator='gpu', logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
