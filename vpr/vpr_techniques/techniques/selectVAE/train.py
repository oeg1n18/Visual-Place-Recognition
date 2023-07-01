from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from vpr.vpr_techniques.techniques.selectVAE.models import ConvVAE
import config
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
from PIL import Image
from torchvision import transforms

import pytorch_lightning as pl
from torch import nn
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(244, 244), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class vaeDataset(Dataset):
    def __init__(self, technique_name: str = "NetVLAD", transform=None):
        pth = config.root_dir + '/vpr/vpr_techniques/techniques/selectVAE/data/'
        with open(pth + technique_name + '_dataset.pkl', 'rb') as f:
            self.images = pickle.load(f)

        if len(self.images) < 5000:
            self.images = [Image.open(img) for img in self.images]
            if transform is not None:
                self.images = [transform(img) for img in self.images]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if len(self.images) > 5000:
            img = Image.open(self.images[idx])
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            return self.images[idx]



class VAEDataModule(pl.LightningDataModule):
    def __init__(self, technique_name: str = "NetVLAD", transform=None, batch_size: int = 32):
        super().__init__()
        self.technique_name = technique_name
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: str):
        self.train_dataset = vaeDataset(self.technique_name, transform=self.transform)
        self.val_dataset = vaeDataset(self.technique_name, transform=self.transform)
        self.test_dataset = vaeDataset(self.technique_name, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)


technique_names = ["NetVLAD", "HOG", "CosPlace", "MixVPR"]



if __name__ == '__main__':
    for technique_name in technique_names:
        dm = VAEDataModule(technique_name="NetVLAD", transform=transform)
        model = ConvVAE(latent_dim=1024, input_channels=3)
        checkpoint_callback = ModelCheckpoint(dirpath="weights/",
                                              filename=technique_name + '_weights',
                                              monitor="train_loss", save_top_k=1,
                                              mode='min')
        tb_logger = TensorBoardLogger(save_dir="logs/")
        trainer = pl.Trainer(max_epochs=200, accelerator='gpu', logger=tb_logger, callbacks=[checkpoint_callback])
        trainer.fit(model=model, datamodule=dm)
        trainer.test(model=model, datamodule=dm)