from torch.utils.data import DataLoader

from models import resnet18Module
import pytorch_lightning as pl
from vpr.vpr_techniques.techniques.selectCNN import selectCNN_config
from vpr.vpr_techniques.techniques.selectCNN.train import TrainingDataModule



def lightning_test():
    model = resnet18Module.load_from_checkpoint(selectCNN_config.weights_path)
    dm = TrainingDataModule()
    dm.setup("train")
    trainer = pl.Trainer()
    trainer.test(model, dm.test_dataloader())



lightning_test()
