
from models import resnetModule
from train import accuracyDataModule
import pytorch_lightning as pl


selection_model = resnetModule.load_from_checkpoint(checkpoint_path="/home/ollie/Documents/Github/Visual-Place-Recognition/vpr/vpr_techniques/techniques/switchCNN/weights/epoch=1-val_loss=4.54-other_metric=0.00.ckpt")
dm = accuracyDataModule()
trainer = pl.Trainer()
trainer.test(model=selection_model, datamodule=dm)