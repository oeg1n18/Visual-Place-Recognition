import pytorch_lightning as pl
import torch
from torch import nn
from torch.functional import F

class mobilenetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        modules = list(mobilenet.children())[:-1]
        self.fc_out = nn.Linear(62720, 4)
        self.model = nn.Sequential(*modules)
        self.model = nn.Sequential(self.model, nn.Flatten(), self.fc_out, nn.Sigmoid())

    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        e = torch.abs(y_hat - y)
        accuracy = torch.mean(torch.sum(torch.sum(e, dim=1)))
        self.log("test_accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)