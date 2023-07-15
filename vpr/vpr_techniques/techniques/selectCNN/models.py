import pytorch_lightning as pl
import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet


class mobilenetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        modules = list(mobilenet.children())[:-1]
        self.fc_out = nn.Linear(62720, 5)
        self.model = nn.Sequential(*modules)
        self.model = nn.Sequential(self.model, nn.Flatten(), self.fc_out)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat[y_hat > 0.5] = 1.
        y_hat[y_hat <= 0.5] = 0.
        accuracy = torch.sum(y_hat == y) / y_hat.shape[0] * y_hat.shape[1]
        self.log("test_accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class resnet18Module(pl.LightningModule):
    def __init__(self, pretrained=True, output_dim=4, output_type='discrete'):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(512, output_dim)
        if output_type == 'discrete':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        correct = 0
        x, y = batch
        y_hat = self.model(x)
        y_pred = torch.argmax(y_hat)
        if torch.argmax(y) == y_pred:
            correct = 1
        else:
            correct = 0
        self.log("test_selection", correct, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)




def create_resnet9_model(output_dim: int = 1) -> nn.Module:
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    in_features = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(in_features, output_dim)
    return model



class resnet9regressionModule(pl.LightningModule):
    def __init__(self, pretrained=True, output_dim=4):
        super().__init__()
        self.model = create_resnet9_model(output_dim=output_dim)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test_mse", loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
