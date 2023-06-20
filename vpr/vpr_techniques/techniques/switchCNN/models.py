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
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        modules = list(model.children())[:-1]
        self.fc_out = nn.Linear(num_features, 5)
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


class mlpModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.conv3 = nn.Conv2d(24, 36, 5)
        self.conv4 = nn.Conv2d(36, 48, 5)
        self.conv5 = nn.Conv2d(48, 64, 5)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        y_hat = self.model(x)
        y_hat[y_hat > 0.5] = 1.
        y_hat[y_hat <= 0.5] = 0.
        accuracy = torch.sum(y_hat == y) / y_hat.shape[0] * y_hat.shape[1]
        self.log("test_accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def create_resnet9_model(output_dim: int = 1) -> nn.Module:
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    in_features = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(in_features, output_dim)
    return model


class resnet9Module(pl.LightningModule):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = create_resnet9_model(output_dim=5)
        self.loss = nn.BCEWithLogitsLoss()

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
        y_hat = F.sigmoid(self.forward(x))
        y_hat[y_hat > 0.5] = 1.
        y_hat[y_hat <= 0.5] = 0.
        accuracy = torch.sum(y_hat == y) / y_hat.shape[0] * y_hat.shape[1]
        self.log("test_accuracy", accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class resnet9_regression_Module(pl.LightningModule):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = create_resnet9_model(output_dim=5)
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
        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class resnet18_regression_Module(pl.LightningModule):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        modules = list(model.children())[:-1]
        self.fc_out = nn.Linear(num_features, 5)
        self.model = nn.Sequential(*modules)
        self.model = nn.Sequential(self.model, nn.Flatten(), self.fc_out)
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
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

