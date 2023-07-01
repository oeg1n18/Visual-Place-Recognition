import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(input_channels, 12, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 42, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(42, 52, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(52, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128, output_channels: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.conv1 = nn.ConvTranspose2d(12, output_channels, 3, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, 24, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(42, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(52, 42, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(128, 52, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fc1 = nn.Linear(latent_dim, 128 * 4 * 4)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv1(x))
        return x


class ConvVAE(pl.LightningModule):
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.enc = Encoder(input_channels=input_channels, latent_dim=latent_dim)
        self.dec = Decoder(output_channels=input_channels, latent_dim=latent_dim)
        self.criterion = nn.MSELoss()
    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def prob(self, img):
        mu, logVar = self.enc(img)
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        return kl_divergence

    def training_step(self, batch, batch_idx):
        mu, logVar = self.enc(batch)
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        z = self.reparameterize(mu, logVar)
        x_hat = self.dec(z)
        recon_loss = self.criterion(x_hat, batch)
        loss = recon_loss + 0.000001*kl_divergence
        self.log("recon_loss", recon_loss, on_epoch=True)
        self.log("kl_div_loss", kl_divergence, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

