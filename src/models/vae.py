import torch.nn as nn
import numpy as np
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


class EncoderCNN(nn.Module):
    def __init__(self, input_shape, hidden_dim, latent_dim):
        super(EncoderCNN, self).__init__()
        final_dim = int(input_shape[-1] / (2 ** 5))
        self.fc = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, 2, 1), # 1
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), # 2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # 3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), # 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), # 5
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(512 * final_dim * final_dim, hidden_dim),
            nn.Flatten(),
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var


class DecoderCNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_shape):
        super(DecoderCNN, self).__init__()
        self.final_dim = int(output_shape[-1] / (2 ** 5))
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.final_dim * self.final_dim),
        )
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), # 5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), # 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), # 3
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), # 2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, output_shape[0], 3, 2, 1, 1), # 1
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.input_layer(x)
        return self.fc(x.view(-1, 512, self.final_dim, self.final_dim))


class VAEModel(nn.Module):
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

    def loss_fn(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x.view(x_hat.size(0), -1), reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld

    def train_helper(self, x, _, __):
        x_hat, mean, log_var = self.forward(x)
        loss = self.loss_fn(x, x_hat, mean, log_var)
        loss.backward()
        return loss

    def show_images(self, cols=4, rows=4, device="cuda"):
        with torch.no_grad():
            noise = torch.randn(rows * cols, self.latent_dim).to(device)
            gen = self.decoder(noise).view(rows * cols, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        return gen.cpu()


class LinearModel(VAEModel):
    def __init__(self, image_shape, hidden_dim, latent_dim):
        super(LinearModel, self).__init__()
        self.image_shape = image_shape
        self.image_dim = np.prod(np.array(image_shape))
        self.encoder = Encoder(input_dim=self.image_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=self.image_dim)
        self.latent_dim = latent_dim


class CNNModel(VAEModel):
    def __init__(self, image_shape, hidden_dim, latent_dim):
        super(CNNModel, self).__init__()
        self.image_shape = image_shape
        self.image_dim = np.prod(np.array(image_shape))
        self.encoder = EncoderCNN(input_shape=self.image_shape, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = DecoderCNN(latent_dim=latent_dim, hidden_dim=hidden_dim, output_shape=self.image_shape)
        self.latent_dim = latent_dim