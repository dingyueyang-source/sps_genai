"""GAN architectures for Assignment 3 (MNIST, 1 x 28 x 28 images)."""

import torch
from torch import nn


LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.reshape(-1, 128, 7, 7)
        return self.network(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    with torch.no_grad():
        fake = generator(torch.randn(4, LATENT_DIM))
        predictions = discriminator(fake)
    print("Generator output:", tuple(fake.shape))
    print("Discriminator output:", tuple(predictions.shape))
    assert fake.shape == (4, 1, 28, 28)
    assert predictions.shape == (4, 1)
    print("Both architecture checks passed.")
