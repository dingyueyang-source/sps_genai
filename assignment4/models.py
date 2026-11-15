import math

import torch
from torch import nn
from torch.nn import functional as F


def time_embedding(timesteps, dimension=64):
    """Match the sinusoidal time embedding in Module 8 Practical 3."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, dimension % 2))


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.time = nn.Linear(64, 2 * out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, t):
        hidden = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.time(t)[:, :, None, None].chunk(2, dim=1)
        hidden = self.norm2(self.conv2(hidden))
        return self.skip(x) + F.silu(hidden * (1 + scale) + shift)


class DiffusionUNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.enc1 = TimeBlock(width, width)
        self.enc2 = TimeBlock(width, 2 * width)
        self.middle = TimeBlock(2 * width, 3 * width)
        self.dec2 = TimeBlock(5 * width, 2 * width)
        self.dec1 = TimeBlock(3 * width, width)
        self.output = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x, timesteps):
        t = self.time_mlp(time_embedding(timesteps))
        skip1 = self.enc1(self.input(x), t)
        skip2 = self.enc2(F.avg_pool2d(skip1, 2), t)
        hidden = self.middle(F.avg_pool2d(skip2, 2), t)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.dec2(torch.cat((hidden, skip2), dim=1), t)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        return self.output(self.dec1(torch.cat((hidden, skip1), dim=1), t))


class Diffusion(nn.Module):
    def __init__(self, timesteps=100, width=24):
        super().__init__()
        self.denoiser = DiffusionUNet(width)
        self.timesteps = timesteps
        # Cosine schedule for the cumulative signal power.
        grid = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        signal = torch.cos((grid + 0.008) / 1.008 * (math.pi / 2)).square()
        betas = (1 - signal[1:] / signal[:-1]).clamp(max=0.999).float()
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))

    def loss(self, clean):
        t = torch.randint(self.timesteps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        signal = self.alpha_bar[t].view(-1, 1, 1, 1)
        noisy = signal.sqrt() * clean + (1 - signal).sqrt() * noise
        return F.mse_loss(self.denoiser(noisy, t), noise)

    @torch.no_grad()
    def sample(self, count, steps, rng):
        # Deterministic DDIM updates along a subset of the training timesteps.
        device = self.alpha_bar.device
        images = torch.randn((count, 3, 32, 32), device=device, generator=rng)
        times = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        for index, t in enumerate(times):
            t_batch = torch.full((count,), t, device=device, dtype=torch.long)
            prediction = self.denoiser(images, t_batch)
            alpha = self.alpha_bar[t]
            clean = ((images - (1 - alpha).sqrt() * prediction) / alpha.sqrt())
            clean = clean.clamp(-1, 1)
            if index == len(times) - 1:
                return clean
            # Recompute epsilon after clipping x0 to keep the two estimates consistent.
            prediction = (images - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            next_alpha = self.alpha_bar[times[index + 1]]
            images = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * prediction


class EnergyModel(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(4 * 4 * 2 * width, 1),
        )

    def forward(self, images):
        return self.energy(images).squeeze(1)


def uniform_images(count, device, rng=None):
    return torch.rand((count, 3, 32, 32), device=device, generator=rng) * 2 - 1


@torch.enable_grad()
def langevin(model, images, steps, step_size=8.0, noise_std=0.01, rng=None):
    # Restart the graph each step and differentiate energy only with respect to x.
    for _ in range(steps):
        images = images.detach().requires_grad_(True)
        gradient = torch.autograd.grad(model(images).sum(), images)[0]
        with torch.no_grad():
            noise = torch.randn(images.shape, device=images.device, generator=rng)
            images = (images - step_size * gradient + noise_std * noise).clamp(-1, 1)
    return images.detach()
