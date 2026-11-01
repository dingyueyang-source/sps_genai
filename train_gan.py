"""Train the Assignment 3 GAN on MNIST and save weights/sample images.

Run: uv run python train_gan.py --epochs 3
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from gan_model import LATENT_DIM, Discriminator, Generator

ROOT = Path(__file__).resolve().parent


def train(epochs: int = 3, batch_size: int = 128, lr: float = 0.0002,
          max_batches: int | None = None) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}", flush=True)

    dataset = datasets.MNIST(
        root=ROOT / "data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # images in [-1, 1] for Tanh
        ]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    output_dir = ROOT / "gan_samples"
    output_dir.mkdir(exist_ok=True)
    fixed_noise = torch.randn(64, LATENT_DIM, device=device)
    history = []

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_total = g_total = 0.0
        steps = 0
        bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for real_images, _ in bar:
            real_images = real_images.to(device)
            n = real_images.size(0)
            real_labels = torch.ones(n, 1, device=device)
            fake_labels = torch.zeros(n, 1, device=device)

            # Update D on real and generated digits.
            optimizer_d.zero_grad()
            noise = torch.randn(n, LATENT_DIM, device=device)
            fake_images = generator(noise)
            loss_d_real = criterion(discriminator(real_images), real_labels)
            loss_d_fake = criterion(discriminator(fake_images.detach()), fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # Update G to make the generated digits look real to D.
            optimizer_g.zero_grad()
            noise = torch.randn(n, LATENT_DIM, device=device)
            fake_images = generator(noise)
            loss_g = criterion(discriminator(fake_images), real_labels)
            loss_g.backward()
            optimizer_g.step()

            d_total += loss_d.item()
            g_total += loss_g.item()
            steps += 1
            if steps % 20 == 0:
                bar.set_postfix(d_loss=f"{loss_d.item():.3f}", g_loss=f"{loss_g.item():.3f}")
            if max_batches is not None and steps >= max_batches:
                break

        # Save a checkpoint after each epoch, in case training is interrupted.
        torch.save(generator.cpu().state_dict(), ROOT / "mnist_gan_generator.pth")
        generator.to(device)
        generator.eval()
        with torch.no_grad():
            generated = generator(fixed_noise).cpu()
        save_image(generated, output_dir / f"epoch_{epoch}.png", nrow=8, normalize=True, value_range=(-1, 1))
        record = {"epoch": epoch, "d_loss": d_total / steps, "g_loss": g_total / steps}
        history.append(record)
        (ROOT / "gan_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}: D loss={record['d_loss']:.4f}, G loss={record['g_loss']:.4f}", flush=True)

    print("Saved mnist_gan_generator.pth and generated samples in gan_samples/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional quick test: limit batches per epoch")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          max_batches=args.max_batches)
