import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import Diffusion, EnergyModel, langevin, uniform_images


class CIFAR10(datasets.CIFAR10):
    # The U. Toronto download now redirects to this host.
    url = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dataset = CIFAR10(
        args.data, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    batches = iter(loader)
    diffusion = args.model == "diffusion"
    model = Diffusion().to(device) if diffusion else EnergyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if diffusion else 2e-4)
    replay = None if diffusion else uniform_images(4096, device)
    pending_losses = []
    save_path = args.output / f"{args.model}.pt"
    args.output.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        clean = next(batches, None)
        if clean is None:
            batches = iter(loader)
            clean = next(batches)
        clean = clean[0].to(device)
        if diffusion:
            loss = model.loss(clean)
        else:
            indices = torch.randperm(len(replay), device=device)[:len(clean)]
            refresh = torch.rand((len(clean), 1, 1, 1), device=device) < 0.05
            initial = torch.where(refresh, uniform_images(len(clean), device), replay[indices])
            negative = langevin(model, initial, args.langevin_steps)
            replay[indices] = negative
            real_energy = model(clean)
            fake_energy = model(negative)
            regularizer = 0.1 * (real_energy.square() + fake_energy.square()).mean()
            loss = (real_energy - fake_energy).mean() + regularizer
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {args.model} loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pending_losses.append(loss.item())

        if step == 1 or step % 50 == 0 or step == args.steps:
            mean_loss = sum(pending_losses) / len(pending_losses)
            print(f"{args.model} step {step}/{args.steps} mean loss {mean_loss:.5f}", flush=True)
            pending_losses.clear()
        if step == 1 or step % 100 == 0 or step == args.steps:
            temporary = save_path.with_suffix(".tmp")
            torch.save({
                "state_dict": model.state_dict(),
                "dataset": "CIFAR-10 training split",
                "train_steps": step,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }, temporary)
            temporary.replace(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("diffusion", "energy"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("checkpoints"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--langevin-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    train(parser.parse_args())
