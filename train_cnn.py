import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn_model import SimpleCNN


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    root = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR10(
        root=str(root / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root=str(root / "data"),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=32, shuffle=False
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    history = []
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            progress.set_postfix(
                loss=f"{total_loss / total:.4f}",
                accuracy=f"{100 * correct / total:.2f}%",
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_accuracy": 100 * correct / total,
        })

        torch.save(
            model.state_dict(), root / "cifar10_cnn.pth"
        )
        print(f"Epoch {epoch + 1} saved.")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader, desc="Testing"
        ):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total

    results = {
        "epochs": epochs,
        "classes": train_data.classes,
        "history": history,
        "test_accuracy": test_accuracy,
    }
    (root / "training_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    Image.fromarray(test_data.data[0]).save(
        root / "test_image.png"
    )
    actual_class = test_data.classes[test_data.targets[0]]

    print(f"Test accuracy: {test_accuracy:.2f}%")
    print("Model saved: cifar10_cnn.pth")
    print("Training results saved: training_results.json")
    print("API test image saved: test_image.png")
    print("Actual class of test image:", actual_class)
    print("Training and evaluation complete.")


if __name__ == "__main__":
    main()
