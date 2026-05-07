from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from dataset import create_dataloaders
from model import CLASS_NAMES
from model_factory import create_model


DATA_DIR = Path("data/Data")
MODEL_DIR = Path("models")
MODEL_NAME = "densenet121"
BEST_MODEL_PATH = MODEL_DIR / f"best_{MODEL_NAME}.pth"

NUM_CLASSES = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            predictions = outputs.argmax(dim=1)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == labels).sum().item()
            running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_accuracy = running_correct / running_total
    return epoch_loss, epoch_accuracy


def save_checkpoint(model: nn.Module, epoch: int, val_loss: float, val_accuracy: float) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        },
        BEST_MODEL_PATH,
    )


def train() -> None:
    device = get_device()
    print(f"Using device: {device}")

    dataloaders = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = run_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_one_epoch(
            model=model,
            dataloader=dataloaders["valid"],
            criterion=criterion,
            device=device,
        )

        improved = val_accuracy > best_val_accuracy or (
            val_accuracy == best_val_accuracy and val_loss < best_val_loss
        )
        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            save_checkpoint(model, epoch, val_loss, val_accuracy)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f} | "
            f"valid loss: {val_loss:.4f}, valid acc: {val_accuracy:.4f}"
            f"{' | saved best' if improved else ''}"
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
