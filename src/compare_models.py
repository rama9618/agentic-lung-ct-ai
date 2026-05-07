from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from dataset import create_dataloaders
from model import CLASS_NAMES
from model_factory import create_model


DATA_DIR = Path("data/Data")
REPORTS_DIR = Path("reports")
CSV_PATH = REPORTS_DIR / "model_comparison.csv"
TXT_PATH = REPORTS_DIR / "model_comparison.txt"
BEST_MODEL_PATH = REPORTS_DIR / "best_model.txt"

BATCH_SIZE = 16
NUM_CLASSES = len(CLASS_NAMES)
MODEL_CHECKPOINTS = {
    "resnet50": Path("models/best_resnet50.pth"),
    "densenet121": Path("models/best_densenet121.pth"),
    "efficientnet_b0": Path("models/best_efficientnet_b0.pth"),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == labels).sum().item()
            running_total += batch_size
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    return (
        running_loss / running_total,
        running_correct / running_total,
        labels_all,
        predictions_all,
    )


def compute_weighted_metrics(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> tuple[float, float, float]:
    total_support = len(labels)
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for class_index in range(num_classes):
        true_positive = sum(
            actual == class_index and predicted == class_index
            for actual, predicted in zip(labels, predictions)
        )
        false_positive = sum(
            actual != class_index and predicted == class_index
            for actual, predicted in zip(labels, predictions)
        )
        false_negative = sum(
            actual == class_index and predicted != class_index
            for actual, predicted in zip(labels, predictions)
        )
        support = sum(actual == class_index for actual in labels)

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1_score * support

    if total_support == 0:
        return 0.0, 0.0, 0.0
    return (
        weighted_precision / total_support,
        weighted_recall / total_support,
        weighted_f1 / total_support,
    )


def load_checkpoint_model(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def save_results(rows: list[dict[str, str | float]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "checkpoint",
        "status",
        "test_loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["Model Comparison", "================", ""]
    for row in rows:
        lines.extend(
            [
                f"Model: {row['model_name']}",
                f"Checkpoint: {row['checkpoint']}",
                f"Status: {row['status']}",
                f"Test loss: {row['test_loss']}",
                f"Accuracy: {row['accuracy']}",
                f"Precision: {row['precision']}",
                f"Recall: {row['recall']}",
                f"Weighted F1-score: {row['f1_score']}",
                "",
            ]
        )
    TXT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    dataloaders = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, str | float]] = []

    for model_name, checkpoint_path in MODEL_CHECKPOINTS.items():
        if not checkpoint_path.exists():
            rows.append(
                {
                    "model_name": model_name,
                    "checkpoint": str(checkpoint_path),
                    "status": "missing",
                    "test_loss": "",
                    "accuracy": "",
                    "precision": "",
                    "recall": "",
                    "f1_score": "",
                }
            )
            continue

        model = load_checkpoint_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        test_loss, accuracy, labels, predictions = evaluate_model(
            model=model,
            dataloader=dataloaders["test"],
            criterion=criterion,
            device=device,
        )
        precision, recall, f1_score = compute_weighted_metrics(
            labels=labels,
            predictions=predictions,
            num_classes=NUM_CLASSES,
        )
        rows.append(
            {
                "model_name": model_name,
                "checkpoint": str(checkpoint_path),
                "status": "evaluated",
                "test_loss": round(test_loss, 6),
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1_score, 6),
            }
        )

    save_results(rows)
    evaluated_rows = [row for row in rows if row["status"] == "evaluated"]
    if not evaluated_rows:
        raise RuntimeError("No saved model checkpoints were found to evaluate.")

    best_row = max(evaluated_rows, key=lambda row: float(row["f1_score"]))
    best_model = str(best_row["model_name"])
    best_f1 = float(best_row["f1_score"])
    BEST_MODEL_PATH.write_text(best_model, encoding="utf-8")

    print(f"Best model: {best_model}")
    print(f"Best F1-score: {best_f1:.6f}")


if __name__ == "__main__":
    main()
