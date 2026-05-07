from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dataset import create_dataloaders
from model import CLASS_NAMES, create_model


DATA_DIR = Path("data/Data")
MODEL_PATH = Path("models/best_resnet50.pth")
REPORT_PATH = Path("reports/test_evaluation.txt")

BATCH_SIZE = 16
NUM_CLASSES = len(CLASS_NAMES)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: nn.Module, model_path: Path, device: torch.device) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

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

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    test_loss = running_loss / running_total
    test_accuracy = running_correct / running_total
    return test_loss, test_accuracy, all_labels, all_predictions


def format_confusion_matrix(matrix, class_names: tuple[str, ...]) -> str:
    header = ["actual \\ predicted", *class_names]
    rows = [header]
    for class_name, row in zip(class_names, matrix):
        rows.append([class_name, *[str(value) for value in row]])

    column_widths = [
        max(len(str(row[column_index])) for row in rows)
        for column_index in range(len(header))
    ]
    return "\n".join(
        "  ".join(
            str(value).ljust(column_widths[column_index])
            for column_index, value in enumerate(row)
        )
        for row in rows
    )


def compute_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual, predicted in zip(labels, predictions):
        matrix[actual][predicted] += 1
    return matrix


def build_classification_report(
    conf_matrix: list[list[int]],
    class_names: tuple[str, ...],
) -> str:
    rows = []
    total_support = sum(sum(row) for row in conf_matrix)
    total_correct = sum(conf_matrix[index][index] for index in range(len(class_names)))

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for index, class_name in enumerate(class_names):
        true_positive = conf_matrix[index][index]
        false_positive = sum(row[index] for row in conf_matrix) - true_positive
        false_negative = sum(conf_matrix[index]) - true_positive
        support = sum(conf_matrix[index])

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
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        rows.append((class_name, precision, recall, f1, support))

    accuracy = total_correct / total_support if total_support > 0 else 0.0
    macro_avg = (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1_scores) / len(f1_scores),
        total_support,
    )
    weighted_avg = (
        sum(score * support for score, support in zip(precisions, supports)) / total_support,
        sum(score * support for score, support in zip(recalls, supports)) / total_support,
        sum(score * support for score, support in zip(f1_scores, supports)) / total_support,
        total_support,
    )

    lines = [f"{'':>28} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}", ""]
    for class_name, precision, recall, f1, support in rows:
        lines.append(f"{class_name:>28} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}")

    lines.extend(
        [
            "",
            f"{'accuracy':>28} {'':>10} {'':>10} {accuracy:>10.2f} {total_support:>10}",
            f"{'macro avg':>28} {macro_avg[0]:>10.2f} {macro_avg[1]:>10.2f} {macro_avg[2]:>10.2f} {macro_avg[3]:>10}",
            f"{'weighted avg':>28} {weighted_avg[0]:>10.2f} {weighted_avg[1]:>10.2f} {weighted_avg[2]:>10.2f} {weighted_avg[3]:>10}",
        ]
    )
    return "\n".join(lines)


def save_report(
    report_path: Path,
    test_loss: float,
    test_accuracy: float,
    class_report: str,
    conf_matrix,
    class_names: tuple[str, ...],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "Test Evaluation\n"
        "===============\n\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n\n"
        "Classification Report\n"
        "---------------------\n"
        f"{class_report}\n\n"
        "Confusion Matrix\n"
        "----------------\n"
        f"{format_confusion_matrix(conf_matrix, class_names)}\n"
    )
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    dataloaders = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = load_checkpoint(model=model, model_path=MODEL_PATH, device=device)
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, labels, predictions = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )

    conf_matrix = compute_confusion_matrix(
        labels=labels,
        predictions=predictions,
        num_classes=len(class_names),
    )
    class_report = build_classification_report(
        conf_matrix=conf_matrix,
        class_names=class_names,
    )

    save_report(
        report_path=REPORT_PATH,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        class_report=class_report,
        conf_matrix=conf_matrix,
        class_names=class_names,
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved evaluation report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
