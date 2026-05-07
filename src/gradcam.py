from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from dataset import LungTumorDataset, create_dataset
from model import CLASS_NAMES, create_model


DATA_DIR = Path("data/Data")
MODEL_PATH = Path("models/best_resnet50.pth")
OUTPUT_DIR = Path("reports/gradcam")

BATCH_SIZE = 1
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (224, 224)
OVERLAY_ALPHA = 0.45


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, tuple[str, ...]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    return model, class_names


def create_test_dataloader(data_dir: Path) -> DataLoader:
    dataset = create_dataset(data_dir=data_dir, split="test")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        predicted_class = int(output.argmax(dim=1).item())
        target_class = predicted_class if class_index is None else class_index

        score = output[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = normalize_heatmap(cam)
        return cam, predicted_class

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_value) / (max_value - min_value)


def apply_colormap(heatmap: np.ndarray) -> Image.Image:
    heatmap_uint8 = np.uint8(255 * heatmap)
    red = np.clip(1.5 * heatmap_uint8 - 64, 0, 255)
    green = np.clip(1.5 * (255 - np.abs(heatmap_uint8 - 128)), 0, 255)
    blue = np.clip(384 - 1.5 * heatmap_uint8, 0, 255)
    colored = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")


def load_original_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(IMAGE_SIZE)


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
    colored_heatmap = apply_colormap(heatmap)
    return Image.blend(original, colored_heatmap, alpha=OVERLAY_ALPHA)


def add_label(image: Image.Image, label: str) -> Image.Image:
    labeled = Image.new("RGB", (image.width, image.height + 28), "white")
    labeled.paste(image, (0, 28))
    draw = ImageDraw.Draw(labeled)
    draw.text((8, 7), label, fill="black")
    return labeled


def make_side_by_side(original: Image.Image, overlay: Image.Image, label: str) -> Image.Image:
    left = add_label(original, "Original")
    right = add_label(overlay, label)
    canvas = Image.new("RGB", (left.width + right.width, left.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def save_gradcam_image(
    output_dir: Path,
    index: int,
    true_label: int,
    predicted_label: int,
    class_names: tuple[str, ...],
    original_path: Path,
    heatmap: np.ndarray,
) -> Path:
    original = load_original_image(original_path)
    overlay = overlay_heatmap(original, heatmap)
    label = f"Grad-CAM: predicted {class_names[predicted_label]}"
    side_by_side = make_side_by_side(original, overlay, label)

    filename = (
        f"{index:04d}_true-{safe_name(class_names[true_label])}"
        f"_pred-{safe_name(class_names[predicted_label])}.png"
    )
    output_path = output_dir / filename
    side_by_side.save(output_path)
    return output_path


def generate_gradcam_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: tuple[str, ...],
    output_dir: Path,
    device: torch.device,
    max_images: int | None = None,
) -> int:
    dataset = dataloader.dataset
    if not isinstance(dataset, LungTumorDataset):
        raise TypeError("Grad-CAM expects a LungTumorDataset with image paths in samples.")

    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam = GradCAM(model=model, target_layer=model.layer4[-1])

    saved_count = 0
    try:
        for index, (image, label) in enumerate(dataloader):
            if max_images is not None and saved_count >= max_images:
                break

            image = image.to(device)
            true_label = int(label.item())
            heatmap, predicted_label = gradcam.generate(image=image)
            save_gradcam_image(
                output_dir=output_dir,
                index=index,
                true_label=true_label,
                predicted_label=predicted_label,
                class_names=class_names,
                original_path=dataset.samples[index].path,
                heatmap=heatmap,
            )
            saved_count += 1
    finally:
        gradcam.close()

    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM images for the test dataset.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for quick checks. By default, all test images are processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model, class_names = load_model(model_path=args.model_path, device=device)
    test_dataloader = create_test_dataloader(data_dir=args.data_dir)
    saved_count = generate_gradcam_outputs(
        model=model,
        dataloader=test_dataloader,
        class_names=class_names,
        output_dir=args.output_dir,
        device=device,
        max_images=args.max_images,
    )
    print(f"Saved {saved_count} Grad-CAM images to: {args.output_dir}")


if __name__ == "__main__":
    main()
